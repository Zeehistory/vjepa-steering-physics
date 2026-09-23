

#!/usr/bin/env python
"""v0-AWARE TRAJECTORY-canonicalized command-only accel operator (Step 2 "Bravo", trajcanon push).

canon (start-only roll) closed part of the position gap but only centres FRAME 0; because v0 varies across
scenes, the shared v0 ramp carries the ball to a scene-dependent cell by late frames -- exactly where the
acceleration curvature signal (~t^2) is strongest -- so the footprint stays misaligned there. The probe_ax
decode (48.5deg ~ random) confirmed the decoder renders accel from the SPATIAL FOOTPRINT, not the global
temporal-pool axis, so finishing the placement job is the right lever.

Fix = roll EACH latent frame t by the shift that centres the ball's actual position at that frame
(``clip_frame_positions`` -> ``traj_shifts`` -> ``roll_layer_frames``). All ranks in a scene share pos0+v0
so one per-frame shift schedule per scene (from the reference clip). In this trajectory-aligned frame the
curvature footprint variance collapses across scenes -> tighter U_trajcanon + command->U map. At steer we
synthesize in the aligned frame and UN-ROLL per frame (negate the shift schedule).

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_trajcanon.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --layers 6,12,18,23 --artifacts_dir .../subspace --ku 16 --ridge 1.0
"""
from __future__ import annotations

# --- repo-root shim: make ``src`` importable however this script is invoked ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

P = vo.COMMAND_FEATURE_DIM  # 13


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=16)
    p.add_argument("--max_global_pairs", type=int, default=800)
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    trs, tes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        trs = {s: trs[s] for s in sorted(trs)[: args.max_scenes]}
        tes = {s: tes[s] for s in sorted(tes)[: args.max_scenes]}
    order = sorted(trs)
    print(f"[a-trajcanon] train {len(trs)} scenes, test {len(tes)}; layers={layers}; P={P} KU={KU}",
          flush=True)

    def scene_shifts(sa, grid):
        return vo.traj_shifts(vo.clip_frame_positions(sa, grid[0]), grid)

    # ---- pass 0: build U_trajcanon from per-frame-aligned Delta H ----------------------------------
    gbuf = {L: [] for L in layers}
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]
        grid = tuple(int(x) for x in sa["grid"]); sh = scene_shifts(sa, grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[trs[s][b]]
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                if len(gbuf[L]) < args.max_global_pairs:
                    gbuf[L].append(vo.roll_layer_frames(dH, grid, sh).astype(np.float32))
        del Ha; gc.collect()
        if all(len(gbuf[L]) >= args.max_global_pairs for L in layers):
            break
    U = {}
    for L in layers:
        basis, _ = vo.pca_gram(np.stack(gbuf[L], 0), k=KU)
        U[L] = basis.astype(np.float64)
        np.save(art / f"global_basis_trajcanon_L{L}.npy", basis.astype(np.float32))
    del gbuf; gc.collect()
    print(f"[a-trajcanon] U_trajcanon built (KU={KU}) + saved", flush=True)

    # ---- pass 1: fit command_features -> U_trajcanon coords -----------------------------------------
    wu = {L: vo.LinearLS(P, KU, args.ridge) for L in layers}
    n = 0
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"]); sh = scene_shifts(sa, grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[trs[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb).reshape(1, P)
            for L in layers:
                cdH = vo.roll_layer_frames(vo.layer_flat(sb["layers"][L]) - Ha[L], grid, sh)
                wu[L].add(phi, (U[L] @ cdH).reshape(1, KU))
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[a-trajcanon]   trained {n}/{len(trs)} scenes", flush=True)
    W_U = {L: wu[L].solve().astype(np.float32) for L in layers}
    del wu; gc.collect()

    # ---- held-out gate ------------------------------------------------------------------------------
    gate = {L: {"cmdU_cos": [], "coord_cos": [], "Uretain_cos": []} for L in layers}
    for s in sorted(tes):
        ranks = sorted(tes[s]); sa = te[tes[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"]); sh = scene_shifts(sa, grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[tes[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb)
            for L in layers:
                cdH = vo.roll_layer_frames(vo.layer_flat(sb["layers"][L]) - Ha[L], grid, sh)
                ctrue = U[L] @ cdH
                cpred = phi @ W_U[L].astype(np.float64)
                gate[L]["coord_cos"].append(vo.cosine(cpred, ctrue))
                gate[L]["cmdU_cos"].append(vo.cosine(cpred @ U[L], cdH))
                gate[L]["Uretain_cos"].append(vo.cosine(ctrue @ U[L], cdH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "KU": KU, "ridge": args.ridge, "quantity": "accel",
               "features": "trajcanon", "n_train_scenes": len(trs), "n_test_scenes": len(tes),
               "per_layer": {}}
    for L in layers:
        row = {k: round(float(np.mean(v)), 4) for k, v in gate[L].items()}
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Wu_trajcanon_L{L}.npy", W_U[L])
        print(f"[a-trajcanon] L{L}: cmd_U cos={row['cmdU_cos']:.3f} (coord={row['coord_cos']:.3f}, "
              f"U retain={row['Uretain_cos']:.3f}) | canon U retain was ~0.34-0.36", flush=True)
    (art / "cmd_operator_meta_trajcanon.json").write_text(json.dumps(summary, indent=2))
    print(f"[a-trajcanon] saved W_U_trajcanon + global_basis_trajcanon + meta -> {art}", flush=True)


if __name__ == "__main__":
    main()
