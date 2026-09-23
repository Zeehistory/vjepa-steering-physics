

#!/usr/bin/env python
"""POSITION-CANONICALIZED command-only accel operator (Step 2 "Bravo", canonicalization push).

After base/quad/appc all plateaued at ~16deg held-out (vs 10.07 full_delta ceiling), the residual was
diagnosed as the HIGH-RANK, POSITION-DEPENDENT spatial path footprint: each scene's ball starts at a
different grid cell (pos0 is shared WITHIN a scene but varies ACROSS scenes), so the SAME acceleration
edit lands on different tokens per scene. A single global subspace/operator then wastes rank spanning
translated copies of one footprint ("token-misalignment cost") and cannot place the edit.

Fix = canonicalize by START POSITION: roll each scene's Delta H so the ball's start cell -> grid centre
(``velocity_ops.canon_shift`` + ``roll_layer``; all ranks in a scene share pos0 so one shift per scene).
In this start-aligned frame the footprint variance collapses -> a tighter low-rank basis U_canon and a
command->U_canon map. At steer time we synthesize the edit in the centred frame and UN-ROLL it back to the
test scene's own start cell (inverse shift) before applying. This is distinct from the HEADING-binning that
was a clean negative for velocity -- here we align translation, not orientation.

Self-contained (no accel_subspace dependency): pass 0 builds U_canon from canonicalized Delta H (early-stops
once the global buffer is full), pass 1 fits W_U_canon: command_features(13) -> U_canon coords. Saves
``global_basis_canon_L*.npy`` + ``cmd_Wu_canon_L*.npy``. Consumed by steer_accel2d.py --features canon.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_canon.py \
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


def inv_shift(sh: tuple[int, int]) -> tuple[int, int]:
    return (-sh[0], -sh[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=16, help="canon U-subspace rank to synthesize into")
    p.add_argument("--max_global_pairs", type=int, default=800, help="pairs used to build U_canon (pass 0)")
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    order = sorted(tr_scenes)
    print(f"[a-canon] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; "
          f"P={P} KU={KU}", flush=True)

    # ---- pass 0: build U_canon from canonicalized Delta H (early-stop once buffers full) -----------
    gbuf = {L: [] for L in layers}
    for s in order:
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]
        grid = tuple(int(x) for x in sa["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                if len(gbuf[L]) < args.max_global_pairs:
                    gbuf[L].append(vo.roll_layer(dH, grid, sh).astype(np.float32))
        del Ha; gc.collect()
        if all(len(gbuf[L]) >= args.max_global_pairs for L in layers):
            break
    U = {}
    for L in layers:
        Xg = np.stack(gbuf[L], 0)
        basis, _ = vo.pca_gram(Xg, k=KU)  # (KU, D)
        U[L] = basis.astype(np.float64)
        np.save(art / f"global_basis_canon_L{L}.npy", basis.astype(np.float32))
        del Xg
    del gbuf; gc.collect()
    print(f"[a-canon] U_canon built (KU={KU}) from <= {args.max_global_pairs} pairs + saved "
          f"global_basis_canon", flush=True)

    # ---- pass 1: fit command_features -> U_canon coords (canonicalized target) ---------------------
    wu = {L: vo.LinearLS(P, KU, args.ridge) for L in layers}
    n = 0
    for s in order:
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb).reshape(1, P)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                cdH = vo.roll_layer(dH, grid, sh)  # canonicalize to centre frame
                wu[L].add(phi, (U[L] @ cdH).reshape(1, KU))
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[a-canon]   trained {n}/{len(tr_scenes)} scenes", flush=True)

    W_U = {L: wu[L].solve().astype(np.float32) for L in layers}
    del wu; gc.collect()

    # ---- held-out gate: cos(pred canon edit, true canon dH) + U_canon retain ----------------------
    gate = {L: {"cmdU_cos": [], "coord_cos": [], "Uretain_cos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb)
            for L in layers:
                cdH = vo.roll_layer(vo.layer_flat(sb["layers"][L]) - Ha[L], grid, sh)
                ctrue = U[L] @ cdH
                cpred = phi @ W_U[L].astype(np.float64)
                gate[L]["coord_cos"].append(vo.cosine(cpred, ctrue))
                gate[L]["cmdU_cos"].append(vo.cosine(cpred @ U[L], cdH))
                gate[L]["Uretain_cos"].append(vo.cosine(ctrue @ U[L], cdH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "KU": KU, "ridge": args.ridge, "quantity": "accel",
               "features": "canon", "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes),
               "per_layer": {}}
    for L in layers:
        row = {k: round(float(np.mean(v)), 4) for k, v in gate[L].items()}
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Wu_canon_L{L}.npy", W_U[L])
        print(f"[a-canon] L{L}: canon cmd_U cos={row['cmdU_cos']:.3f} (coord cos={row['coord_cos']:.3f}, "
              f"U_canon retain={row['Uretain_cos']:.3f}) | base non-canon U16 retain was ~0.55", flush=True)
    summary["artifacts"] = {"W_U": "cmd_Wu_canon_L*.npy (P,KU); edit_centred = coords @ U_canon, "
                                   "then roll back by -canon_shift(start)",
                            "global_basis": "global_basis_canon_L*.npy (KU,D)"}
    (art / "cmd_operator_meta_canon.json").write_text(json.dumps(summary, indent=2))
    print(f"[a-canon] saved W_U_canon + global_basis_canon + meta -> {art} (KU={KU})", flush=True)


if __name__ == "__main__":
    main()
