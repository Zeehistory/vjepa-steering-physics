

#!/usr/bin/env python
"""COMBINED position-canonicalized + appearance-conditioned accel operator (Step 2 "Bravo" finale).

The two winning levers peel DIFFERENT slices of the mixed-appearance residual: position-canonicalization
(canon, 14.46deg) aligns WHERE the edit lands; appearance conditioning (appc, 16.1 vs 16.88) supplies the
per-scene colour/bg modulation. This stacks them: fit

    [ command_features(a_a,a_b) (13)  ||  appc(H_a) (Ka) ]  ->  U_canon coordinates (KU)

i.e. the canon operator's canonicalized target with appc's appearance-augmented input. REUSES the appc
appearance basis (appc_mean/appc_basis, from fit_..._appc) and the canon subspace (global_basis_canon,
from fit_..._canon) already on disk, so this is a SINGLE streaming pass (fast). At steer time: appc read
from the reference H_a, synthesize coords, reconstruct in U_canon, then UN-ROLL to each scene's start cell.
Saves cmd_Wu_canon_appc_L*.npy. Consumed by steer_accel2d.py --features canon_appc.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_canon_appc.py \
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


def pooled(sample_layer) -> np.ndarray:
    return np.asarray(sample_layer, dtype=np.float64).mean(axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True,
                   help="must already contain appc_mean/appc_basis (from appc fit) + global_basis_canon "
                        "(from canon fit); outputs land here")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=16, help="canon U-subspace rank to synthesize into")
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir)

    # reuse the appearance basis + canon subspace already fit
    appc_mean = {L: np.load(art / f"appc_mean_L{L}.npy").astype(np.float64) for L in layers}
    appc_basis = {L: np.load(art / f"appc_basis_L{L}.npy").astype(np.float64) for L in layers}
    U = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64)[:KU] for L in layers}
    Ka = appc_basis[layers[0]].shape[0]
    Pa = P + Ka
    for L in layers:
        if U[L].shape[0] < KU:
            raise SystemExit(f"global_basis_canon_L{L} has {U[L].shape[0]} rows < ku={KU}")
        if appc_basis[L].shape[0] != Ka:
            raise SystemExit(f"appc_basis rank mismatch across layers ({L}: {appc_basis[L].shape[0]} != {Ka})")

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    print(f"[a-canon-appc] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; "
          f"P={P} Ka={Ka} Pa={Pa} KU={KU}", flush=True)

    def appc_feat(sa) -> dict:
        return {L: (pooled(sa["layers"][L]) - appc_mean[L]) @ appc_basis[L].T for L in layers}

    # single pass: fit [command || appc(H_a)] -> U_canon coords (canonicalized dH)
    wu = {L: vo.LinearLS(Pa, KU, args.ridge) for L in layers}
    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = appc_feat(sa)
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]; vb = vo.clip_acceleration(sb)
            cmd = vo.command_features(va, vb)
            for L in layers:
                phi = np.concatenate([cmd, ca[L]]).reshape(1, Pa)
                cdH = vo.roll_layer(vo.layer_flat(sb["layers"][L]) - Ha[L], grid, sh)
                wu[L].add(phi, (U[L] @ cdH).reshape(1, KU))
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[a-canon-appc]   trained {n}/{len(tr_scenes)} scenes", flush=True)

    W_U = {L: wu[L].solve().astype(np.float32) for L in layers}
    del wu; gc.collect()

    gate = {L: {"cmdU_cos": [], "coord_cos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = appc_feat(sa)
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; vb = vo.clip_acceleration(sb)
            cmd = vo.command_features(va, vb)
            for L in layers:
                cdH = vo.roll_layer(vo.layer_flat(sb["layers"][L]) - Ha[L], grid, sh)
                ctrue = U[L] @ cdH
                cpred = np.concatenate([cmd, ca[L]]) @ W_U[L].astype(np.float64)
                gate[L]["coord_cos"].append(vo.cosine(cpred, ctrue))
                gate[L]["cmdU_cos"].append(vo.cosine(cpred @ U[L], cdH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "Ka": Ka, "Pa": Pa, "KU": KU, "ridge": args.ridge,
               "quantity": "accel", "features": "canon_appc",
               "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes), "per_layer": {}}
    for L in layers:
        row = {k: round(float(np.mean(v)), 4) for k, v in gate[L].items()}
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Wu_canon_appc_L{L}.npy", W_U[L])
        print(f"[a-canon-appc] L{L}: canon+appc cmd_U cos={row['cmdU_cos']:.3f} "
              f"(coord cos={row['coord_cos']:.3f})", flush=True)
    summary["artifacts"] = {"W_U": "cmd_Wu_canon_appc_L*.npy (P+Ka,KU); edit_centred = "
                                   "[cmd||appc(H_a)] @ W_U @ U_canon, then roll back by -canon_shift(start)"}
    (art / "cmd_operator_meta_canon_appc.json").write_text(json.dumps(summary, indent=2))
    print(f"[a-canon-appc] saved W_U_canon_appc + meta -> {art} (KU={KU} Ka={Ka})", flush=True)


if __name__ == "__main__":
    main()
