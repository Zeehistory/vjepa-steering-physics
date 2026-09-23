

#!/usr/bin/env python
"""Fit the BILINEAR reference-conditioned accel operator (Step 2 "Bravo", bilinear push).

The 2nd-order per-frame translation field (transfield screen) was a clean NEGATIVE: dH is not a low-order
position-conditioned rigid ball translation (recon cos BELOW the base command floor on deep layers). But the
bilinear screen showed the winning lever is GLOBAL reference conditioning: predicting the accel edit from
[command || coords_a || command (x) coords_a] where coords_a = U @ H_a is the reference clip's projection on
the accel subspace. This beats the command-only floor at every layer (recon cos 0.27-0.31 -> 0.34-0.46).
Mechanism: the same Delta-accel writes a per-scene-different dH because pos0/v0/appearance (all in H_a)
modulate it; conditioning MULTIPLICATIVELY on the reference projection expresses that modulation, which a
linear command map cannot. Still H_b-free (coords_a is read from the reference clip at steer time).

Feature (dim 237) = [ command_features(13) || coords_a(KU) || vec(command (x) coords_a)(13*KU) ], fit to the
accel-subspace target coords y = U @ dH, then the edit = (feat @ W) @ U. Reuses the SAME U built from dH so
the number is directly comparable to the base cmd_U operator (16.88). Saves cmd_Wbilin_L{L}.npy +
global_basis_bilin_L{L}.npy + cmd_operator_meta_bilinear.json.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_bilinear.py --train_dir .../train/vjepa2_large \
        --test_dir .../test/vjepa2_large --layers 6,12,18,23 --ku 16 --ridge 1.0 --artifacts_dir .../subspace
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


def bilin_feat(phi: np.ndarray, ca: np.ndarray) -> np.ndarray:
    """[command(13) || coords_a(KU) || vec(command (x) coords_a)(13*KU)] -- byte-identical in fit & steer."""
    return np.concatenate([phi, ca, np.outer(phi, ca).reshape(-1)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=16)
    p.add_argument("--max_global_pairs", type=int, default=800)
    args = p.parse_args()

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    trs, tes = vo.group_scenes(tr), vo.group_scenes(te)
    order = sorted(trs)
    F = P + KU + P * KU
    print(f"[bilinfit] train {len(trs)} scenes, test {len(tes)}; layers={layers}; KU={KU} F={F}", flush=True)

    # ---- pass 0: build accel subspace U from dH (same recipe/basis as the base cmd_U operator) --------
    gbuf = {L: [] for L in layers}
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[trs[s][b]]
            for L in layers:
                if len(gbuf[L]) < args.max_global_pairs:
                    gbuf[L].append((vo.layer_flat(sb["layers"][L]) - Ha[L]).astype(np.float32))
        del Ha; gc.collect()
        if all(len(gbuf[L]) >= args.max_global_pairs for L in layers):
            break
    U = {}
    for L in layers:
        basis, _ = vo.pca_gram(np.stack(gbuf[L], 0), k=KU)
        U[L] = basis.astype(np.float64)
        np.save(art / f"global_basis_bilin_L{L}.npy", basis.astype(np.float32))
    del gbuf; gc.collect()
    print("[bilinfit] U built + saved", flush=True)

    # ---- pass 1: fit [command || coords_a || outer] -> U coords ---------------------------------------
    wb = {L: vo.LinearLS(F, KU, args.ridge) for L in layers}
    n = 0
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = {L: (U[L] @ Ha[L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[trs[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb)
            for L in layers:
                y = (U[L] @ (vo.layer_flat(sb["layers"][L]) - Ha[L])).reshape(1, KU)
                wb[L].add(bilin_feat(phi, ca[L]).reshape(1, F), y)
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[bilinfit]   fit {n}/{len(trs)}", flush=True)
    W = {L: wb[L].solve().astype(np.float32) for L in layers}
    for L in layers:
        np.save(art / f"cmd_Wbilin_L{L}.npy", W[L])
    del wb; gc.collect()

    # ---- held-out gate: recon cos of edit vs true dH -------------------------------------------------
    gate = {L: [] for L in layers}
    for s in sorted(tes):
        ranks = sorted(tes[s]); sa = te[tes[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = {L: (U[L] @ Ha[L]) for L in layers}
        for b in ranks[1:]:
            sb = te[tes[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                pred = (bilin_feat(phi, ca[L]) @ W[L].astype(np.float64)) @ U[L]
                gate[L].append(vo.cosine(pred, dH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "KU": KU, "F": F, "ridge": args.ridge, "quantity": "accel",
               "features": "bilinear", "n_train": len(trs), "n_test": len(tes),
               "heldout_recon_cos": {str(L): round(float(np.mean(v)), 4) for L, v in gate.items()}}
    (art / "cmd_operator_meta_bilinear.json").write_text(json.dumps(summary, indent=2))
    print("[bilinfit] HELD-OUT recon cos:", flush=True)
    for L in layers:
        print(f"  L{L}: {summary['heldout_recon_cos'][str(L)]:.3f}", flush=True)
    print(f"[bilinfit] saved cmd_Wbilin + global_basis_bilin + meta -> {art}", flush=True)


if __name__ == "__main__":
    main()
