

#!/usr/bin/env python
"""Cheap latent-only screen: can a BILINEAR command(x)reference operator reconstruct the true accel delta
dH=H_b-H_a better than the linear command-only operator (recon cos ~0.28, the ~14.5deg steer floor)?

Every operator tried so far (base/quad/appc/canon/canon_appc/trajcanon) is LINEAR in the command features.
But the mechanism is that per-scene state (appearance, v0, position -- all readable from the reference clip
H_a) MODULATES how the same Delta-accel writes into the latent. Modulation is MULTIPLICATIVE, which a linear
map on [command || H_a] cannot express; it needs the OUTER PRODUCT command (x) reference. This screen fits,
per layer, three ridge operators to the SAME accel-subspace target coords y=U@dH and reports held-out recon
cosine (cos of predicted-edit vs true dH in full latent space):
    base   : command_features(13)                                  -> reproduces the ~0.28 floor
    lin    : [command || coords_a]   (coords_a = U @ H_a, KU dims)  -> linear reference conditioning
    bilin  : [command || coords_a || vec(command (x) coords_a)]     -> multiplicative modulation
DECISIVE: bilin recon cos >> 0.28 -> a bilinear operator can beat 14.5, build the decode. bilin ~ 0.28 ->
14.46 is the linear+bilinear command-only ceiling; the residual is genuinely off-subspace footprint.

    python experiments/threads/acceleration/07_eval/accel_bilinear_screen.py --train_dir .../train/vjepa2_large \
        --test_dir .../test/vjepa2_large --layers 6,12,18,23 --ku 16 --ridge 1.0 --out .../bilinear_screen.json
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
    p.add_argument("--out", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=16)
    p.add_argument("--max_global_pairs", type=int, default=800)
    args = p.parse_args()

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    trs, tes = vo.group_scenes(tr), vo.group_scenes(te)
    order = sorted(trs)
    print(f"[bilin] train {len(trs)} scenes, test {len(tes)}; layers={layers}; P={P} KU={KU}", flush=True)

    # ---- pass 0: build accel subspace U from dH (same recipe as the base cmd operator) --------------
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
    del gbuf; gc.collect()
    print("[bilin] U built", flush=True)

    def feats(phi, ca):
        """phi: (P,) command features; ca: (KU,) reference coords. Returns the 3 feature vectors."""
        outer = np.outer(phi, ca).reshape(-1)  # P*KU
        return {
            "base": phi,
            "lin": np.concatenate([phi, ca]),
            "bilin": np.concatenate([phi, ca, outer]),
        }

    dims = {"base": P, "lin": P + KU, "bilin": P + KU + P * KU}
    models = {m: {L: vo.LinearLS(dims[m], KU, args.ridge) for L in layers} for m in dims}

    # ---- pass 1: fit all three operators -----------------------------------------------------------
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
                fs = feats(phi, ca[L])
                for m in dims:
                    models[m][L].add(fs[m].reshape(1, dims[m]), y)
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[bilin]   fit {n}/{len(trs)}", flush=True)
    W = {m: {L: models[m][L].solve() for L in layers} for m in dims}
    del models; gc.collect()

    # ---- held-out recon cosine ---------------------------------------------------------------------
    acc = {m: {L: [] for L in layers} for m in dims}
    for s in sorted(tes):
        ranks = sorted(tes[s]); sa = te[tes[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = {L: (U[L] @ Ha[L]) for L in layers}
        for b in ranks[1:]:
            sb = te[tes[s][b]]; vb = vo.clip_acceleration(sb)
            phi = vo.command_features(va, vb)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                fs = feats(phi, ca[L])
                for m in dims:
                    pred_coords = fs[m] @ W[m][L]           # (KU,)
                    acc[m][L].append(vo.cosine(pred_coords @ U[L], dH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "KU": KU, "ridge": args.ridge,
               "metric": "held-out recon cos of predicted edit vs true dH (full latent)",
               "n_train": len(trs), "n_test": len(tes), "recon_cos": {}}
    for m in dims:
        summary["recon_cos"][m] = {str(L): round(float(np.mean(acc[m][L])), 4) for L in layers}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("[bilin] RECON COS (held-out, full-latent):", flush=True)
    for L in layers:
        print(f"  L{L}: base={summary['recon_cos']['base'][str(L)]:.3f}  "
              f"lin={summary['recon_cos']['lin'][str(L)]:.3f}  "
              f"bilin={summary['recon_cos']['bilin'][str(L)]:.3f}", flush=True)
    print(f"[bilin] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
