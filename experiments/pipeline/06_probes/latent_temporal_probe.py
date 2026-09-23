

#!/usr/bin/env python
"""EXP 2 (user 2026-07-07): temporal-derivative acceleration probes on the VJEPA latent.

The latent for a clip is H in R^{8 x 256 x 1024}: 8 TEMPORAL tokens (tubelet size 2 -> token t pools
video frames 2t, 2t+1), each a 16x16=256 spatial grid of 1024-d features. Write the per-token slab
Z_t = H[t] in R^{256 x 1024}. Acceleration is SECOND-ORDER in time, so the hypothesis (user) is that it
reads out cleanest from the discrete second difference

    Z_t   (raw slab)              -- position-ish, should read accel only weakly / growing with t
    DZ_t  = Z_{t+1} - Z_t         -- velocity-ish (first difference)
    D2Z_t = Z_{t+1} - 2Z_t + Z_{t-1}   -- curvature/acceleration-ish (second difference)

We spatial-pool each slab to a 1024-d per-frame descriptor (validated: pooled latent still encodes
velocity R2=0.998 and accel R2~0.99 -- a positional shortcut would die under pooling, this does not),
then fit a leakage-free ridge probe pooled-feature(1024) -> a=(a_x,a_y) for EACH (layer, order, t) and
report held-out test R2 / angle / |a|-R2. Sweeps t to find which temporal token carries accel best, and
compares the three derivative orders. A y-shuffle control gives the R2 floor.

    python experiments/pipeline/06_probes/latent_temporal_probe.py --config configs/train/moving_ball_scene_decoder.yaml \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --output_dir .../analysis/moving_ball_accel2d_mixed/temporal_probe
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
import json
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config


def _evict(ds: LatentDataset, keep: int = 2) -> None:
    """Drop cached shards so a full pass does not pin all shards in RAM (cache never self-evicts)."""
    if len(ds._shard_cache) > keep:
        ds._shard_cache.clear()


def pooled_frames(sample, layers) -> dict[int, np.ndarray]:
    """{L: (T, D)} spatial-mean of each temporal slab. grid=(T,H,W); tokens are temporal-major."""
    grid = tuple(int(x) for x in sample["grid"])
    T, H, W = grid
    out = {}
    for L in layers:
        arr = np.asarray(sample["layers"][L], dtype=np.float64)  # (T*H*W, D)
        D = arr.shape[-1]
        out[L] = arr.reshape(T, H * W, D).mean(axis=1)           # (T, D)
    return out


def collect(ds, layers, max_scenes=None):
    """Return {L: (N, T, D)} pooled per-frame features + accel (N,2), grouped by scene for held-out split."""
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)
    if max_scenes:
        scene_ids = scene_ids[:max_scenes]
    feats = {L: [] for L in layers}
    accel, sids = [], []
    n = 0
    for s in scene_ids:
        for rank, idx in sorted(scenes[s].items()):
            sample = ds[idx]
            pf = pooled_frames(sample, layers)
            for L in layers:
                feats[L].append(pf[L])
            accel.append(vo.clip_acceleration(sample))
            sids.append(s)
            n += 1
            if n % 128 == 0:
                _evict(ds)
    feats = {L: np.stack(feats[L]) for L in layers}   # (N, T, D)
    return feats, np.asarray(accel), np.asarray(sids)


def ridge_fit(X, Y, lam):
    """Closed-form ridge with standardized X. Returns (W, mu, sd, Ybar) for predict()."""
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Ybar = Y.mean(0)
    Yc = Y - Ybar
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    W = np.linalg.solve(A, Xs.T @ Yc)                 # (D, 2)
    return W, mu, sd, Ybar


def ridge_pred(X, W, mu, sd, Ybar):
    return ((X - mu) / sd) @ W + Ybar


def score(Yhat, Y):
    """Per-component R2, combined R2, angle err (deg), |a| R2 + magnitude corr."""
    ss_res = ((Yhat - Y) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    r2_all = 1.0 - ss_res.sum() / ss_tot.sum()
    cos = (Yhat * Y).sum(1) / (np.linalg.norm(Yhat, axis=1) * np.linalg.norm(Y, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    md, mt = np.linalg.norm(Yhat, axis=1), np.linalg.norm(Y, axis=1)
    ss_res_m = ((md - mt) ** 2).sum(); ss_tot_m = ((mt - mt.mean()) ** 2).sum() + 1e-30
    return {"r2_x": round(float(r2[0]), 4), "r2_y": round(float(r2[1]), 4),
            "r2_all": round(float(r2_all), 4), "angle_deg": round(float(ang.mean()), 2),
            "mag_r2": round(float(1 - ss_res_m / ss_tot_m), 4),
            "mag_corr": round(float(np.corrcoef(md, mt)[0, 1]), 4)}


def derivative(feat, order):
    """feat (N,T,D) -> {t: (N,D)} for the requested derivative order (0=Z, 1=DZ, 2=D2Z)."""
    N, T, D = feat.shape
    out = {}
    if order == 0:
        for t in range(T):
            out[t] = feat[:, t, :]
    elif order == 1:
        for t in range(T - 1):
            out[t] = feat[:, t + 1, :] - feat[:, t, :]
    else:
        for t in range(1, T - 1):
            out[t] = feat[:, t + 1, :] - 2 * feat[:, t, :] + feat[:, t - 1, :]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--max_train_scenes", type=int, default=0)
    p.add_argument("--max_test_scenes", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config, [])
    layers = list(cfg.encoder.layers)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[probe] layers={layers} lam={args.lam}", flush=True)
    dtr = LatentDataset(args.train_dir, layers=layers)
    dte = LatentDataset(args.test_dir, layers=layers)
    Xtr, Ytr, _ = collect(dtr, layers, args.max_train_scenes or None)
    Xte, Yte, _ = collect(dte, layers, args.max_test_scenes or None)
    print(f"[probe] train N={len(Ytr)} test N={len(Yte)}", flush=True)

    order_name = {0: "Z", 1: "DZ", 2: "D2Z"}
    results = {}
    best = {}
    rng = np.random.default_rng(0)
    for L in layers:
        results[L] = {}
        for order in (0, 1, 2):
            dtr_o = derivative(Xtr[L], order)
            dte_o = derivative(Xte[L], order)
            per_t = {}
            for t in sorted(dtr_o):
                W, mu, sd, Yb = ridge_fit(dtr_o[t], Ytr, args.lam)
                sc = score(ridge_pred(dte_o[t], W, mu, sd, Yb), Yte)
                per_t[t] = sc
                key = (L, order_name[order], t)
                if not best or sc["r2_all"] > best["r2_all"]:
                    best = {"layer": L, "order": order_name[order], "t": t, **sc}
            # all-t concatenated probe for this order
            Xc_tr = np.concatenate([dtr_o[t] for t in sorted(dtr_o)], axis=1)
            Xc_te = np.concatenate([dte_o[t] for t in sorted(dte_o)], axis=1)
            W, mu, sd, Yb = ridge_fit(Xc_tr, Ytr, args.lam)
            sc_all = score(ridge_pred(Xc_te, W, mu, sd, Yb), Yte)
            # shuffle control on the concat
            Ysh = Ytr[rng.permutation(len(Ytr))]
            W, mu, sd, Yb = ridge_fit(Xc_tr, Ysh, args.lam)
            sc_sh = score(ridge_pred(Xc_te, W, mu, sd, Yb), Yte)
            results[L][order_name[order]] = {"per_t": {str(k): v for k, v in per_t.items()},
                                             "concat_all_t": sc_all, "shuffle_ctrl": sc_sh}
            print(f"  L{L} {order_name[order]:3s}: concat r2_all={sc_all['r2_all']:.3f} "
                  f"ang={sc_all['angle_deg']:.1f} mag_r2={sc_all['mag_r2']:.3f} | "
                  f"per-t r2 " + " ".join(f"t{t}:{per_t[t]['r2_all']:.2f}" for t in sorted(per_t)) +
                  f" | shuffle r2={sc_sh['r2_all']:.3f}", flush=True)

    summary = {"layers": layers, "lam": args.lam, "n_train": int(len(Ytr)), "n_test": int(len(Yte)),
               "best_single_probe": best, "results": results}
    (out / "temporal_probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[probe] BEST single probe: {best}")
    print(f"[probe] -> {out}/temporal_probe_summary.json", flush=True)


if __name__ == "__main__":
    main()
