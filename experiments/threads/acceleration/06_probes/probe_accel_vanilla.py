

#!/usr/bin/env python
"""Does the VANILLA edit ``H_a + alpha*(H_b - H_a)`` install the correct acceleration MAGNITUDE?

Question: for the plain interpolation edit we only ever reported the *angle* after steering; we
never checked whether the steered acceleration MAGNITUDE matches what we want. This script answers it in
latent space with an independent linear probe (no decoder, so no decoder leniency confound).

Setup (mirrors ``probe_accel.py``): fit a ridge probe ``H -> a`` (2D accel) on TRAIN clips using the
temporal representation (spatial-pool per frame, keep the T time tokens -- acceleration is second-order so
time matters), pick the best (rep, layer) by held-out axis R2. Then, for each TEST scene pair
(a_a -> a_b, Delta a = a_b - a_a, Delta H = H_b - H_a), sweep alpha and read the interpolated latent:

    H(alpha) = H_a + alpha * Delta H          (alpha=0 -> H_a=a_a, alpha=1 -> H_b=a_b)
    expected accel  E(alpha) = a_a + alpha * Delta a     (linear-in-alpha ground truth)
    probe reads     P(alpha) = probe(H(alpha))

If the latent's acceleration axis is linear and carries magnitude, P(alpha) should track E(alpha) in BOTH
direction and magnitude, so |P(alpha)| should grow ~linearly with alpha and match |E(alpha)|. We report,
per alpha: angle(P, E), the magnitude ratio |P|/|E|, and the magnitude correlation across scenes. We also
report the magnitude SLOPE d|P|/d alpha vs the ideal d|E|/d alpha -- the crisp "does magnitude scale?"
number. This is the ceiling method (the true Delta H), so it isolates what the LATENT supports from what
the command operator loses.

    python experiments/threads/acceleration/06_probes/probe_accel_vanilla.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --layers 6,12,18,23 --alphas 0,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0 --output_dir .../accel_probe
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


def reps_from_layer(flat: np.ndarray, grid: tuple[int, int, int]) -> dict:
    T, H, W = grid
    D = flat.size // (T * H * W)
    x = flat.reshape(T, H, W, D)
    return {"pool": x.reshape(-1, D).mean(0), "temporal": x.mean(axis=(1, 2)).reshape(-1)}


def ridge_fit(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, mx, my


def ridge_pred(X, W, mx, my):
    return (X - mx) @ W + my


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def angle_mag(pred, true):
    """Mean angle err (deg), magnitude correlation r, mean magnitude ratio |pred|/|true|."""
    cos = (pred * true).sum(1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-12)
    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
    mp, mt = np.linalg.norm(pred, axis=1), np.linalg.norm(true, axis=1)
    magr = float(np.corrcoef(mp, mt)[0, 1]) if len(mp) > 1 else float("nan")
    return ang, magr, float((mp / (mt + 1e-12)).mean())


def collect(ds, scenes, layers, want_reps):
    X = {r: {L: [] for L in layers} for r in want_reps}
    Y = []
    for s in sorted(scenes):
        for rank, idx in sorted(scenes[s].items()):
            smp = ds[idx]
            grid = tuple(int(x) for x in smp["grid"])
            Y.append(vo.clip_acceleration(smp))
            for L in layers:
                rr = reps_from_layer(vo.layer_flat(smp["layers"][L]), grid)
                for r in want_reps:
                    X[r][L].append(rr[r])
        if hasattr(ds, "_shard_cache") and len(ds._shard_cache) > 2:
            ds._shard_cache.clear()
    Y = np.asarray(Y)
    X = {r: {L: np.asarray(X[r][L]) for L in layers} for r in want_reps}
    return X, Y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    reps = ["pool", "temporal"]

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    print(f"[vanilla] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}", flush=True)

    Xtr, Ytr = collect(tr, tr_scenes, layers, reps)
    Xte, Yte = collect(te, te_scenes, layers, reps)
    print(f"[vanilla] collected {len(Ytr)} train / {len(Yte)} test clips", flush=True)

    # fit probes + pick best (rep, L) by held-out axis R2 (same criterion as probe_accel.py)
    probes, r2mean = {}, {}
    for r in reps:
        for L in layers:
            W, mx, my = ridge_fit(Xtr[r][L], Ytr, args.lam)
            probes[(r, L)] = (W, mx, my)
            pred = ridge_pred(Xte[r][L], W, mx, my)
            r2v = r2(pred, Yte)
            r2mean[(r, L)] = float(r2v.mean())
            print(f"[vanilla] probe {r:8s} L{L}: R2 a=({r2v[0]:.3f},{r2v[1]:.3f})", flush=True)
    best = max(r2mean, key=r2mean.get)
    br, bL = best
    W, mx, my = probes[best]
    print(f"[vanilla] best probe = {br} L{bL} (R2mean {r2mean[best]:.3f}) -> alpha sweep", flush=True)
    del Xtr, Xte; gc.collect()

    # ---- alpha sweep on the vanilla edit H_a + alpha*(H_b - H_a), read best layer with the probe --------
    def rep_of(flat, grid):
        return reps_from_layer(flat, grid)[br].reshape(1, -1)

    # accumulate per-alpha: probe reading P and expected E across all test scenes
    P = {a: [] for a in alphas}
    E = {a: [] for a in alphas}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); ia, ib = te_scenes[s][ranks[0]], te_scenes[s][ranks[-1]]
        sa, sb = te[ia], te[ib]
        grid = tuple(int(x) for x in sa["grid"])
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        da = ab - aa
        HaL = vo.layer_flat(sa["layers"][bL])
        dHL = vo.layer_flat(sb["layers"][bL]) - HaL
        for a in alphas:
            steered = HaL + a * dHL
            P[a].append(ridge_pred(rep_of(steered, grid), W, mx, my)[0])
            E[a].append(aa + a * da)
        if hasattr(te, "_shard_cache") and len(te._shard_cache) > 2:
            te._shard_cache.clear()

    sweep = {}
    mp_by_alpha, me_by_alpha = [], []
    for a in alphas:
        Pa, Ea = np.asarray(P[a]), np.asarray(E[a])
        ang, magr, magratio = angle_mag(Pa, Ea)
        mp, me = np.linalg.norm(Pa, axis=1).mean(), np.linalg.norm(Ea, axis=1).mean()
        mp_by_alpha.append(mp); me_by_alpha.append(me)
        sweep[f"{a:g}"] = {
            "angle_err_deg": round(ang, 2), "mag_corr_r": round(magr, 3),
            "mag_ratio_mean": round(magratio, 3),
            "mean_probe_mag": round(float(mp), 6), "mean_expected_mag": round(float(me), 6),
        }
        print(f"[vanilla] alpha={a:<4g} angle={ang:6.2f} mag_corr_r={magr:+.3f} "
              f"|P|/|E|={magratio:.3f}  |P|={mp:.5f} |E|={me:.5f}", flush=True)

    # magnitude SLOPE: regress mean |P| and mean |E| on alpha; ideal ratio of slopes = 1
    A = np.asarray(alphas)
    def slope(y):
        y = np.asarray(y); Am = np.stack([A, np.ones_like(A)], 1)
        return float(np.linalg.lstsq(Am, y, rcond=None)[0][0])
    sp, se = slope(mp_by_alpha), slope(me_by_alpha)

    summary = {
        "quantity": "accel", "edit": "vanilla H_a + alpha*(H_b - H_a)",
        "probe": f"{br}_L{bL}", "probe_r2_mean": round(r2mean[best], 3),
        "lam": args.lam, "n_test_scenes": len(te_scenes), "alphas": alphas,
        "per_alpha": sweep,
        "magnitude_slope_probe_d|P|/dalpha": round(sp, 6),
        "magnitude_slope_expected_d|E|/dalpha": round(se, 6),
        "magnitude_slope_ratio": round(sp / (se + 1e-12), 3),
        "interpretation": ("slope_ratio ~1 and mag_corr_r high => the vanilla edit installs correct accel "
                           "MAGNITUDE (magnitude scales linearly with alpha and matches). slope_ratio <<1 "
                           "or mag_corr_r ~0 => magnitude is NOT faithfully written even by the true Delta H."),
    }
    (out / "accel_vanilla_magnitude.json").write_text(json.dumps(summary, indent=2))
    print(f"[vanilla] magnitude slope: probe {sp:.6f} vs expected {se:.6f} -> ratio {sp/(se+1e-12):.3f}",
          flush=True)
    print(f"[vanilla] wrote {out}/accel_vanilla_magnitude.json", flush=True)


if __name__ == "__main__":
    main()
