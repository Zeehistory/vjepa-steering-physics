

#!/usr/bin/env python
"""POLAR-vs-RAW read-side steering GATE for angular velocity (big-object latents).

Decoder-free steering probe. For each coordinate space (RAW Cartesian token grid vs per-scene-centered
LOG-POLAR), independently: (1) fit a pooled ridge READ probe omega = w . pool(feature) on TRAIN scenes;
(2) build the shared omega-AXIS u = top-PCA of per-train-scene omega-directions (full-token); (3) on HELD-OUT
test scenes, edit a base clip toward each rank's target omega: f_edit = f_base + g*(omega_t-omega_base)*u,
then READ omega_read = w . pool(f_edit); calibrate the scalar gain g on train to min MSE, report held-out
rho / sign_acc / mag_ratio.

Absolute read-side numbers are partly CIRCULAR (the probe can read back the injected edit), so the headline
is the POLAR - RAW DIFFERENCE under the identical protocol + identical-style probe: circularity is common to
both, so a polar>>raw gap is real evidence the log-polar reframe improves WRITABILITY (not a probe artifact).
Gate: if polar can't beat raw here, the shared axis (best|cos|~0.45) is too weak -> need more token
resolution before any decoder retrain. Reuses to_polar / scene_omega_dir from _diag_angvel_polar.
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

import argparse, json, os
import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config
from _diag_angvel_polar import to_grid, center_cell, roll_center, to_polar


def clip_polar(x, cen, grid, n_r, n_phi, r_lo, r_hi):
    hc, wc = center_cell(cen, grid)
    return to_polar(x, hc, wc, n_r, n_phi, phi0=0.0, r_lo=r_lo, r_hi=r_hi)


def clip_centered(x, cen, grid):
    hc, wc = center_cell(cen, grid)
    return roll_center(x, hc, wc)


def fit_ridge(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def scene_dir(feats, omgs):
    """full-token omega-direction (least-squares slope of feature wrt omega), unit-normalized."""
    F = np.stack(feats) - np.stack(feats).mean(0, keepdims=True)
    o = omgs - omgs.mean()
    d = (o @ F) / ((o @ o) + 1e-12)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def eval_space(train, test, transform, lam=10.0):
    """train/test = list of scenes; each scene = list of (omega, full_feature_flat, pooled_feature)."""
    # --- read probe on pooled train features ---
    Xtr = np.stack([p for sc in train for (_, _, p) in sc])
    ytr = np.array([o for sc in train for (o, _, _) in sc])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    w = fit_ridge(Xtr, ytr, lam)
    # held-out read R2
    Xte = np.stack([p for sc in test for (_, _, p) in sc]); yte = np.array([o for sc in test for (o, _, _) in sc])
    Xte = (Xte - mu) / sd
    pr = Xte @ w
    r2 = 1 - float(((yte - pr) ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum())

    # --- shared omega-axis u = top PCA of per-train-scene full-token omega-directions ---
    dirs = []
    for sc in train:
        omgs = np.array([o for (o, _, _) in sc]); feats = [f for (_, f, _) in sc]
        dirs.append(scene_dir(feats, omgs))
    Dm = np.stack(dirs); Dm = Dm - Dm.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Dm, full_matrices=False)
    u = Vt[0]                                                 # full-token unit direction

    # pooled-read of the pure edit direction: pool u over tokens -> standardize -> project on w
    Tn = None
    def pool_flat(vec, ncol):
        return vec.reshape(-1, ncol).mean(0)
    ncol = sc[0][2].shape[0]                                  # pooled dim D
    u_pooled = pool_flat(u, ncol)
    u_read = ((u_pooled) / sd) @ w                            # how much omega-read moves per unit edit coeff

    # --- steering eval on held-out test scenes ---
    # base = min-|omega| rank; target = every other rank's omega
    def read_edit(base_full, coeff):
        edited = base_full + coeff * u
        pooled = pool_flat(edited, ncol)
        return float(((pooled - mu) / sd) @ w)
    # calibrate gain g on TRAIN scenes (min MSE of omega_read vs omega_t)
    def collect(scenes):
        rows = []
        for sc in scenes:
            omg = np.array([o for (o, _, _) in sc]); full = [f for (_, f, _) in sc]
            b = int(np.argmin(np.abs(omg)))
            for r in range(len(sc)):
                rows.append((full[b], omg[b], omg[r]))
        return rows
    tr_rows = collect(train)
    # omega_read(edit) = read(base) + g*(omega_t-omega_b)*u_read ; solve g by LS vs omega_t
    base_reads_tr = np.array([read_edit(bf, 0.0) for (bf, _, _) in tr_rows])
    slopes_tr = np.array([(ot - ob) * u_read for (_, ob, ot) in tr_rows])
    tgt_tr = np.array([ot for (_, _, ot) in tr_rows])
    # minimize sum((base + g*slope - tgt)^2)
    g = float((slopes_tr @ (tgt_tr - base_reads_tr)) / ((slopes_tr @ slopes_tr) + 1e-12))

    te_rows = collect(test)
    read = np.array([read_edit(bf, g * (ot - ob)) for (bf, ob, ot) in te_rows])
    tgt = np.array([ot for (_, _, ot) in te_rows])
    rho = float(np.corrcoef(tgt, read)[0, 1])
    sign = float(np.mean(np.sign(read) == np.sign(tgt)))
    # mag ratio: slope of read vs tgt
    magr = float((np.polyfit(tgt, read, 1)[0]))
    mae = float(np.mean(np.abs(read - tgt)))
    return dict(read_R2=r2, gain=g, u_read=float(u_read), steer_rho=rho, sign_acc=sign,
                mag_ratio=magr, mae=mae, n_test=len(te_rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--n_r", type=int, default=8)
    p.add_argument("--n_phi", type=int, default=16)
    p.add_argument("--r_lo", type=float, default=1.5)
    p.add_argument("--r_hi", type=float, default=7.5)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    ds = LatentDataset(args.dir, layers=cfg.encoder.layers)
    L = args.layer
    sc = vo.group_scenes(ds)
    sids = sorted(sc)
    ntr = int(len(sids) * args.train_frac)
    tr_ids, te_ids = sids[:ntr], sids[ntr:]
    print(f"[steer] layer={L} scenes train={len(tr_ids)} test={len(te_ids)}")

    def build(sids_sub, space):
        scenes = []
        for s in sids_sub:
            clips = []
            for rank, idx in sc[s].items():
                smp = ds[idx]; grid = smp["grid"]
                x = to_grid(smp["layers"][L], grid)
                omega = float(vo.clip_angvel(smp)[0])
                keys = list(smp["state_keys"]); st = np.asarray(smp["state"])
                cen = np.array([st[0, keys.index("obj0_pos_x")], st[0, keys.index("obj0_pos_y")]])
                if space == "raw":
                    f = x
                elif space == "centered":
                    f = clip_centered(x, cen, grid)
                elif space == "polar":
                    f = clip_polar(x, cen, grid, args.n_r, args.n_phi, args.r_lo, args.r_hi)
                ff = f.reshape(-1).astype(np.float64)
                pooled = f.reshape(-1, f.shape[-1]).mean(0).astype(np.float64)
                clips.append((omega, ff, pooled))
            if len(clips) >= 2:
                scenes.append(clips)
        return scenes

    out = {}
    for space in ["raw", "centered", "polar"]:
        tr = build(tr_ids, space); te = build(te_ids, space)
        res = eval_space(tr, te, space)
        out[space] = res
        print(f"  {space:9s}: read_R2={res['read_R2']:+.3f} | STEER rho={res['steer_rho']:+.3f} "
              f"sign_acc={res['sign_acc']:.3f} mag_ratio={res['mag_ratio']:+.3f} (gain={res['gain']:.2f})")
    print(f"\n  >>> POLAR - RAW steer_rho gap = {out['polar']['steer_rho'] - out['raw']['steer_rho']:+.3f} "
          f"(the circularity-cancelled headline)")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[steer] wrote {args.out}")


if __name__ == "__main__":
    main()
