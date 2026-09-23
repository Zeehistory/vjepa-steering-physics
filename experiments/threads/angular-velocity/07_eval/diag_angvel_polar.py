

#!/usr/bin/env python
"""ANGULAR-VELOCITY COORDINATE DIAGNOSTIC (user hypothesis: the failure is a coordinate issue).

Prior negative (angular-velocity-steering.md): the per-scene omega-directions in RAW Cartesian latent token
space are ~88 deg apart => "no shared command axis" => cmd-U8 can't steer. But a rigid rotation about a
scene's centre is a PURE TRANSLATION ALONG phi in log-polar coordinates centred at that point -- and
translation velocity DOES steer (5.73 deg). So the 88 deg may be a COORDINATE artefact: each scene rotates
about a different centre with a different initial phase, so in raw (x,y) token space the omega-direction
points a different way per scene, even though they are the SAME operation in polar space.

This script tests that directly on the EXISTING latents (no re-encode). For a chosen layer it builds each
scene's omega-direction d_s (the latent direction that co-varies with omega across the scene's K ranks) in a
LADDER of coordinate frames and reports the pairwise cosine / mean principal angle across scenes:
  raw       : Cartesian token grid, no alignment            (expected ~88 deg, cos~0 -- reproduces the negative)
  centered  : roll each scene's centre cell to the grid centre (removes the different-centre confound)
  polar     : resample each spatial slice to a (n_r,n_phi) grid about the centre (rotation -> phi-shift)
  polar_pha : polar + roll the phi axis by -theta0 so all scenes share a phi origin (removes phase confound)
If the cosine climbs the ladder => coordinate issue CONFIRMED, and we know which transform unlocks a shared
angular-velocity axis (=> port cmd-U8 in that frame). If it stays ~0 everywhere => not a coordinate issue.

Also runs an ORIENTATION probe -- instead of the scalar omega, regress
(cos theta_t, sin theta_t) per latent temporal token from the pooled latent (ridge, held-out R^2). theta is
the primitive (omega is its time-derivative); if orientation reads cleanly the signal is unambiguously there.
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


# ----------------------------------------------------------------------------------------------------
# coordinate transforms on a single clip's layer, reshaped to (T, H, W, D)
# ----------------------------------------------------------------------------------------------------
def to_grid(sample_layer, grid):
    T, H, W = grid
    arr = np.asarray(sample_layer, dtype=np.float32)
    D = arr.size // (T * H * W)
    return arr.reshape(T, H, W, D)


def center_cell(center_xy, grid):
    _, H, W = grid
    w = int(np.clip(round(center_xy[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(center_xy[1] * (H - 1)), 0, H - 1))
    return h, w


def roll_center(x, hc, wc):
    """Roll the (T,H,W,D) grid so cell (hc,wc) lands at the grid centre."""
    _, H, W, _ = x.shape
    return np.roll(x, shift=(H // 2 - hc, W // 2 - wc), axis=(1, 2))


def to_polar(x, hc, wc, n_r, n_phi, phi0=0.0, r_lo=0.0, r_hi=None):
    """Resample each (H,W) spatial slice of (T,H,W,D) onto a polar grid (n_r,n_phi) about (hc,wc).

    Bilinear sampling. r spans [r_lo, r_hi] (default [0, R_max], R_max = half the grid); use a narrow
    [r_lo,r_hi] ANNULUS to concentrate on the object's rotating radius. phi spans [phi0, phi0+2pi). A
    rotation of the underlying content about the centre is (approximately) a pure roll along the phi axis in
    the output. Returns (T, n_r, n_phi, D).
    """
    T, H, W, D = x.shape
    R_max = 0.5 * min(H, W) if r_hi is None else r_hi
    rs = r_lo + (np.arange(n_r) + 0.5) / n_r * (R_max - r_lo)   # avoid r=0 singularity
    phis = phi0 + (np.arange(n_phi) + 0.0) / n_phi * 2 * np.pi
    # sample coords (image convention: col=w ~ x = center + r cos, row=h ~ y = center + r sin)
    rr, pp = np.meshgrid(rs, phis, indexing="ij")            # (n_r, n_phi)
    ww = wc + rr * np.cos(pp)
    hh = hc + rr * np.sin(pp)
    h0 = np.floor(hh).astype(int); w0 = np.floor(ww).astype(int)
    fh = (hh - h0)[..., None]; fw = (ww - w0)[..., None]
    def gather(hi, wi):
        hi = np.clip(hi, 0, H - 1); wi = np.clip(wi, 0, W - 1)
        return x[:, hi, wi, :]                                # (T, n_r, n_phi, D)
    c00 = gather(h0, w0); c01 = gather(h0, w0 + 1)
    c10 = gather(h0 + 1, w0); c11 = gather(h0 + 1, w0 + 1)
    top = c00 * (1 - fw) + c01 * fw
    bot = c10 * (1 - fw) + c11 * fw
    return top * (1 - fh) + bot * fh


# ----------------------------------------------------------------------------------------------------
# per-scene omega direction in a given coordinate frame
# ----------------------------------------------------------------------------------------------------
def _transform(x, frame, hc, wc, th0, n_r, n_phi, r_lo, r_hi):
    if frame == "raw":
        return x
    if frame == "centered":
        return roll_center(x, hc, wc)
    if frame == "polar":
        return to_polar(x, hc, wc, n_r, n_phi, phi0=0.0)
    if frame == "polar_pha":
        return to_polar(x, hc, wc, n_r, n_phi, phi0=float(th0))
    if frame == "polar_ann":                                  # object-annulus, finer phi
        return to_polar(x, hc, wc, n_r, n_phi, phi0=0.0, r_lo=r_lo, r_hi=r_hi)
    raise ValueError(frame)


def scene_omega_dir(clips, frame, grid, n_r, n_phi, r_lo=0.0, r_hi=None, return_grid=False):
    """clips = list of (omega, (T,H,W,D) grid, center_xy, theta0). Return unit d_s (omega-covariance)."""
    feats, omgs, shp = [], [], None
    for omega, x, cen, th0 in clips:
        hc, wc = center_cell(cen, grid)
        f = _transform(x, frame, hc, wc, th0, n_r, n_phi, r_lo, r_hi)
        shp = f.shape
        feats.append(f.reshape(-1).astype(np.float64))
        omgs.append(float(omega))
    F = np.stack(feats); omgs = np.asarray(omgs)
    F -= F.mean(0, keepdims=True)                             # remove the shared (frame-0-ish) content
    o = omgs - omgs.mean()
    d = (o @ F) / ((o @ o) + 1e-12)                          # least-squares slope of feature wrt omega
    n = np.linalg.norm(d)
    d = d / n if n > 1e-12 else d
    return (d, shp) if return_grid else d


def phialign_pairwise(dirs, shp):
    """Best-|cosine| over all integer phi-rolls, for polar dirs of shape (T,n_r,n_phi,D).

    Tests 'same operation up to a rotation': if a scene's omega-direction is another's rolled along phi, the
    fixed-phase cosine is low but the phi-aligned cosine is high => rotation == phi-translation at the token
    level. Reports mean best-|cos| and mean phi-aligned angle across scene pairs.
    """
    T, nr, nphi, D = shp
    grids = [d.reshape(shp) for d in dirs]
    N = len(grids); best = []
    for i in range(N):
        for j in range(i + 1, N):
            a = grids[i]
            bs = [np.roll(grids[j], k, axis=2) for k in range(nphi)]
            cs = [abs(float((a.reshape(-1) @ b.reshape(-1)))) for b in bs]  # dirs already unit-norm
            best.append(max(cs))
    best = np.asarray(best)
    ang = np.degrees(np.arccos(np.clip(best, 0, 1)))
    return dict(mean_best_abs_cos=float(best.mean()), mean_aligned_angle_deg=float(ang.mean()),
                median_aligned_angle_deg=float(np.median(ang)))


def pairwise_stats(dirs):
    dirs = np.stack(dirs)
    G = dirs @ dirs.T
    iu = np.triu_indices(len(dirs), k=1)
    cos = G[iu]
    ang = np.degrees(np.arccos(np.clip(np.abs(cos), 0, 1)))   # principal angle (sign-free)
    # shared-axis: energy in top PCA component of the direction set
    _, S, _ = np.linalg.svd(dirs - dirs.mean(0, keepdims=True), full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    return dict(mean_abs_cos=float(np.abs(cos).mean()), mean_cos=float(cos.mean()),
                mean_principal_angle_deg=float(ang.mean()), median_angle_deg=float(np.median(ang)),
                top1_pca_frac=float(ev[0]), top2_pca_frac=float(ev[:2].sum()))


# ----------------------------------------------------------------------------------------------------
# orientation (cos,sin theta) probe -- pooled features, ridge, held-out R^2
# ----------------------------------------------------------------------------------------------------
def token_theta(sample, T):
    """Per latent temporal token: circular-mean of the tubelet's per-frame obj0_theta -> (T, 2) unit dirs."""
    keys = list(sample["state_keys"]); st = np.asarray(sample["state"])
    th = st[:, keys.index("obj0_theta")]                     # (F,)
    F = len(th); per = max(1, F // T)
    out = []
    for t in range(T):
        seg = th[t * per:(t + 1) * per] if t < T - 1 else th[t * per:]
        c, s = np.cos(seg).mean(), np.sin(seg).mean()
        out.append([c, s])
    return np.asarray(out)                                    # (T,2), not unit-normalized (mag<=1)


def orient_probe(tr, te, layers, max_clips):
    def build(ds, cap):
        Xp = {L: [] for L in layers}; Y = []
        n = 0
        for i in range(len(ds)):
            smp = ds[i]
            grid = smp["grid"]; T = grid[0]
            th = token_theta(smp, T)                          # (T,2)
            for L in layers:
                g = to_grid(smp["layers"][L], grid)           # (T,H,W,D)
                Xp[L].append(g.reshape(T, -1, g.shape[-1]).mean(1))  # pool space -> (T,D)
            Y.append(th)
            n += 1
            if cap and n >= cap:
                break
        return {L: np.concatenate(Xp[L], 0) for L in layers}, np.concatenate(Y, 0)
    Xtr, Ytr = build(tr, max_clips); Xte, Yte = build(te, None)
    res = {}
    for L in layers:
        mu, sd = Xtr[L].mean(0), Xtr[L].std(0) + 1e-6
        a = (Xtr[L] - mu) / sd; b = (Xte[L] - mu) / sd
        d = a.shape[1]; W = np.linalg.solve(a.T @ a + 10.0 * np.eye(d), a.T @ Ytr)
        pred = b @ W
        ss_res = ((Yte - pred) ** 2).sum(0); ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
        r2 = 1 - ss_res / ss_tot
        # angular error of the predicted orientation vs GT (both as directions)
        ang_gt = np.arctan2(Yte[:, 1], Yte[:, 0]); ang_pr = np.arctan2(pred[:, 1], pred[:, 0])
        derr = np.degrees(np.abs(((ang_pr - ang_gt) + np.pi) % (2 * np.pi) - np.pi))
        res[L] = dict(r2_cos=float(r2[0]), r2_sin=float(r2[1]),
                      median_orient_err_deg=float(np.median(derr)))
        print(f"  L{L:2d}: R2_cos={r2[0]:+.3f} R2_sin={r2[1]:+.3f} median_orient_err={np.median(derr):.1f}deg")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layer", type=int, default=18, help="layer for the polar pairwise-cosine ladder")
    p.add_argument("--n_scenes", type=int, default=80)
    p.add_argument("--n_r", type=int, default=8)
    p.add_argument("--n_phi", type=int, default=16)
    p.add_argument("--max_probe_clips", type=int, default=800)
    p.add_argument("--skip_probe", action="store_true")
    p.add_argument("--ann_r_lo", type=float, default=1.0)
    p.add_argument("--ann_r_hi", type=float, default=5.0)
    p.add_argument("--ann_n_r", type=int, default=6)
    p.add_argument("--ann_n_phi", type=int, default=32)
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    print(f"[diag] layers={layers} polar_layer={args.layer} grid={tr[0]['grid']}")

    # ---- (A) orientation probe -------------------------------------------------------------------
    orient = None
    if not args.skip_probe:
        print("\n=== (A) ORIENTATION probe: regress (cos theta_t, sin theta_t) per temporal token ===")
        orient = orient_probe(tr, te, layers, args.max_probe_clips)

    # ---- (B) coordinate ladder for omega-direction pairwise cosine -------------------------------
    print(f"\n=== (B) omega-direction pairwise cosine across coordinate frames (layer {args.layer}) ===")
    L = args.layer
    sc = vo.group_scenes(te)
    scene_ids = sorted(sc)[:args.n_scenes]
    scenes = []; radii = []
    for s in scene_ids:
        clips = []
        for rank, idx in sc[s].items():
            smp = te[idx]
            grid = smp["grid"]
            x = to_grid(smp["layers"][L], grid)
            omega = float(vo.clip_angvel(smp)[0])
            keys = list(smp["state_keys"]); st = np.asarray(smp["state"])
            cen = np.array([st[0, keys.index("obj0_pos_x")], st[0, keys.index("obj0_pos_y")]])
            th0 = float(st[0, keys.index("obj0_theta")])
            rad = float(st[0, keys.index("obj0_radius")])
            clips.append((omega, x, cen, th0))
        if len(clips) >= 2:
            scenes.append((grid, clips)); radii.append(rad)
    radii = np.asarray(radii)
    print(f"[diag] {len(scenes)} scenes, ~{np.mean([len(c) for _,c in scenes]):.1f} ranks each")

    ladder = {}
    for frame in ["raw", "centered", "polar", "polar_pha", "polar_ann"]:
        nr = args.ann_n_r if frame == "polar_ann" else args.n_r
        nphi = args.ann_n_phi if frame == "polar_ann" else args.n_phi
        rlo = args.ann_r_lo if frame == "polar_ann" else 0.0
        rhi = args.ann_r_hi if frame == "polar_ann" else None
        pairs = [scene_omega_dir(clips, frame, grid, nr, nphi, r_lo=rlo, r_hi=rhi, return_grid=True)
                 for grid, clips in scenes]
        dirs = [d for d, _ in pairs]; shp = pairs[0][1]
        st = pairwise_stats(dirs)
        if frame in ("polar", "polar_pha", "polar_ann"):
            st.update(phialign_pairwise(dirs, shp))          # best-cosine over phi-rolls
        ladder[frame] = st
        extra = (f" | phi-aligned angle={st['mean_aligned_angle_deg']:.1f}deg "
                 f"best|cos|={st['mean_best_abs_cos']:.3f}") if "mean_best_abs_cos" in st else ""
        print(f"  {frame:10s}: mean_angle={st['mean_principal_angle_deg']:.1f}deg "
              f"top1_pca={st['top1_pca_frac']:.3f} top2_pca={st['top2_pca_frac']:.3f}{extra}")

    # ---- (C) size stratification: does a BIGGER object (more token cells) give a cleaner shared axis? --
    # If yes, resolution is the limiter and higher-res encoding would crack it (predicts the fix).
    print(f"\n=== (C) SIZE stratification of polar_ann shared-axis (radius terciles) ===")
    strat = {}
    order = np.argsort(radii)
    terciles = np.array_split(order, 3)
    for name, idxs in zip(["small", "mid", "large"], terciles):
        sub = [scenes[i] for i in idxs]
        pairs = [scene_omega_dir(clips, "polar_ann", grid, args.ann_n_r, args.ann_n_phi,
                                 r_lo=args.ann_r_lo, r_hi=args.ann_r_hi, return_grid=True)
                 for grid, clips in sub]
        dirs = [d for d, _ in pairs]; shp = pairs[0][1]
        stt = pairwise_stats(dirs); stt.update(phialign_pairwise(dirs, shp))
        rlo, rhi = float(radii[idxs].min()), float(radii[idxs].max())
        strat[name] = dict(radius_lo=rlo, radius_hi=rhi, n=len(idxs), **stt)
        print(f"  {name:5s} (r={rlo:.3f}-{rhi:.3f}, n={len(idxs)}): "
              f"top1_pca={stt['top1_pca_frac']:.3f} phi-aligned best|cos|={stt['mean_best_abs_cos']:.3f} "
              f"aligned_angle={stt['mean_aligned_angle_deg']:.1f}deg")

    out = dict(layer=L, n_scenes=len(scenes), n_r=args.n_r, n_phi=args.n_phi,
               orientation_probe=orient, coordinate_ladder=ladder, size_stratified=strat)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
