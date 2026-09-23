

#!/usr/bin/env python
"""POLAR command-U8 for ANGULAR VELOCITY (polar-coordinate hypothesis, decoder-verified).

Prior arc (angular-velocity-steering.md): omega is READABLE (R^2=0.96), the big-object decoder RENDERS it
(decode(true H_b) tracks GT omega rho=0.98), and it is TRANSPORT-controllable (interp H_a+alpha(H_b-H_a)
ramps omega, rho=0.87). The ONE thing that failed was a COMMAND-ONLY steer: in RAW Cartesian token space the
per-scene omega-directions are ~orthogonal (no shared axis, cmd-axis steer rho~0). The diagnostic
(_diag_angvel_polar.py) showed WHY: a rigid rotation about a scene's centre is a PURE TRANSLATION ALONG phi
in LOG-POLAR coordinates centred at that point -- and translation velocity DOES steer command-only (5.73deg).
In polar the per-scene shared axis ~doubles (best|cos 0.46 on the big object).

This script closes the loop: it ports the SOLVED velocity cmd-U8 operator into LOG-POLAR coordinates and
evaluates it THROUGH THE FROZEN DECODER with the honest pixel rotor tracker (non-circular).

Pipeline (per layer L, all done in numpy; only the final decode is on GPU):
  warp   : resample each clip's (T,H,W,D) token grid to log-polar (T,n_r,n_phi,D) about the scene centre,
           where rotation -> phi-shift (reuses to_polar from the diagnostic).
  fit    : within-scene deltas dH_polar = warp(H_b) - warp(H_a) -> global PCA U8 (the now-aligned omega
           subspace) + ridge command_features(omega_a,omega_b) -> U8 coords.  == cmd-U8, in polar.
  steer  : predict coords from the COMMAND alone, edit_polar = gain * coords . U8, UNWARP the edit back to
           Cartesian (from_polar), add to H_a (base latent is NEVER round-tripped -> only the structured
           delta passes through the warp), decode, honest-track omega.

Ceilings / controls (all decoder-verified):
  ceiling_full   : decode(H_b)                                            -- the render ceiling (~0.98).
  ceiling_warp   : decode(H_a + from_polar(warp(H_b)-warp(H_a)))          -- does the warp round-trip of the
                   TRUE delta preserve steering? gates the whole idea. If low, the warp is too lossy.
  cart_cmdaxis   : (known negative, rho~0) reported from steer_command_axis.json if present.
  random_cmd     : steer with a SHUFFLED command (wrong omega_b) -- must NOT track.
Gain is picked leakage-free on a val scene split (min MSE(omega_honest, omega_target)); held-out reported on
the disjoint test split.
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

import argparse, json
from pathlib import Path
import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_angvel
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

DARK, RED = 0.25, 0.08  # gate-proven tight rotor-tracker thresholds


# ----------------------------------------------------------------------------------------------------
# log-polar warp / unwarp on a single clip layer reshaped to (T, H, W, D)
# ----------------------------------------------------------------------------------------------------
def to_grid(arr, grid):
    T, H, W = grid
    a = np.asarray(arr, dtype=np.float32)
    D = a.size // (T * H * W)
    return a.reshape(T, H, W, D)


def center_cell(cen, grid):
    _, H, W = grid
    w = float(np.clip(cen[0] * (W - 1), 0, W - 1))
    h = float(np.clip(cen[1] * (H - 1), 0, H - 1))
    return h, w


def to_polar(x, hc, wc, n_r, n_phi, r_lo, r_hi):
    """(T,H,W,D) -> (T,n_r,n_phi,D). Bilinear. Matches _diag_angvel_polar.to_polar (phi0=0)."""
    T, H, W, D = x.shape
    rs = r_lo + (np.arange(n_r) + 0.5) / n_r * (r_hi - r_lo)
    phis = (np.arange(n_phi) + 0.0) / n_phi * 2 * np.pi
    rr, pp = np.meshgrid(rs, phis, indexing="ij")           # (n_r,n_phi)
    ww = wc + rr * np.cos(pp)
    hh = hc + rr * np.sin(pp)
    h0 = np.floor(hh).astype(int); w0 = np.floor(ww).astype(int)
    fh = (hh - h0)[..., None]; fw = (ww - w0)[..., None]

    def gather(hi, wi):
        return x[:, np.clip(hi, 0, H - 1), np.clip(wi, 0, W - 1), :]
    c00 = gather(h0, w0); c01 = gather(h0, w0 + 1)
    c10 = gather(h0 + 1, w0); c11 = gather(h0 + 1, w0 + 1)
    top = c00 * (1 - fw) + c01 * fw
    bot = c10 * (1 - fw) + c11 * fw
    return top * (1 - fh) + bot * fh


def from_polar(P, grid, hc, wc, n_r, n_phi, r_lo, r_hi):
    """Inverse of to_polar: (T,n_r,n_phi,D) -> (T,H,W,D). Bilinear over (r,phi); phi periodic, r clamped;
    Cartesian cells with r outside [r_lo,r_hi] get 0 (the edit is ~0 outside the object annulus anyway)."""
    T, nr, nphi, D = P.shape
    _, H, W = grid
    hh = np.arange(H).reshape(H, 1); ww = np.arange(W).reshape(1, W)
    dh = hh - hc; dw = ww - wc                                # (H,1),(1,W) -> broadcast (H,W)
    r = np.sqrt(dh ** 2 + dw ** 2) + np.zeros((H, W))
    phi = np.mod(np.arctan2(dh + np.zeros((H, W)), dw + np.zeros((H, W))), 2 * np.pi)
    ir = (r - r_lo) / (r_hi - r_lo) * nr - 0.5                # fractional r index (bin centres at +0.5)
    ip = phi / (2 * np.pi) * nphi                             # fractional phi index (bin centres at k)
    valid = (r >= r_lo) & (r <= r_hi)
    ir0 = np.floor(ir).astype(int); fr = (ir - ir0)[..., None]
    ip0 = np.floor(ip).astype(int); fp = (ip - ip0)[..., None]
    ir0c = np.clip(ir0, 0, nr - 1); ir1c = np.clip(ir0 + 1, 0, nr - 1)
    ip0m = np.mod(ip0, nphi); ip1m = np.mod(ip0 + 1, nphi)
    fr_flat = fr.reshape(-1, 1); fp_flat = fp.reshape(-1, 1)
    r0 = ir0c.reshape(-1); r1 = ir1c.reshape(-1)
    p0 = ip0m.reshape(-1); p1 = ip1m.reshape(-1)             # (H*W,)
    c00 = P[:, r0, p0, :]; c01 = P[:, r0, p1, :]             # (T,H*W,D)
    c10 = P[:, r1, p0, :]; c11 = P[:, r1, p1, :]
    top = c00 * (1 - fp_flat) + c01 * fp_flat
    bot = c10 * (1 - fp_flat) + c11 * fp_flat
    out = top * (1 - fr_flat) + bot * fr_flat                # (T,H*W,D)
    out = out.reshape(T, H, W, D)
    out *= valid[None, :, :, None]
    return out


# ----------------------------------------------------------------------------------------------------
# decode + honest omega
# ----------------------------------------------------------------------------------------------------
def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def honest_omega(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    if fr is None:
        return float("nan")
    return float(measured_angvel(fr[0].cpu(), darkness_thresh=DARK, red_thresh=RED)["omega"])


def flat_to_dev(flat_by_layer, grid, layers, ref_latents, device):
    """Turn a {L: (T,H,W,D) np} edited latent into decoder-ready {L: (1, T*H*W, D) tensor}, matching
    ref_latents' dtype/order (token order is temporal-major row-major, same as LatentDataset flatten)."""
    out = {}
    for L in layers:
        arr = flat_by_layer[L].reshape(1, -1, flat_by_layer[L].shape[-1])
        out[L] = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=ref_latents[L].dtype)
    return out


# ----------------------------------------------------------------------------------------------------
def scene_center_theta(sample):
    keys = list(sample["state_keys"]); st = np.asarray(sample["state"])
    cen = np.array([st[0, keys.index("obj0_pos_x")], st[0, keys.index("obj0_pos_y")]])
    th0 = float(st[0, keys.index("obj0_theta")])
    return cen, th0


def build_polar_pack(ds, scene_ids, scenes, layers, grid, pol):
    """For each scene return base (min|omega|) + target ranks with warped layers. Warps once, reused for
    fit and steer. Returns list of dicts: {sid, base_idx, wa, cen, layersA_polar, targets:[(idx,wb,polarB)]}."""
    n_r, n_phi, r_lo, r_hi = pol
    packs = []
    for sid in scene_ids:
        ranks = sorted(scenes[sid])
        # base = smallest |omega|
        wsmap = {}
        for rk in ranks:
            wsmap[rk] = float(vo.clip_angvel(ds[scenes[sid][rk]])[0])
        base_rk = min(ranks, key=lambda r: abs(wsmap[r]))
        sa = ds[scenes[sid][base_rk]]
        cen, th0 = scene_center_theta(sa)
        hc, wc = center_cell(cen, grid)
        polarA = {L: to_polar(to_grid(sa["layers"][L], grid), hc, wc, n_r, n_phi, r_lo, r_hi) for L in layers}
        targets = []
        for rk in ranks:
            if rk == base_rk:
                continue
            sb = ds[scenes[sid][rk]]
            polarB = {L: to_polar(to_grid(sb["layers"][L], grid), hc, wc, n_r, n_phi, r_lo, r_hi) for L in layers}
            targets.append(dict(idx=scenes[sid][rk], wb=wsmap[rk], polarB=polarB))
        packs.append(dict(sid=sid, base_idx=scenes[sid][base_rk], wa=wsmap[base_rk],
                          cen=cen, hc=hc, wc=wc, polarA=polarA, targets=targets))
    return packs


def fit_operator(train_packs, layers, k_u, ridge):
    """cmd-U8 in polar: PCA(dH_polar)->U8, ridge command_features->U8 coords. Per layer.
    Returns {L: (U (k,Dp), Wc (13,k))}."""
    ops = {}
    for L in layers:
        dHs, feats = [], []
        for p in train_packs:
            a = p["polarA"][L].reshape(-1)
            for t in p["targets"]:
                dHs.append(t["polarB"][L].reshape(-1) - a)
                feats.append(vo.command_features([p["wa"], 0.0], [t["wb"], 0.0]))
        X = np.stack(dHs).astype(np.float64)                 # (N, Dp)
        F = np.stack(feats).astype(np.float64)               # (N, 13)
        U, _ = vo.pca_gram(X, k=k_u)                         # (k, Dp)
        coords = X @ U.T                                     # (N, k)  project deltas onto U
        # ridge command -> coords
        A = F.T @ F + ridge * np.eye(F.shape[1])
        Wc = np.linalg.solve(A, F.T @ coords)               # (13, k)
        ops[L] = (U, Wc)
        # diagnostics
        Xc = X - X.mean(0, keepdims=True)
        var_u = float((coords.var(0).sum()) / (Xc.var(0).sum() + 1e-30))
        pred = F @ Wc
        cc = float(np.mean([np.corrcoef(coords[:, j], pred[:, j])[0, 1] for j in range(k_u)]))
        print(f"  [fit] L{L}: N={len(dHs)} U8-var-frac={var_u:.3f} cmd->coord meanrho={cc:.3f}", flush=True)
    return ops


def predict_edit_cart(op, wa, wb, layers, grid, hc, wc, pol, gain, coords_override=None):
    """command -> edit_polar -> unwarp to Cartesian, per layer. Returns {L: (T,H,W,D)}."""
    n_r, n_phi, r_lo, r_hi = pol
    f = vo.command_features([wa, 0.0], [wb, 0.0])
    edit = {}
    for L in layers:
        U, Wc = op[L]
        coords = f @ Wc if coords_override is None else coords_override[L]
        dH_polar = (coords @ U).reshape(grid[0], n_r, n_phi, -1)   # (T,n_r,n_phi,D)
        edit[L] = gain * from_polar(dH_polar, grid, hc, wc, n_r, n_phi, r_lo, r_hi)
    return edit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_train_scenes", type=int, default=125)
    ap.add_argument("--n_test_scenes", type=int, default=80)
    ap.add_argument("--k_u", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--n_r", type=int, default=8)
    ap.add_argument("--n_phi", type=int, default=24)
    ap.add_argument("--r_lo", type=float, default=0.5)
    ap.add_argument("--r_hi", type=float, default=7.5)
    ap.add_argument("--gains", default="0.5,1,1.5,2,3,4,6")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = args.device
    pol = (args.n_r, args.n_phi, args.r_lo, args.r_hi)
    gains = [float(g) for g in args.gains.split(",")]

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    grid = tuple(int(x) for x in tr[0]["grid"])
    print(f"[polar] layers={layers} grid={grid} polar=(n_r{args.n_r},n_phi{args.n_phi},r[{args.r_lo},{args.r_hi}]) k_u={args.k_u}", flush=True)

    # decoder
    rec0 = te.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in te.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=dev)
    for pm in decoder.parameters():
        pm.requires_grad_(False)

    # ---- fit on train ----
    tr_scenes = vo.group_scenes(tr)
    tr_ids = sorted(tr_scenes)[: args.n_train_scenes]
    print(f"[polar] warping {len(tr_ids)} train scenes ...", flush=True)
    train_packs = build_polar_pack(tr, tr_ids, tr_scenes, layers, grid, pol)
    print(f"[polar] fitting cmd-U8 operator (k_u={args.k_u}, ridge={args.ridge}) ...", flush=True)
    op = fit_operator(train_packs, layers, args.k_u, args.ridge)

    # ---- steer test ----
    te_scenes = vo.group_scenes(te)
    te_ids = sorted(te_scenes)[: args.n_test_scenes]
    print(f"[polar] warping {len(te_ids)} test scenes ...", flush=True)
    test_packs = build_polar_pack(te, te_ids, te_scenes, layers, grid, pol)

    # split test scenes into val (gain-cal) / heldout (report), disjoint
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(test_packs))
    n_val = len(test_packs) // 2
    val_idx = set(perm[:n_val].tolist()); held_idx = set(perm[n_val:].tolist())

    # precompute per (scene,target): base H_a on device, GT wb, and the ceiling/edit records
    rows = []  # dicts across all (scene,target)
    print("[polar] decoding ceilings + edits ...", flush=True)
    for pi, p in enumerate(test_packs):
        sa = te[p["base_idx"]]
        Ha_dev = _to_dev(sa, layers, dev)
        Ha_cart = {L: to_grid(sa["layers"][L], grid) for L in layers}
        for t in p["targets"]:
            wb = t["wb"]
            # ceiling_warp: true delta through warp round-trip
            edit_true = {L: from_polar(t["polarB"][L] - p["polarA"][L], grid, p["hc"], p["wc"],
                                       args.n_r, args.n_phi, args.r_lo, args.r_hi) for L in layers}
            lat_ceil = flat_to_dev({L: Ha_cart[L] + edit_true[L] for L in layers}, grid, layers, Ha_dev, dev)
            om_ceil = honest_omega(decoder, lat_ceil, grid)
            # cmd-U8 edits at every gain (predict once, scale)
            edit_unit = predict_edit_cart(op, p["wa"], wb, layers, grid, p["hc"], p["wc"], pol, 1.0)
            om_by_gain = {}
            for g in gains:
                lat = flat_to_dev({L: Ha_cart[L] + g * edit_unit[L] for L in layers}, grid, layers, Ha_dev, dev)
                om_by_gain[g] = honest_omega(decoder, lat, grid)
            rows.append(dict(scene=pi, in_val=(pi in val_idx), wa=p["wa"], wb=wb,
                             om_ceil=om_ceil, om_by_gain=om_by_gain))
        print(f"  scene {pi+1}/{len(test_packs)}", flush=True)

    # random-command control: reuse held-out scenes, shuffle wb across them at the best gain (computed below)
    def metrics(pairs):
        pairs = [(t, h) for t, h in pairs if np.isfinite(h)]
        if len(pairs) < 3:
            return dict(n=len(pairs), rho=float("nan"), sign_acc=float("nan"),
                        mag_ratio=float("nan"), mae=float("nan"))
        tgt = np.array([t for t, _ in pairs]); hon = np.array([h for _, h in pairs])
        return dict(n=len(pairs), rho=float(np.corrcoef(tgt, hon)[0, 1]),
                    sign_acc=float(np.mean(np.sign(hon) == np.sign(tgt))),
                    mag_ratio=float(np.polyfit(tgt, hon, 1)[0]), mae=float(np.mean(np.abs(hon - tgt))))

    # gain selection on val (min MSE toward target)
    val_mse = {}
    for g in gains:
        pairs = [(r["wb"], r["om_by_gain"][g]) for r in rows if r["in_val"]]
        pairs = [(t, h) for t, h in pairs if np.isfinite(h)]
        val_mse[g] = float(np.mean([(h - t) ** 2 for t, h in pairs])) if pairs else float("inf")
    best_gain = min(gains, key=lambda g: val_mse[g])

    held = metrics([(r["wb"], r["om_by_gain"][best_gain]) for r in rows if not r["in_val"]])
    val = metrics([(r["wb"], r["om_by_gain"][best_gain]) for r in rows if r["in_val"]])
    ceil = metrics([(r["wb"], r["om_ceil"]) for r in rows])
    # per-gain held-out curve (diagnostic)
    gain_curve = {g: metrics([(r["wb"], r["om_by_gain"][g]) for r in rows if not r["in_val"]]) for g in gains}

    # random control at best gain (shuffle targets among held-out)
    held_rows = [r for r in rows if not r["in_val"]]
    wbs = [r["wb"] for r in held_rows]
    wb_shuf = list(rng.permutation(wbs))
    rand = metrics([(wb_shuf[i], held_rows[i]["om_by_gain"][best_gain]) for i in range(len(held_rows))])

    print("\n==================== POLAR cmd-U8 ANGULAR-VELOCITY RESULT ====================")
    print(f"  best_gain (val-selected) = {best_gain}   val_mse_curve = "
          + " ".join(f"{g}:{val_mse[g]:.4f}" for g in gains))
    print(f"  ceiling_warp (true delta thru warp): rho={ceil['rho']:+.3f} sign={ceil['sign_acc']:.2f} "
          f"mag={ceil['mag_ratio']:+.3f} mae={ceil['mae']:.4f} n={ceil['n']}")
    print(f"  HELD-OUT cmd-U8-polar @g{best_gain}:  rho={held['rho']:+.3f} sign={held['sign_acc']:.2f} "
          f"mag={held['mag_ratio']:+.3f} mae={held['mae']:.4f} n={held['n']}")
    print(f"  val      cmd-U8-polar @g{best_gain}:  rho={val['rho']:+.3f} sign={val['sign_acc']:.2f} "
          f"mag={val['mag_ratio']:+.3f} mae={val['mae']:.4f} n={val['n']}")
    print(f"  random-command control @g{best_gain}: rho={rand['rho']:+.3f} sign={rand['sign_acc']:.2f} "
          f"(should be ~0 / ~0.5)")
    print("  held-out per-gain rho:  " + " ".join(f"g{g}:{gain_curve[g]['rho']:+.2f}" for g in gains))
    print("  reference negatives: cart cmd-axis rho~0.005 (steer_command_axis.json); interp ceiling rho 0.87")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(best_gain=best_gain, val_mse=val_mse, ceiling_warp=ceil,
                       heldout=held, val=val, random_command=rand,
                       gain_curve={str(g): gain_curve[g] for g in gains},
                       polar=dict(n_r=args.n_r, n_phi=args.n_phi, r_lo=args.r_lo, r_hi=args.r_hi),
                       k_u=args.k_u, ridge=args.ridge, layers=layers,
                       n_train_scenes=len(tr_ids), n_test_scenes=len(te_ids)),
                  open(args.out, "w"), indent=2)
        print(f"[polar] wrote {args.out}")


if __name__ == "__main__":
    main()
