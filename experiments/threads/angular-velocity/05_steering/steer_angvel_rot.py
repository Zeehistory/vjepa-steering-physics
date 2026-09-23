

#!/usr/bin/env python
"""ROTATION-TRANSPORT command-only angular-velocity steer (the round-trip-free polar operator).

polar cmd-U8 (steer_angvel_polar.py) confirmed the coordinate idea HELPS (ceiling_warp rho 0.54 >> Cartesian
cmd-axis 0.005; cmd->coord rho 0.3 >> ~0) but the naive warp->edit->UNWARP round-trip loses half the omega
signal (double bilinear resample), capping the held-out steer at rho~0.14. The fix is to realize the SAME
"translation along phi in log-polar" WITHOUT a round-trip: a rigid rotation about the scene centre is exactly
that phi-translation, and in the (T,H,W,D) token grid it is a SINGLE resample.

Operator (command-only, NO H_b, NO learned map -- pure geometry):
  to raise omega_a -> omega_b, the extra rotation at temporal token t is  dtheta(t) = scale*(omega_b-omega_a)*tau_t
  (tau_t = the token's mean video-frame index).  Rotate frame t's spatial token grid by dtheta(t) about the
  scene centre cell (bilinear).  Uniform background is rotation-invariant, so rotating the whole grid only
  moves the object.  scale (~1) absorbs V-JEPA's approximate rotation-equivariance; picked leakage-free on a
  val split, reported on the disjoint held-out split.  Decode with the frozen big-object decoder + honest
  rotor tracker (non-circular).

Variants:
  full     : rotate the entire spatial grid about the centre.
  annulus  : rotate only cells with r in [r_lo,r_hi] (the object), keep the rest (robust if bg not uniform).
Ceilings/controls: interp ceiling (rho 0.87, needs H_b) reported from interp_controllability.json; random
command control (shuffled omega_b) must not track; do-nothing (scale 0) baseline.
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

DARK, RED = 0.25, 0.08


def to_grid(arr, grid):
    T, H, W = grid
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(T, H, W, a.size // (T * H * W))


def center_cell(cen, grid):
    _, H, W = grid
    return (float(np.clip(cen[1] * (H - 1), 0, H - 1)), float(np.clip(cen[0] * (W - 1), 0, W - 1)))


def rotate_frames(x, hc, wc, dthetas, r_lo=None, r_hi=None):
    """Rotate each temporal slice of (T,H,W,D) by its OWN angle dthetas[t] (rad) about (hc,wc). Bilinear.
    If r_lo/r_hi given, only cells with r in [r_lo,r_hi] are rotated (others kept). Returns (T,H,W,D)."""
    T, H, W, D = x.shape
    hh = np.arange(H).reshape(H, 1) + np.zeros((H, W))
    ww = np.arange(W).reshape(1, W) + np.zeros((H, W))
    dh = hh - hc; dw = ww - wc
    r = np.sqrt(dh ** 2 + dw ** 2)
    out = np.empty_like(x)
    for t in range(T):
        a = -float(dthetas[t])                      # sample from the pre-rotation position (inverse map)
        ca, sa = np.cos(a), np.sin(a)
        sh = ca * dh - sa * dw + hc                  # source row
        sw = sa * dh + ca * dw + wc                  # source col
        h0 = np.floor(sh).astype(int); w0 = np.floor(sw).astype(int)
        fh = (sh - h0)[..., None]; fw = (sw - w0)[..., None]

        def g(hi, wi):
            return x[t, np.clip(hi, 0, H - 1), np.clip(wi, 0, W - 1), :]
        top = g(h0, w0) * (1 - fw) + g(h0, w0 + 1) * fw
        bot = g(h0 + 1, w0) * (1 - fw) + g(h0 + 1, w0 + 1) * fw
        rot = top * (1 - fh) + bot * fh              # (H,W,D)
        if r_lo is not None:
            m = ((r >= r_lo) & (r <= r_hi))[..., None]
            out[t] = np.where(m, rot, x[t])
        else:
            out[t] = rot
    return out


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def honest_omega(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    if fr is None:
        return float("nan")
    return float(measured_angvel(fr[0].cpu(), darkness_thresh=DARK, red_thresh=RED)["omega"])


def flat_to_dev(flat_by_layer, layers, ref, device):
    out = {}
    for L in layers:
        a = flat_by_layer[L].reshape(1, -1, flat_by_layer[L].shape[-1])
        out[L] = torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=ref[L].dtype)
    return out


def scene_center(sample):
    keys = list(sample["state_keys"]); st = np.asarray(sample["state"])
    return np.array([st[0, keys.index("obj0_pos_x")], st[0, keys.index("obj0_pos_y")]])


def metrics(pairs):
    pairs = [(t, h) for t, h in pairs if np.isfinite(h)]
    if len(pairs) < 3:
        return dict(n=len(pairs), rho=float("nan"), sign_acc=float("nan"), mag_ratio=float("nan"), mae=float("nan"))
    tgt = np.array([t for t, _ in pairs]); hon = np.array([h for _, h in pairs])
    return dict(n=len(pairs), rho=float(np.corrcoef(tgt, hon)[0, 1]),
                sign_acc=float(np.mean(np.sign(hon) == np.sign(tgt))),
                mag_ratio=float(np.polyfit(tgt, hon, 1)[0]), mae=float(np.mean(np.abs(hon - tgt))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_test_scenes", type=int, default=30)
    ap.add_argument("--variant", default="full", choices=["full", "annulus"])
    ap.add_argument("--r_lo", type=float, default=2.0)
    ap.add_argument("--r_hi", type=float, default=8.0)
    ap.add_argument("--scales", default="0,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = args.device
    scales = [float(s) for s in args.scales.split(",")]

    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in te[0]["layers"].keys())
    grid = tuple(int(x) for x in te[0]["grid"])
    T = grid[0]
    # token time axis tau_t (mean video-frame index per latent temporal token)
    F = int(np.asarray(te[0]["state"]).shape[0])
    tau = vo.frame_token_times(F, T)                 # (T,)
    print(f"[rot] layers={layers} grid={grid} F={F} tau={np.round(tau,1)} variant={args.variant} scales={scales}", flush=True)

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

    scenes = vo.group_scenes(te)
    sids = sorted(scenes)[: args.n_test_scenes]
    rl, rh = (args.r_lo, args.r_hi) if args.variant == "annulus" else (None, None)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(sids))
    val_sids = set(np.array(sids)[perm[: len(sids) // 2]].tolist())

    rows = []
    for n, s in enumerate(sids):
        ranks = sorted(scenes[s])
        wmap = {r: float(vo.clip_angvel(te[scenes[s][r]])[0]) for r in ranks}
        base = min(ranks, key=lambda r: abs(wmap[r]))
        sa = te[scenes[s][base]]
        Ha_dev = _to_dev(sa, layers, dev)
        Ha_cart = {L: to_grid(sa["layers"][L], grid) for L in layers}
        hc, wc = center_cell(scene_center(sa), grid)
        wa = wmap[base]
        for r in ranks:
            if r == base:
                continue
            wb = wmap[r]
            om_by_scale = {}
            for sc in scales:
                dth = sc * (wb - wa) * tau           # (T,) extra rotation per token
                edited = {L: rotate_frames(Ha_cart[L], hc, wc, dth, rl, rh) for L in layers}
                lat = flat_to_dev(edited, layers, Ha_dev, dev)
                om_by_scale[sc] = honest_omega(decoder, lat, grid)
            rows.append(dict(sid=s, in_val=(s in val_sids), wa=wa, wb=wb, om=om_by_scale))
        print(f"  scene {n+1}/{len(sids)}", flush=True)

    # scale selection on val (min MSE to target)
    val_mse = {}
    for sc in scales:
        pr = [(rw["wb"], rw["om"][sc]) for rw in rows if rw["in_val"] and np.isfinite(rw["om"][sc])]
        val_mse[sc] = float(np.mean([(h - t) ** 2 for t, h in pr])) if pr else float("inf")
    best = min(scales, key=lambda s: val_mse[s])

    held = metrics([(rw["wb"], rw["om"][best]) for rw in rows if not rw["in_val"]])
    val = metrics([(rw["wb"], rw["om"][best]) for rw in rows if rw["in_val"]])
    donothing = metrics([(rw["wb"], rw["om"][0.0]) for rw in rows if not rw["in_val"]]) if 0.0 in scales else None
    curve = {sc: metrics([(rw["wb"], rw["om"][sc]) for rw in rows if not rw["in_val"]]) for sc in scales}
    held_rows = [rw for rw in rows if not rw["in_val"]]
    wb_sh = list(rng.permutation([rw["wb"] for rw in held_rows]))
    rand = metrics([(wb_sh[i], held_rows[i]["om"][best]) for i in range(len(held_rows))])

    print("\n==================== ROTATION-TRANSPORT ANGULAR-VELOCITY RESULT ====================")
    print(f"  variant={args.variant}  best_scale(val)={best}  val_mse=" + " ".join(f"{s}:{val_mse[s]:.4f}" for s in scales))
    print(f"  HELD-OUT @scale{best}: rho={held['rho']:+.3f} sign={held['sign_acc']:.2f} mag={held['mag_ratio']:+.3f} mae={held['mae']:.4f} n={held['n']}")
    print(f"  val      @scale{best}: rho={val['rho']:+.3f} sign={val['sign_acc']:.2f} mag={val['mag_ratio']:+.3f} mae={val['mae']:.4f}")
    if donothing:
        print(f"  do-nothing (scale0):  rho={donothing['rho']:+.3f} sign={donothing['sign_acc']:.2f} mag={donothing['mag_ratio']:+.3f}")
    print(f"  random-command ctrl:  rho={rand['rho']:+.3f} sign={rand['sign_acc']:.2f} (should be ~0)")
    print("  held-out per-scale rho: " + " ".join(f"s{s}:{curve[s]['rho']:+.2f}(m{curve[s]['mag_ratio']:+.2f})" for s in scales))
    print("  reference: interp ceiling rho 0.87 (needs H_b); Cartesian cmd-axis 0.005; polar cmd-U8 0.14")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(variant=args.variant, best_scale=best, val_mse=val_mse, heldout=held, val=val,
                       do_nothing=donothing, random_command=rand,
                       scale_curve={str(s): curve[s] for s in scales}, layers=layers,
                       n_test_scenes=len(sids)), open(args.out, "w"), indent=2)
        print(f"[rot] wrote {args.out}")


if __name__ == "__main__":
    main()
