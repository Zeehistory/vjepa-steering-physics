

#!/usr/bin/env python
"""COMMAND-STYLE autonomous omega steer along a GLOBAL shared axis (decoder-verified, non-circular).

Free-edit TTO stalls (locally-stiff latent->omega). This asks whether a COHERENT GLOBAL direction -- learned
from OTHER scenes, never using the held-out scene's H_b -- can move the decoded rotation where a local
gradient couldn't. Pipeline:
  1) From the big TRAIN latents, fit the per-layer global omega-axis d_L = least-squares slope of the
     full-token latent wrt omega (units: Delta-H per unit Delta-omega), across all training clips.
  2) Calibrate ONE scalar gain g on a few TRAIN scenes (honest-decode grid search).
  3) HELD-OUT TEST: for each scene, base = min|omega| rank; for each target rank, edit
     H_edit = H_a + g*(omega_t - omega_a)*d_L, decode, read HONEST measured_angvel (pixel tracker, thr
     0.25/0.08 -- gate-proven). Report rho / sign_acc / mag_ratio vs the old cmd_U8 -0.25.
Non-circular: the judge is the pixel tracker, not a latent probe; the axis never sees the test scene's target.
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


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def honest_omega(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    if fr is None:
        return float("nan")
    return float(measured_angvel(fr[0].cpu(), darkness_thresh=DARK, red_thresh=RED)["omega"])


def fit_axis(train_ds, layers):
    """Per-layer un-normalized least-squares slope of full-token latent wrt omega, over all train clips."""
    n = len(train_ds)
    sw = sww = 0.0
    sH = {L: None for L in layers}; swH = {L: None for L in layers}
    for i in range(n):
        smp = train_ds[i]
        w = float(vo.clip_angvel(smp)[0])
        sw += w; sww += w * w
        for L in layers:
            v = np.asarray(smp["layers"][L], dtype=np.float64).reshape(-1)
            sH[L] = v.copy() if sH[L] is None else sH[L] + v
            swH[L] = w * v if swH[L] is None else swH[L] + w * v
    wbar = sw / n
    denom = sww - n * wbar * wbar + 1e-12
    d = {}
    for L in layers:
        # cov(H, w) / var(w) = (swH - n*wbar*Hbar)/denom
        Hbar = sH[L] / n
        d[L] = (swH[L] - n * wbar * Hbar) / denom      # (Ltok*D,) slope, units Delta-H per unit omega
    return d


def edit_latent(Ha, d, layers, coeff, shapes):
    return {L: Ha[L] + coeff * torch.as_tensor(d[L].reshape(shapes[L]), dtype=Ha[L].dtype,
                                               device=Ha[L].device) for L in layers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_test", type=int, default=25)
    p.add_argument("--num_cal", type=int, default=6)
    p.add_argument("--gains", default="0.5,1,1.5,2,3,4")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides); dev = args.device
    gains = [float(x) for x in args.gains.split(",")]

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    print(f"[axis] fitting global omega-axis from {len(tr)} train clips, layers={layers}", flush=True)
    d = fit_axis(tr, layers)
    for L in layers:
        print(f"  layer {L}: |d|={np.linalg.norm(d[L]):.3f}")

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

    def scene_pairs(ds, sids):
        out = []
        sc = vo.group_scenes(ds)
        for s in sids:
            ranks = sorted(sc[s]); ia = sc[s][ranks[0]]
            sa = ds[ia]; grid = tuple(int(x) for x in sa["grid"])
            wa = float(vo.clip_angvel(sa)[0]); Ha = _to_dev(sa, layers, dev)
            shapes = {L: Ha[L].shape for L in layers}
            tgts = [float(vo.clip_angvel(ds[sc[s][r]])[0]) for r in ranks[1:]]
            out.append((Ha, grid, wa, tgts, shapes))
        return out

    sc_tr = sorted(vo.group_scenes(tr))[: args.num_cal]
    cal = scene_pairs(tr, sc_tr)
    # ---- calibrate gain g on train scenes ----
    print(f"[axis] calibrating gain over {gains} on {len(cal)} train scenes", flush=True)
    best_g, best_err = gains[0], 1e9
    for g in gains:
        errs = []
        for (Ha, grid, wa, tgts, shapes) in cal:
            for wt in tgts:
                ho = honest_omega(decoder, edit_latent(Ha, d, layers, g * (wt - wa), shapes), grid)
                if np.isfinite(ho):
                    errs.append((ho - wt) ** 2)
        mse = float(np.mean(errs)) if errs else 1e9
        print(f"    g={g}: cal MSE={mse:.5f} (n={len(errs)})")
        if mse < best_err:
            best_err, best_g = mse, g
    print(f"[axis] best gain g={best_g}", flush=True)

    # ---- held-out test ----
    sc_te = sorted(vo.group_scenes(te))[: args.num_test]
    test = scene_pairs(te, sc_te)
    tgt_all, hon_all, uns_all = [], [], []
    for n, (Ha, grid, wa, tgts, shapes) in enumerate(test):
        uns = honest_omega(decoder, Ha, grid)
        for wt in tgts:
            ho = honest_omega(decoder, edit_latent(Ha, d, layers, best_g * (wt - wa), shapes), grid)
            if np.isfinite(ho):
                tgt_all.append(wt); hon_all.append(ho); uns_all.append(uns)
        print(f"  test scene {n+1}/{len(test)}", flush=True)

    tgt = np.array(tgt_all); hon = np.array(hon_all); uns = np.array(uns_all)
    rho = float(np.corrcoef(tgt, hon)[0, 1]); sign = float(np.mean(np.sign(hon) == np.sign(tgt)))
    slope = float(np.polyfit(tgt, hon, 1)[0]); mae = float(np.mean(np.abs(hon - tgt)))
    # baseline: unsteered omega vs target (what you'd get doing nothing)
    brho = float(np.corrcoef(tgt, uns)[0, 1])
    print(f"\n=== COMMAND-AXIS steer (held-out, g={best_g}, n={len(tgt)} decoded) ===")
    print(f"  STEERED honest omega vs target: rho={rho:+.3f} sign_acc={sign:.3f} mag_ratio={slope:+.3f} MAE={mae:.4f}")
    print(f"  baseline (unsteered omega vs target): rho={brho:+.3f}   [do-nothing control]")
    print(f"  vs old cmd_U8 raw-space held-out rho=-0.25")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(rho=rho, sign_acc=sign, mag_ratio=slope, mae=mae, baseline_rho=brho,
                       gain=best_g, n=len(tgt)), open(args.out, "w"), indent=2)
        print(f"[axis] wrote {args.out}")


if __name__ == "__main__":
    main()
