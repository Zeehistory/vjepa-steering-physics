

#!/usr/bin/env python
"""Diagnostic: does the ViT-g angaccel decoder render ALPHA (curvature), or only ORIENTATION/omega?

The first sweep (_diag_angvel_decode.py) showed decode(true H_b) is honestly trackable for OMEGA
(rho ~0.73 at tight thresholds; n_valid=0 at the default 0.5/0.25 because the ViT-g marker redness maxes
~0.19 < 0.25). But steer_fourier.py already uses (0.25,0.08) and STILL read the alpha ceiling at 0.111 =
chance. So the open question is not thresholds-in-general but the QUANTITY: alpha is the 2nd derivative of
orientation and amplifies pixel noise (see the readout-noise control). This decodes true H_b for held-out
ViT-g angaccel scenes and reports BOTH measured_angvel (vs clip omega0) and measured_angaccel (vs clip
alpha) across the same 4 threshold pairs. If alpha rho recovers at some threshold -> calibration after all;
if omega rho is high while alpha rho stays ~chance at every threshold -> the ViT-g decoder renders
orientation faithfully but not curvature (a genuine 2nd-order render-noise floor, not a bug).
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

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_angvel, measured_angaccel
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _corr(gt, meas):
    d = np.array(meas); g = np.array(gt); ok = np.isfinite(d)
    n = int(ok.sum())
    if n < 2:
        return None
    rho = float(np.corrcoef(g[ok], d[ok])[0, 1])
    magr = float(np.median(np.abs(d[ok]) / (np.abs(g[ok]) + 1e-9)))
    sgn = float(np.mean(np.sign(d[ok]) == np.sign(g[ok])))
    return n, rho, sgn, magr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_scenes", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    dev = args.device
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=dev)

    thresholds = [(0.5, 0.25), (0.35, 0.15), (0.25, 0.08), (0.15, 0.04)]
    print(f"thresholds (dark,red): {thresholds}")
    gt_w, gt_a = [], []
    w_by_thr = {th: [] for th in thresholds}
    a_by_thr = {th: [] for th in thresholds}

    @torch.no_grad()
    def decode(sample):
        Ha = _to_dev(sample, layers, dev)
        grid = tuple(int(x) for x in sample["grid"])
        out = decoder(Ha, grid)
        return None if out.frames is None else out.frames[0].cpu()

    for s in scene_ids:
        ranks = sorted(scenes[s]); ib = scenes[s][ranks[-1]]
        sb = ds[ib]
        wb = float(vo.clip_angvel(sb)[0]); ab = float(vo.clip_angaccel(sb)[0])
        fb = decode(sb)
        if fb is None:
            print(f"scene{s:05d}: H_b decode None"); continue
        r, g, b = fb[:, 0], fb[:, 1], fb[:, 2]
        redness = (r - torch.maximum(g, b)).clamp(min=0)
        rmax = float(redness.max())
        row = []
        for th in thresholds:
            mw = measured_angvel(fb, darkness_thresh=th[0], red_thresh=th[1])
            ma = measured_angaccel(fb, darkness_thresh=th[0], red_thresh=th[1])
            w_by_thr[th].append(mw["omega"]); a_by_thr[th].append(ma["alpha"])
            row.append(f"a={ma['alpha']:+.4f}(nv{ma['n_valid']})")
        gt_w.append(wb); gt_a.append(ab)
        print(f"scene{s:05d} GT w0={wb:+.4f} alpha={ab:+.4f} | rmax={rmax:.3f} | alpha@thr: {row}")

    print("\n=== OMEGA: measured(H_b) vs clip omega0, by threshold ===")
    for th in thresholds:
        c = _corr(gt_w, w_by_thr[th])
        print(f"  thr{th}: " + (f"n={c[0]}/{len(gt_w)} rho={c[1]:+.3f} sign={c[2]:.2f} magr={c[3]:.3f}" if c else "too few"))
    print("\n=== ALPHA: measured(H_b) vs clip alpha, by threshold (THE test) ===")
    for th in thresholds:
        c = _corr(gt_a, a_by_thr[th])
        print(f"  thr{th}: " + (f"n={c[0]}/{len(gt_a)} rho={c[1]:+.3f} sign={c[2]:.2f} magr={c[3]:.3f}" if c else "too few"))


if __name__ == "__main__":
    main()
