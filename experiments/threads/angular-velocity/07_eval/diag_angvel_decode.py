#!/usr/bin/env python
"""Diagnostic: is the DECODED rotation honest-trackable, and at what tracker thresholds?

The decopt smoke shows the soft tracker converges (marker sweeps at omega_b) but the honest
`measured_angvel` returns NaN even on the UNSTEERED decode -- its hard thresholds (darkness>0.5,
redness>0.25) are tuned for crisp GT frames while the decoder renders a lower-contrast marker. This script
decodes the TRUE H_a and TRUE H_b full latents for a handful of test scenes and reports measured_angvel at
several thresholds vs the GT omega, plus marker redness / bar darkness stats. If a relaxed threshold makes
measured_angvel(decode(H_b)) ~= GT omega, the fix is tracker calibration (not a decoder retrain).
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
from pathlib import Path

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_angvel
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_scenes", type=int, default=10)
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
    hb_gt, hb_by_thr = [], {th: [] for th in thresholds}
    ha_by_thr = {th: [] for th in thresholds}

    @torch.no_grad()
    def decode(sample):
        Ha = _to_dev(sample, layers, dev)
        grid = tuple(int(x) for x in sample["grid"])
        out = decoder(Ha, grid)
        return None if out.frames is None else out.frames[0].cpu()

    for s in scene_ids:
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        wa, wb = float(vo.clip_angvel(sa)[0]), float(vo.clip_angvel(sb)[0])
        fa, fb = decode(sa), decode(sb)
        if fb is None:
            print(f"scene{s:05d}: H_b decode None"); continue
        # marker redness / bar darkness stats on H_b decode
        r, g, b = fb[:, 0], fb[:, 1], fb[:, 2]
        redness = (r - torch.maximum(g, b)).clamp(min=0)
        gray = fb.mean(1); dark = (1 - gray).clamp(min=0)
        rmax, rmean_top = float(redness.max()), float(redness.flatten().topk(50).values.mean())
        dmax = float(dark.max())
        row = []
        for th in thresholds:
            mb = measured_angvel(fb, darkness_thresh=th[0], red_thresh=th[1])
            ma = measured_angvel(fa, darkness_thresh=th[0], red_thresh=th[1]) if fa is not None else {"omega": float("nan"), "n_valid": 0}
            hb_by_thr[th].append(mb["omega"]); ha_by_thr[th].append(ma["omega"])
            row.append(f"{mb['omega']:+.4f}(nv{mb['n_valid']})")
        hb_gt.append(wb)
        print(f"scene{s:05d} GT wa={wa:+.4f} wb={wb:+.4f} | rmax={rmax:.3f} r_top50={rmean_top:.3f} "
              f"dmax={dmax:.3f} | H_b omega@thr: {row}")

    print("\n=== correlation of measured(H_b decode) vs GT omega_b, by threshold ===")
    gt = np.array(hb_gt)
    for th in thresholds:
        d = np.array(hb_by_thr[th]); ok = np.isfinite(d)
        n = int(ok.sum())
        if n >= 2:
            rho = float(np.corrcoef(gt[ok], d[ok])[0, 1])
            magr = float(np.median(np.abs(d[ok]) / (np.abs(gt[ok]) + 1e-9)))
            sgn = float(np.mean(np.sign(d[ok]) == np.sign(gt[ok])))
            print(f"  thr{th}: n_valid={n}/{len(gt)} rho={rho:+.3f} sign_acc={sgn:.2f} mag_ratio={magr:.3f}")
        else:
            print(f"  thr{th}: n_valid={n}/{len(gt)} (too few)")


if __name__ == "__main__":
    main()
