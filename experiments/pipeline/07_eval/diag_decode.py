#!/usr/bin/env python
"""One-off diagnostic: is the moving-ball decoder reconstructing the ball, or collapsing to blank?

Decodes the first few clips (alpha=0, unperturbed) and reports, per clip:
  * decoded frame pixel stats (min/max/mean) and per-frame darkest pixel + dark-mass
  * ground-truth frame stats + dark-mass + measured_velocity(GT)  (does the tracker even work on GT?)
  * measured_velocity(decoded)
Also dumps a GT-vs-decoded recon grid PNG so we can eyeball it.
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

from src.analysis import visualization as viz
from src.analysis.ball_tracking import measured_velocity, ball_centroids
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    ds = LatentDataset(args.latent_dir, layers=cfg.encoder.layers)
    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    step = load_checkpoint(args.checkpoint, decoder, map_location=dev)
    print(f"[diag] loaded checkpoint step={step}  layers_loaded={cfg.encoder.layers}")

    def dark_mass(fr, thr=0.5):
        g = fr.float().mean(1)              # (T,H,W)
        d = (1.0 - g).clamp(min=0.0)
        d = torch.where(d > (1.0 - thr), d, torch.zeros_like(d))
        return d.sum(dim=(1, 2))            # (T,)

    for i in range(min(args.num_samples, len(ds))):
        batch = latent_collate([ds[i]])
        sid = batch["id"][0]
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(dev) for k, v in batch["layers"].items()}
        gt = batch["frames"][0]             # (T,C,H,W)
        dec = decoder(latents, grid).frames[0].cpu()

        print(f"\n=== clip {i} id={sid} grid={grid} ===")
        print(f"  GT     : shape={tuple(gt.shape)} min={gt.min():.3f} max={gt.max():.3f} "
              f"mean={gt.mean():.3f}  darkmass[min,max]={dark_mass(gt).min():.1f},{dark_mass(gt).max():.1f}")
        print(f"  DECODED: shape={tuple(dec.shape)} min={dec.min():.3f} max={dec.max():.3f} "
              f"mean={dec.mean():.3f}  darkmass[min,max]={dark_mass(dec).min():.1f},{dark_mass(dec).max():.1f}")
        # darkest pixel per frame (how dark does the decoded ball get?)
        print(f"  DECODED darkest-pixel per frame (first 8): "
              f"{[round(float(dec[t].mean(0).min()),3) for t in range(min(8, dec.shape[0]))]}")
        print(f"  measured_velocity(GT)      = {measured_velocity(gt)}")
        print(f"  measured_velocity(DECODED) = {measured_velocity(dec)}")
        # adaptive-threshold tracker test on decoded: use relative darkness
        g = dec.float().mean(1)
        bg = g.flatten(1).median(1).values.view(-1, 1, 1)   # per-frame background level
        rel = (bg - g).clamp(min=0.0)                        # how much darker than bg
        print(f"  DECODED rel-darkness max per frame (first 8): "
              f"{[round(float(rel[t].max()),3) for t in range(min(8, rel.shape[0]))]}")
        viz.reconstruction_grid(gt, dec, out / f"{sid}_diag_recon.png")

    print(f"\n[diag] wrote recon grids -> {out}")


if __name__ == "__main__":
    main()
