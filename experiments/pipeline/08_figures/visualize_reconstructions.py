#!/usr/bin/env python
"""Render reconstruction grids, error maps, trajectory overlays, and panel videos for a trained decoder.

Example
-------
    python experiments/pipeline/08_figures/visualize_reconstructions.py --config configs/train/smoke_synthetic.yaml \
        --latent_dir outputs/smoke/latents --checkpoint outputs/smoke/runs/decoder/checkpoints/last.pt \
        --output_dir outputs/smoke/viz
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

import torch

from src.analysis import visualization as viz
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def visualize(cfg, latent_dir: str, checkpoint: str, output_dir: str | Path,
              num_samples: int = 3, device: str = "cpu") -> Path:
    dataset = LatentDataset(latent_dir, layers=cfg.encoder.layers)
    rec0 = dataset.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(dataset.available_layers())
    load_checkpoint(checkpoint, decoder, map_location=device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(num_samples, len(dataset))):
        batch = latent_collate([dataset[i]])
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}
        result = decoder(latents, grid)
        target = batch["frames"][0]
        sid = batch["id"][0]
        if result.frames is not None:
            recon = result.frames[0].cpu()
            viz.reconstruction_grid(target, recon, out / f"{sid}_recon_grid.png")
            viz.error_map(target.unsqueeze(0)[0], recon, out / f"{sid}_error_map.png")
            viz.panel_video(target, recon, out / f"{sid}_panel.mp4", fps=cfg.data.fps)
        if batch["state_mask"].sum() > 0:
            viz.trajectory_overlay(target, batch["state"][0], batch["state_keys"],
                                   out / f"{sid}_trajectory.png")
    print(f"[visualize] wrote panels -> {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    visualize(cfg, args.latent_dir, args.checkpoint, args.output_dir, args.num_samples, args.device)


if __name__ == "__main__":
    main()
