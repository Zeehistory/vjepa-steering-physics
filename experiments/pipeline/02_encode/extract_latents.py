

#!/usr/bin/env python
"""Extract frozen-encoder latents from a dataset and cache them to disk.

Example
-------
    python experiments/pipeline/02_encode/extract_latents.py \
        --dataset physics_iq --encoder vjepa2_large --layers all \
        --output_dir outputs/latents/physics_iq/vjepa2_large

The mock encoder runs offline on CPU:
    python experiments/pipeline/02_encode/extract_latents.py --config configs/train/smoke_synthetic.yaml \
        --output_dir outputs/smoke/latents
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


from src.data import build_dataset
from src.encoders import build_encoder
from src.encoders.feature_extractor import extract_latents
from src.utils.config import load_config, to_container
from src.utils.gpu import resolve_device
from src.utils.reproducibility import run_metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="experiment YAML (provides encoder/data blocks)")
    p.add_argument("--encoder", default=None, help="encoder config name under configs/encoder/")
    p.add_argument("--dataset", default=None, help="data config name under configs/data/")
    p.add_argument("--layers", default=None, help="'all' or comma-separated layer indices")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--shard_size", type=int, default=64)
    p.add_argument("--store_frames_size", type=int, default=None)
    p.add_argument("--latent_dtype", default="float32", choices=["float32", "float16"],
                   help="storage dtype for latents; float16 halves cache size (upcast on read)")
    p.add_argument("--frames_dtype", default="float32", choices=["float32", "float16", "uint8"],
                   help="storage dtype for target frames; uint8 is lossless for uint8-sourced video")
    p.add_argument("--no_store_frames", action="store_true",
                   help="skip target frames entirely (for splits used only to fit probes/operators)")
    p.add_argument("overrides", nargs="*", help="extra key=value config overrides")
    args = p.parse_args()

    from omegaconf import OmegaConf

    cfg = load_config(args.config, args.overrides)
    if args.encoder:
        cfg.encoder = OmegaConf.merge(cfg.encoder, OmegaConf.load(f"configs/encoder/{args.encoder}.yaml"))
    if args.dataset:
        cfg.data = OmegaConf.merge(cfg.data, OmegaConf.load(f"configs/data/{args.dataset}.yaml"))
    if args.layers:
        cfg.encoder.layers = "all" if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    # Re-apply the dotlist AFTER the --encoder/--dataset YAMLs are merged in. load_config applies
    # overrides against the defaults, but those merges land on top and silently win, so a
    # `data.num_clips=800` passed alongside `--dataset foo` was being clobbered by foo.yaml's own
    # num_clips. An override the caller typed explicitly has to beat a file default.
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))

    device = resolve_device(cfg.encoder.device)
    encoder = build_encoder(cfg.encoder)
    dataset = build_dataset(cfg.data, encoder_image_size=cfg.encoder.image_size,
                            encoder_frames=cfg.encoder.num_frames)
    store_size = args.store_frames_size if args.store_frames_size is not None else cfg.decoder.out_image_size

    out = extract_latents(
        encoder, dataset, args.output_dir,
        layers=cfg.encoder.layers, batch_size=args.batch_size, device=device,
        shard_size=args.shard_size, store_frames_size=store_size,
        latent_dtype=args.latent_dtype, frames_dtype=args.frames_dtype,
        store_frames=not args.no_store_frames,
        extract_meta={"config": to_container(cfg.encoder), "provenance": run_metadata()},
    )
    print(f"[extract_latents] wrote latent cache -> {out}")


if __name__ == "__main__":
    main()
