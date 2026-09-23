

#!/usr/bin/env python
"""Train a decoder on cached frozen latents.

Example
-------
    accelerate launch experiments/pipeline/03_train/train_decoder.py \
        --config configs/train/physics_iq_transformer_large.yaml \
        --latent_dir outputs/latents/physics_iq/vjepa2_large \
        --output_dir outputs/runs/physics_iq_decoder_large

For the offline smoke run use experiments/pipeline/00_common/run_full_pipeline.py (which extracts latents first).
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


from src.training import train_decoder
from src.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--encoder", default=None, help="(metadata only) encoder name used for latents")
    p.add_argument("--latent_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.latent_dir:
        cfg.latent_dir = args.latent_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir

    summary = train_decoder(cfg)
    print(f"[train_decoder] done: {summary}")


if __name__ == "__main__":
    main()
