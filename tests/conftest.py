"""Shared pytest fixtures: tiny configs, a mock encoder, and a small latent cache."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tiny_cfg():
    """A minimal experiment config (overrides the smoke config to be even smaller/faster)."""
    cfg = load_config("configs/train/smoke_synthetic.yaml")
    cfg.encoder.hidden_dim = 64
    cfg.encoder.num_layers = 3
    cfg.encoder.num_heads = 4
    cfg.encoder.image_size = 64
    cfg.decoder.hidden_dim = 64
    cfg.decoder.depth = 2
    cfg.decoder.heads = 4
    cfg.decoder.num_query_tokens_per_frame = 16
    cfg.decoder.out_image_size = 32
    cfg.data.image_size = 32
    cfg.data.num_frames = 8
    cfg.data.num_clips = 6
    cfg.data.scenarios = ["bouncing_ball", "projectile"]
    cfg.decoder.out_num_frames = 8
    cfg.optim.max_steps = 4
    cfg.optim.warmup_steps = 1
    cfg.train.batch_size = 2
    cfg.train.log_every = 1
    cfg.train.ckpt_every = 100
    return cfg


@pytest.fixture(scope="session")
def latent_cache(tiny_cfg, tmp_path_factory):
    """Build a small latent cache once for the test session."""
    from src.data import build_dataset
    from src.encoders import build_encoder
    from src.encoders.feature_extractor import extract_latents

    out = tmp_path_factory.mktemp("latents")
    encoder = build_encoder(tiny_cfg.encoder)
    dataset = build_dataset(tiny_cfg.data, encoder_image_size=tiny_cfg.encoder.image_size,
                            encoder_frames=tiny_cfg.encoder.num_frames)
    extract_latents(encoder, dataset, out, layers=tiny_cfg.encoder.layers, batch_size=2,
                    device="cpu", store_frames_size=tiny_cfg.decoder.out_image_size, shard_size=4)
    return str(out)
