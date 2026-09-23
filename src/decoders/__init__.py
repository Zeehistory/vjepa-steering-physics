"""Decoders: transformer video decoder (main), state decoder, conv/diffusion baselines, losses."""

from __future__ import annotations

from typing import Any

from .base import DecoderBase, DecoderOutput
from .conv_baselines import ConvVideoDecoder
from .loss_functions import DecoderLoss
from .state_decoder import LinearStateDecoder, StateDecoder
from .transformer_video_decoder import TransformerVideoDecoder


def build_decoder(cfg: Any, encoder_hidden_dim: int, state_dim: int) -> DecoderBase:
    """Instantiate a decoder by ``cfg.name``."""
    name = cfg.name
    builders = {
        "transformer": TransformerVideoDecoder,
        "conv": ConvVideoDecoder,
        "state": StateDecoder,
        "linear_state": LinearStateDecoder,
    }
    if name in builders:
        return builders[name](cfg, encoder_hidden_dim, state_dim)
    if name == "diffusion":
        from .latent_diffusion_decoder import LatentDiffusionDecoder

        return LatentDiffusionDecoder(cfg, encoder_hidden_dim, state_dim)
    raise KeyError(f"Unknown decoder '{name}'. Known: transformer, conv, state, linear_state, diffusion.")


__all__ = [
    "DecoderBase",
    "DecoderOutput",
    "TransformerVideoDecoder",
    "ConvVideoDecoder",
    "StateDecoder",
    "LinearStateDecoder",
    "DecoderLoss",
    "build_decoder",
]
