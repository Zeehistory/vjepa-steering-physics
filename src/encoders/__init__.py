"""Frozen video encoders (mock + real V-JEPA/V-JEPA2/VGFR) and latent extraction."""

from __future__ import annotations

from typing import Any

from .base import EncoderWrapper, LatentBundle, TokenGrid
from .mock_encoder import MockVJEPAEncoder

_ENCODER_BUILDERS = {
    "mock": MockVJEPAEncoder,
}


def build_encoder(cfg: Any) -> EncoderWrapper:
    """Instantiate an encoder by ``cfg.name``.

    Real wrappers are imported lazily so the offline mock path never imports ``transformers``.
    """
    name = cfg.name
    if name in _ENCODER_BUILDERS:
        return _ENCODER_BUILDERS[name](cfg)
    if name in ("vjepa2", "vjepa2_large"):
        from .vjepa2_wrapper import VJEPA2Encoder

        return VJEPA2Encoder(cfg)
    if name in ("vjepa", "vjepa_large"):
        from .vjepa_wrapper import VJEPAEncoder

        return VJEPAEncoder(cfg)
    if name == "vgfr":
        from .vgfr_wrapper import VGFREncoder

        return VGFREncoder(cfg)
    raise KeyError(f"Unknown encoder '{name}'. Known: mock, vjepa2, vjepa, vgfr.")


__all__ = [
    "EncoderWrapper",
    "LatentBundle",
    "TokenGrid",
    "MockVJEPAEncoder",
    "build_encoder",
]
