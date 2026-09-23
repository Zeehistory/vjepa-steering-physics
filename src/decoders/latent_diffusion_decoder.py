"""Latent-diffusion refinement decoder (deferred stub).

Planned Stage-7 component: a conditional diffusion head that refines the transformer decoder's frame
output for sharper, more temporally-coherent reconstructions, conditioned on the frozen V-JEPA latents
(cross-attention conditioning, v-prediction, DDPM/DDIM sampling). Kept as a typed stub with the
intended interface so configs and training code can reference it before the heavy implementation lands.
"""

from __future__ import annotations

from typing import Any

from .base import DecoderBase, DecoderOutput


class LatentDiffusionDecoder(DecoderBase):
    def __init__(self, cfg: Any, encoder_hidden_dim: int, state_dim: int) -> None:
        super().__init__(cfg, encoder_hidden_dim, state_dim)
        raise NotImplementedError(
            "LatentDiffusionDecoder is a Stage-7 stub. Intended design: conditional v-prediction "
            "diffusion over frame latents, cross-attending frozen V-JEPA tokens, with DDIM sampling. "
            "See docs/method.md (roadmap)."
        )

    def forward(self, latents, grid) -> DecoderOutput:  # pragma: no cover - stub
        raise NotImplementedError

    def sample(self, latents, grid, steps: int = 50) -> DecoderOutput:  # pragma: no cover - stub
        """Planned sampling entry point (DDIM)."""
        raise NotImplementedError
