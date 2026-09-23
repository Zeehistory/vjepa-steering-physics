"""Decoder abstraction and shared output contracts for modes A/B/C/D.

All decoders consume a batch of frozen latents (``layers: {idx: (B, L, D)}`` plus a token grid) and
produce a :class:`DecoderOutput`. The four research modes:

* **A reconstruct** — latents -> reconstructed frames ``(B, T, C, H, W)``.
* **B future**      — context latents -> future frames (same output type; supervised on future frames).
* **C state**       — latents -> physical state ``(B, T, state_dim)``.
* **D diagram**     — latents -> simplified physical visualization (trajectory / arrows / masks),
                      realized as channels in the frame output and decoded by visualization utilities.

Modes A/B/D share the frame-output head; mode C uses a state head. A single decoder can produce both
(``frames`` and ``state``) when configured, which is how we co-train reconstruction + state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DecoderOutput:
    frames: torch.Tensor | None = None       # (B, T, C, H, W) in [0, 1]
    state: torch.Tensor | None = None        # (B, T, state_dim)
    aux: dict[str, Any] = field(default_factory=dict)


class DecoderBase(ABC, nn.Module):
    """Base class for all decoders."""

    def __init__(self, cfg: Any, encoder_hidden_dim: int, state_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_hidden_dim = encoder_hidden_dim
        self.state_dim = state_dim
        self.mode = cfg.mode

    @abstractmethod
    def forward(self, latents: dict[int, torch.Tensor], grid: tuple[int, int, int]) -> DecoderOutput:
        """Map per-layer latent tokens to a :class:`DecoderOutput`.

        Parameters
        ----------
        latents: ``{layer_index: (B, L, D)}``.
        grid: ``(T', Hp, Wp)`` token grid for folding tokens to a spatiotemporal layout.
        """

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
