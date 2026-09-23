"""Encoder abstraction and the latent contract shared by mock and real wrappers.

Every encoder wrapper returns a :class:`LatentBundle`. The mock encoder and the real V-JEPA2 wrapper
produce *the same* structure and shapes, which is what makes "use real weights" a config swap rather
than a code change (asserted by a test).

Token layout. V-JEPA tokenizes a clip of ``T`` frames into spatiotemporal patches: ``T' = T //
tubelet`` temporal positions and ``N = (H // patch) * (W // patch)`` spatial tokens per temporal
position, flattened to ``L = T' * N`` tokens. A :class:`LatentBundle` stores, per requested layer, a
tensor of shape ``(B, L, D)`` plus the grid metadata needed to fold tokens back to ``(B, T', Hp, Wp,
D)`` for the decoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TokenGrid:
    """Spatiotemporal token grid metadata for one clip batch."""

    temporal: int  # T' temporal positions
    height: int    # Hp spatial token rows
    width: int     # Wp spatial token cols

    @property
    def num_tokens(self) -> int:
        return self.temporal * self.height * self.width


@dataclass
class LatentBundle:
    """Frozen-encoder output.

    Attributes
    ----------
    layers:
        Mapping ``layer_index -> (B, L, D)`` tensor of token features for each requested block.
    grid:
        :class:`TokenGrid` describing how ``L`` factorizes into ``(T', Hp, Wp)``.
    pooled:
        Optional ``(B, D)`` global/CLS-style pooled representation per layer
        (``layer_index -> (B, D)``), when the model exposes one.
    attention:
        Optional ``layer_index -> (B, heads, L, L)`` attention maps (only if requested/available).
    hidden_dim:
        Feature width ``D``.
    meta:
        Free-form provenance (model id, checkpoint, preprocessing, commit).
    """

    layers: dict[int, torch.Tensor]
    grid: TokenGrid
    hidden_dim: int
    pooled: dict[int, torch.Tensor] = field(default_factory=dict)
    attention: dict[int, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self.layers.keys())

    def fold(self, layer: int) -> torch.Tensor:
        """Return layer tokens folded to ``(B, T', Hp, Wp, D)``."""
        x = self.layers[layer]
        b, _, d = x.shape
        g = self.grid
        return x.reshape(b, g.temporal, g.height, g.width, d)

    def to(self, device: torch.device | str) -> LatentBundle:
        return LatentBundle(
            layers={k: v.to(device) for k, v in self.layers.items()},
            grid=self.grid,
            hidden_dim=self.hidden_dim,
            pooled={k: v.to(device) for k, v in self.pooled.items()},
            attention={k: v.to(device) for k, v in self.attention.items()},
            meta=self.meta,
        )


class EncoderWrapper(ABC, torch.nn.Module):
    """Base class for frozen V-JEPA-style encoders.

    Subclasses implement :meth:`forward_features`. The encoder is frozen by default; ``parameters`` are
    set ``requires_grad=False`` and ``eval()`` mode is enforced in :meth:`encode`.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.frozen = bool(getattr(cfg, "frozen", True))

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def forward_features(
        self, video: torch.Tensor, layers: list[int], extract_attention: bool
    ) -> LatentBundle:
        """Run the encoder and return a :class:`LatentBundle`.

        Parameters
        ----------
        video: ``(B, T, C, H, W)`` encoder-preprocessed input.
        layers: which block outputs to return.
        extract_attention: whether to also return attention maps.
        """

    def resolve_layers(self, layers: Any) -> list[int]:
        if layers == "all" or layers is None:
            return list(range(self.num_layers))
        return [int(x) for x in layers]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(
        self,
        video: torch.Tensor,
        layers: Any = "all",
        extract_attention: bool = False,
    ) -> LatentBundle:
        """Frozen-mode feature extraction (no grad). Use ``forward_features`` if fine-tuning."""
        if self.frozen:
            self.eval()
        layer_idx = self.resolve_layers(layers)
        return self.forward_features(video, layer_idx, extract_attention)
