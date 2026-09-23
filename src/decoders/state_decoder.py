"""Mode-C state decoder: frozen latents -> per-frame physical state.

A focused decoder that maps latent tokens directly to the physical state vector (position, velocity,
acceleration, gravity, collision, ...). Unlike the lightweight linear/MLP *probes* (which read a single
pooled vector and exist to measure decodability), this is a small transformer that aggregates the full
token grid and predicts state per output frame — useful when state prediction is the end goal rather
than only a measurement.

Output: ``(B, T_out, state_dim)``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import DecoderBase, DecoderOutput
from .transformer_video_decoder import _DecoderBlock, _sincos_1d


class StateDecoder(DecoderBase):
    def __init__(self, cfg: Any, encoder_hidden_dim: int, state_dim: int) -> None:
        super().__init__(cfg, encoder_hidden_dim, state_dim)
        if state_dim <= 0:
            raise ValueError("StateDecoder requires state_dim > 0 (dataset must expose GT state).")
        dim = int(cfg.hidden_dim)
        self.dim = dim
        self.out_frames = int(cfg.out_num_frames)
        self.latent_proj = nn.Linear(encoder_hidden_dim, dim)
        self.frame_queries = nn.Parameter(torch.randn(self.out_frames, dim) * 0.02)
        self.register_buffer("query_pos", _sincos_1d(self.out_frames, dim), persistent=False)
        self.blocks = nn.ModuleList(
            [_DecoderBlock(dim, int(cfg.heads), float(cfg.mlp_ratio), float(cfg.dropout))
             for _ in range(int(cfg.depth))]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, state_dim))

    def _memory(self, latents: dict[int, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.latent_proj(latents[i]) for i in sorted(latents)], dim=1)

    def forward(self, latents: dict[int, torch.Tensor], grid: tuple[int, int, int]) -> DecoderOutput:
        b = next(iter(latents.values())).shape[0]
        memory = self._memory(latents)
        q = (self.frame_queries + self.query_pos).unsqueeze(0).expand(b, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, memory)
        return DecoderOutput(state=self.head(q))


class LinearStateDecoder(DecoderBase):
    """Trivial pooled-linear state decoder; a strong-but-simple Mode-C baseline."""

    def __init__(self, cfg: Any, encoder_hidden_dim: int, state_dim: int) -> None:
        super().__init__(cfg, encoder_hidden_dim, state_dim)
        self.out_frames = int(cfg.out_num_frames)
        self.head = nn.Linear(encoder_hidden_dim, state_dim * self.out_frames)
        self.state_dim = state_dim

    def forward(self, latents: dict[int, torch.Tensor], grid: tuple[int, int, int]) -> DecoderOutput:
        pooled = torch.stack([latents[i].mean(dim=1) for i in sorted(latents)], 0).mean(0)
        b = pooled.shape[0]
        out = self.head(pooled).reshape(b, self.out_frames, self.state_dim)
        return DecoderOutput(state=out)
