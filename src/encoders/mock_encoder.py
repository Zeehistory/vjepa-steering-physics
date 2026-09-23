"""Deterministic mock V-JEPA encoder.

A frozen, randomly-initialized ViT-style video encoder that produces V-JEPA2-shaped
:class:`LatentBundle` outputs. Its purpose is twofold:

1. Let the *entire* pipeline run offline on CPU/MPS with no weight downloads.
2. Provide a controlled encoder whose latents we understand, so we can write sanity tests (e.g. a
   linear probe should recover physical state that we deliberately inject into early-layer features).

To make the mock useful for probing tests (not just shape tests), the patch embedding is a fixed
random projection of the *actual pixels*, so latents genuinely depend on the input video and carry
decodable spatial information. Everything is deterministic given ``cfg`` (seeded).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import EncoderWrapper, LatentBundle, TokenGrid


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class MockVJEPAEncoder(EncoderWrapper):
    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        gen = torch.Generator().manual_seed(1234)
        self._dim = int(cfg.hidden_dim)
        self._layers = int(cfg.num_layers)
        self.patch = int(cfg.patch_size)
        self.tubelet = int(cfg.tubelet_size)
        self.heads = int(cfg.num_heads)

        in_dim = 3 * self.patch * self.patch * self.tubelet
        # Fixed random patch projection so latents depend on real pixels (deterministic).
        proj = torch.empty(in_dim, self._dim)
        nn.init.normal_(proj, std=in_dim**-0.5, generator=gen)
        self.register_buffer("patch_proj", proj)

        self.blocks = nn.ModuleList([_Block(self._dim, self.heads) for _ in range(self._layers)])
        # Deterministic init for all block params.
        self._deterministic_init(seed=4321)
        self.freeze()

    def _deterministic_init(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, generator=gen)
            else:
                nn.init.zeros_(p)

    @property
    def hidden_dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._layers

    def _patchify(self, video: torch.Tensor) -> tuple[torch.Tensor, TokenGrid]:
        """``(B, T, C, H, W)`` -> tokens ``(B, L, in_dim)`` and the token grid."""
        b, t, c, h, w = video.shape
        tp, p = self.tubelet, self.patch
        t2 = (t // tp) * tp
        video = video[:, :t2]
        hp, wp = h // p, w // p
        x = video.reshape(b, t2 // tp, tp, c, hp, p, wp, p)
        # (B, T', Hp, Wp, tp, C, p, p) -> flatten patch content
        x = x.permute(0, 1, 4, 6, 2, 3, 5, 7).contiguous()
        x = x.reshape(b, (t2 // tp) * hp * wp, tp * c * p * p)
        return x, TokenGrid(temporal=t2 // tp, height=hp, width=wp)

    def forward_features(
        self, video: torch.Tensor, layers: list[int], extract_attention: bool
    ) -> LatentBundle:
        tokens, grid = self._patchify(video)
        x = tokens.to(self.patch_proj.dtype) @ self.patch_proj  # (B, L, D)

        out_layers: dict[int, torch.Tensor] = {}
        pooled: dict[int, torch.Tensor] = {}
        attention: dict[int, torch.Tensor] = {}
        want = set(layers)
        for i, blk in enumerate(self.blocks):
            x, attn_w = blk(x, need_weights=extract_attention and i in want)
            if i in want:
                out_layers[i] = x
                pooled[i] = x.mean(dim=1)
                if extract_attention and attn_w is not None:
                    attention[i] = attn_w
        return LatentBundle(
            layers=out_layers,
            grid=grid,
            hidden_dim=self._dim,
            pooled=pooled,
            attention=attention,
            meta={"encoder": "mock", "model_id": "mock-vjepa", "frozen": self.frozen},
        )
