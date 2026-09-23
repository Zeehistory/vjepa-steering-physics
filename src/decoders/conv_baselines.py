"""Convolutional / UNet-style decoder baseline.

Reshapes the latent token grid ``(B, T', Hp, Wp, D)`` into a feature map and upsamples to frames with a
conv stack. This is the "is a transformer actually needed?" baseline — the repo must prove the
transformer decoder improves over simpler decoders, so this one is a real, trainable model.

Output: ``(B, T_out, C, H, W)`` in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DecoderBase, DecoderOutput


class _UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConvVideoDecoder(DecoderBase):
    def __init__(self, cfg: Any, encoder_hidden_dim: int, state_dim: int) -> None:
        super().__init__(cfg, encoder_hidden_dim, state_dim)
        self.out_size = int(cfg.out_image_size)
        self.out_ch = int(cfg.out_channels)
        self.out_frames = int(cfg.out_num_frames)
        base = int(cfg.hidden_dim) // 4
        self.proj = nn.Conv2d(encoder_hidden_dim, base * 4, 1)
        # number of x2 upsamples is inferred at runtime from the token grid
        self.up = nn.ModuleList([_UpBlock(base * 4, base * 2), _UpBlock(base * 2, base), _UpBlock(base, base)])
        self.to_rgb = nn.Conv2d(base, self.out_ch, 3, padding=1)

    def forward(self, latents: dict[int, torch.Tensor], grid: tuple[int, int, int]) -> DecoderOutput:
        tp, hp, wp = grid
        # average requested layers, fold tokens to a (B*T', D, Hp, Wp) map
        x = torch.stack([latents[i] for i in sorted(latents)], 0).mean(0)  # (B, L, D)
        b, _l, d = x.shape
        fmap = x[:, : tp * hp * wp].reshape(b * tp, hp, wp, d).permute(0, 3, 1, 2)
        h = self.proj(fmap)
        cur = hp
        for up in self.up:
            if cur >= self.out_size:
                break
            h = up(h)
            cur *= 2
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        rgb = torch.sigmoid(self.to_rgb(h))  # (B*T', C, H, W)
        frames = rgb.reshape(b, tp, self.out_ch, self.out_size, self.out_size)
        if tp != self.out_frames:
            frames = F.interpolate(
                frames.permute(0, 2, 1, 3, 4), size=(self.out_frames, self.out_size, self.out_size),
                mode="trilinear", align_corners=False,
            ).permute(0, 2, 1, 3, 4)
        return DecoderOutput(frames=frames)
