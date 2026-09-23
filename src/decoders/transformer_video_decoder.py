"""Transformer video decoder: learned query tokens cross-attend frozen V-JEPA latents.

The main model. Architecture:

1. **Latent memory.** Each requested encoder layer is linearly projected to ``hidden_dim`` and tagged
   with (a) a learned *layer embedding* and (b) factorized spatial + temporal positional embeddings
   derived from the token grid ``(T', Hp, Wp)``. All layers are concatenated into one memory sequence.
2. **Query tokens.** ``out_num_frames * q*q`` learned query tokens (q*q spatial queries per output
   frame) carry their own temporal + spatial positional embeddings.
3. **Decoder blocks.** Each block: query self-attention -> cross-attention into the latent memory ->
   MLP, pre-norm, residual. Optional gradient checkpointing; attention uses PyTorch SDPA.
4. **Heads.**
   * *frame head*: per-query linear to a pixel patch + a small conv refinement / learned upsample to
     ``out_image_size``  (modes A/B/D).
   * *state head*: per-frame pooled query features -> physical state (mode C). Co-trainable with frames.

Configurable depth/width/heads/dropout. Scales via ``configs/decoder/decoder_{small,base,large,huge}``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .base import DecoderBase, DecoderOutput


def _sincos_1d(length: int, dim: int) -> torch.Tensor:
    """Standard 1D sinusoidal positional embedding ``(length, dim)``."""
    pos = torch.arange(length).float().unsqueeze(1)
    omega = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    emb = torch.zeros(length, dim)
    emb[:, 0::2] = torch.sin(pos * omega)
    emb[:, 1::2] = torch.cos(pos * omega)
    return emb


class _DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm_q(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        q = q + self.cross_attn(self.norm_x(q), self.norm_m(memory), self.norm_m(memory),
                                need_weights=False)[0]
        q = q + self.mlp(self.norm_mlp(q))
        return q


class _ResBlock(nn.Module):
    """Pre-activation 3x3 residual block (GroupNorm + GELU), the workhorse of the conv head."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, ch), ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return x + h


class _ConvUpHead(nn.Module):
    """Token grid -> full-resolution frame, by progressive x2 PixelShuffle upsampling.

    The default head projects each query token straight to its own ``patch x patch`` pixel block with
    one linear layer, then tries to hide the seams with two 3x3 convs. At 256px / 16x16 queries that
    patch is 16px wide but the refinement's receptive field is only 5px, so it structurally cannot
    fix a patch boundary -- which is exactly the blockiness in the decoded frames. Here the tokens
    instead become a 16x16 FEATURE map that is upsampled x2 four times with a residual block at each
    scale, so every output pixel is produced by a stack with a wide receptive field and there are no
    patch boundaries to hide.
    """

    def __init__(self, dim: int, q: int, out_size: int, out_ch: int, width: int = 256) -> None:
        super().__init__()
        n_up = int(round(math.log2(out_size / q)))
        if 2 ** n_up * q != out_size:
            raise ValueError(f"out_image_size/{q} must be a power of two (got {out_size}/{q}).")
        self.q, self.out_size, self.out_ch = q, out_size, out_ch
        self.proj = nn.Linear(dim, width)
        chans = [max(32, width >> i) for i in range(n_up + 1)]
        stages = []
        for i in range(n_up):
            c_in, c_out = chans[i], chans[i + 1]
            stages += [_ResBlock(c_in),
                       nn.Conv2d(c_in, c_out * 4, 3, padding=1),   # x4 channels -> PixelShuffle x2
                       nn.PixelShuffle(2),
                       _ResBlock(c_out)]
        self.stages = nn.Sequential(*stages)
        self.out_norm = nn.GroupNorm(min(8, chans[-1]), chans[-1])
        self.to_rgb = nn.Conv2d(chans[-1], out_ch, 3, padding=1)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, q_tok: torch.Tensor, b: int, t: int) -> torch.Tensor:
        x = self.proj(q_tok)                                   # (B, T*q*q, width)
        x = x.reshape(b * t, self.q, self.q, -1).permute(0, 3, 1, 2).contiguous()
        x = self.stages(x)
        x = self.to_rgb(F.gelu(self.out_norm(x)))
        return x.reshape(b, t, self.out_ch, self.out_size, self.out_size)


class TransformerVideoDecoder(DecoderBase):
    def __init__(self, cfg: Any, encoder_hidden_dim: int, state_dim: int) -> None:
        super().__init__(cfg, encoder_hidden_dim, state_dim)
        dim = int(cfg.hidden_dim)
        self.dim = dim
        self.out_frames = int(cfg.out_num_frames)
        self.out_size = int(cfg.out_image_size)
        self.out_ch = int(cfg.out_channels)
        self.gradient_checkpointing = bool(cfg.gradient_checkpointing)
        self.use_layer_embedding = bool(cfg.use_layer_embedding)

        # spatial query grid q x q per frame
        qpf = int(cfg.num_query_tokens_per_frame)
        self.q = int(round(math.sqrt(qpf)))
        if self.q * self.q != qpf:
            raise ValueError("num_query_tokens_per_frame must be a perfect square (e.g. 64 -> 8x8).")
        self.patch = self.out_size // self.q
        if self.patch * self.q != self.out_size:
            raise ValueError(f"out_image_size ({self.out_size}) must be divisible by q ({self.q}).")

        # latent -> hidden projection (shared) and per-layer embedding table (sized lazily).
        self.latent_proj = nn.Linear(encoder_hidden_dim, dim)
        self.layer_embedding = nn.ParameterDict()  # str(layer_idx) -> (dim,)

        # learned query tokens + their positional embeddings
        self.query_tokens = nn.Parameter(torch.randn(self.out_frames, qpf, dim) * 0.02)

        self.blocks = nn.ModuleList(
            [_DecoderBlock(dim, int(cfg.heads), float(cfg.mlp_ratio), float(cfg.dropout))
             for _ in range(int(cfg.depth))]
        )
        self.norm_out = nn.LayerNorm(dim)

        # frame head. "patch" is the original (linear -> pixel block -> 2-conv refine) and stays the
        # default so every existing checkpoint loads unchanged. "convup" is the high-fidelity head.
        self.frame_head = str(getattr(cfg, "frame_head", "patch"))
        if self.frame_head == "convup":
            self.convup = _ConvUpHead(dim, self.q, self.out_size, self.out_ch,
                                      int(getattr(cfg, "head_width", 256)))
        elif self.frame_head == "patch":
            self.to_patch = nn.Linear(dim, self.patch * self.patch * self.out_ch)
            self.refine = nn.Sequential(
                nn.Conv2d(self.out_ch, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, self.out_ch, 3, padding=1),
            )
        else:
            raise ValueError(f"unknown frame_head '{self.frame_head}' (expected patch|convup)")

        # state head (mode C / co-training)
        self.state_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                        nn.Linear(dim, state_dim)) if state_dim > 0 else None

        # positional-embedding caches (sin/cos buffers grow on demand)
        self._register_query_pos()

    def _register_query_pos(self) -> None:
        t_emb = _sincos_1d(self.out_frames, self.dim)              # (T, dim)
        s_emb = _sincos_1d(self.q * self.q, self.dim)             # (q*q, dim)
        qpos = t_emb[:, None, :] + s_emb[None, :, :]              # (T, q*q, dim)
        self.register_buffer("query_pos", qpos, persistent=False)

    def prime_layers(self, layer_indices: list[int]) -> None:
        """Pre-create layer-embedding parameters for ``layer_indices``.

        Layer embeddings are created lazily on first forward; call this before ``load_state_dict`` so a
        freshly-built decoder has the same parameter set as the checkpoint.
        """
        if self.use_layer_embedding:
            for idx in layer_indices:
                self._layer_emb(int(idx))

    def _layer_emb(self, idx: int) -> torch.Tensor:
        key = str(idx)
        if key not in self.layer_embedding:
            p = nn.Parameter(torch.randn(self.dim, device=self.latent_proj.weight.device) * 0.02)
            self.layer_embedding[key] = p
        return self.layer_embedding[key]

    def _build_memory(self, latents: dict[int, torch.Tensor], grid: tuple[int, int, int]) -> torch.Tensor:
        tp, hp, wp = grid
        n = tp * hp * wp
        # factorized spatial+temporal pos emb for the token grid
        t_emb = _sincos_1d(tp, self.dim).to(self.latent_proj.weight.device)
        h_emb = _sincos_1d(hp, self.dim).to(self.latent_proj.weight.device)
        w_emb = _sincos_1d(wp, self.dim).to(self.latent_proj.weight.device)
        grid_pos = (t_emb[:, None, None, :] + h_emb[None, :, None, :] + w_emb[None, None, :, :])
        grid_pos = grid_pos.reshape(1, n, self.dim)

        chunks = []
        for idx in sorted(latents.keys()):
            x = self.latent_proj(latents[idx])  # (B, L, dim)
            if x.shape[1] != n:  # be robust to encoders that add tokens
                grid_pos_l = grid_pos[:, : x.shape[1]] if x.shape[1] < n else F.pad(
                    grid_pos, (0, 0, 0, x.shape[1] - n))
            else:
                grid_pos_l = grid_pos
            x = x + grid_pos_l
            if self.use_layer_embedding:
                x = x + self._layer_emb(idx).view(1, 1, -1)
            chunks.append(x)
        return torch.cat(chunks, dim=1)  # (B, num_layers * L, dim)

    def forward(self, latents: dict[int, torch.Tensor], grid: tuple[int, int, int]) -> DecoderOutput:
        b = next(iter(latents.values())).shape[0]
        memory = self._build_memory(latents, grid)

        q = (self.query_tokens + self.query_pos).reshape(1, -1, self.dim).expand(b, -1, -1).contiguous()
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                q = checkpoint(blk, q, memory, use_reentrant=False)
            else:
                q = blk(q, memory)
        q = self.norm_out(q)  # (B, T*q*q, dim)

        out = DecoderOutput()
        if self.mode in ("reconstruct", "future", "diagram"):
            out.frames = self._decode_frames(q, b)
        if self.mode == "state" or (self.state_head is not None and self.mode in ("reconstruct", "future")):
            out.state = self._decode_state(q, b)
        if out.frames is None and out.state is None:
            raise RuntimeError(f"Decoder mode '{self.mode}' produced no output.")
        return out

    def _decode_frames(self, q: torch.Tensor, b: int) -> torch.Tensor:
        if self.frame_head == "convup":
            return torch.sigmoid(self.convup(q, b, self.out_frames))
        patches = self.to_patch(q)  # (B, T*q*q, patch*patch*C)
        t, qq = self.out_frames, self.q
        patches = patches.reshape(b, t, qq, qq, self.out_ch, self.patch, self.patch)
        # (B, T, qq, qq, C, p, p) -> (B, T, C, qq*p, qq*p)
        frames = patches.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        frames = frames.reshape(b, t, self.out_ch, qq * self.patch, qq * self.patch)
        # conv refinement per frame
        flat = frames.reshape(b * t, self.out_ch, self.out_size, self.out_size)
        flat = flat + self.refine(flat)
        frames = torch.sigmoid(flat).reshape(b, t, self.out_ch, self.out_size, self.out_size)
        return frames

    def _decode_state(self, q: torch.Tensor, b: int) -> torch.Tensor:
        assert self.state_head is not None
        t = self.out_frames
        per_frame = q.reshape(b, t, -1, self.dim).mean(dim=2)  # pool spatial queries -> (B, T, dim)
        return self.state_head(per_frame)
