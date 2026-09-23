"""Decoder output shapes for modes A (frames) and C (state), plus baselines."""

from __future__ import annotations

import torch

from src.decoders import build_decoder


def _fake_latents(cfg, batch=2):
    enc_dim = cfg.encoder.hidden_dim
    tp = cfg.encoder.num_frames // cfg.encoder.tubelet_size
    hp = wp = cfg.encoder.image_size // cfg.encoder.patch_size
    grid = (tp, hp, wp)
    n = tp * hp * wp
    layers = {li: torch.rand(batch, n, enc_dim) for li in range(cfg.encoder.num_layers)}
    return layers, grid


def test_transformer_decoder_frames(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.decoder.mode = "reconstruct"
    dec = build_decoder(cfg.decoder, cfg.encoder.hidden_dim, state_dim=0)
    latents, grid = _fake_latents(cfg)
    out = dec(latents, grid)
    assert out.frames.shape == (2, cfg.decoder.out_num_frames, 3,
                                cfg.decoder.out_image_size, cfg.decoder.out_image_size)
    assert float(out.frames.min()) >= 0 and float(out.frames.max()) <= 1


def test_transformer_decoder_state(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.decoder.mode = "state"
    dec = build_decoder(cfg.decoder, cfg.encoder.hidden_dim, state_dim=11)
    latents, grid = _fake_latents(cfg)
    out = dec(latents, grid)
    assert out.state.shape == (2, cfg.decoder.out_num_frames, 11)


def test_conv_baseline_frames(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.decoder.name = "conv"
    dec = build_decoder(cfg.decoder, cfg.encoder.hidden_dim, state_dim=0)
    latents, grid = _fake_latents(cfg)
    out = dec(latents, grid)
    assert out.frames.shape[-1] == cfg.decoder.out_image_size


def test_state_decoder(tiny_cfg):
    cfg = tiny_cfg.copy()
    cfg.decoder.name = "state"
    dec = build_decoder(cfg.decoder, cfg.encoder.hidden_dim, state_dim=9)
    latents, grid = _fake_latents(cfg)
    out = dec(latents, grid)
    assert out.state.shape == (2, cfg.decoder.out_num_frames, 9)


def test_gradients_flow(tiny_cfg):
    cfg = tiny_cfg.copy()
    dec = build_decoder(cfg.decoder, cfg.encoder.hidden_dim, state_dim=5)
    latents, grid = _fake_latents(cfg)
    out = dec(latents, grid)
    loss = out.frames.mean() + out.state.mean()
    loss.backward()
    assert any(p.grad is not None for p in dec.parameters())
