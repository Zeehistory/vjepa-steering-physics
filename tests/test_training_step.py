"""Training smoke: a few real steps reduce loss; checkpoint round-trips; probe controls behave."""

from __future__ import annotations

from pathlib import Path

import torch

from src.decoders import DecoderLoss, build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training import train_decoder
from src.training.checkpoints import load_checkpoint, save_checkpoint


def test_train_decoder_runs_and_checkpoints(tiny_cfg, latent_cache, tmp_path):
    cfg = tiny_cfg.copy()
    cfg.latent_dir = latent_cache
    cfg.output_dir = str(tmp_path / "run")
    summary = train_decoder(cfg)
    assert Path(summary["checkpoint"]).exists()
    assert summary["steps"] >= cfg.optim.max_steps
    # provenance written
    assert (Path(cfg.output_dir) / "run_meta.json").exists()
    assert (Path(cfg.output_dir) / "resolved_config.yaml").exists()


def test_single_step_reduces_loss(tiny_cfg, latent_cache):
    cfg = tiny_cfg.copy()
    ds = LatentDataset(latent_cache, layers=cfg.encoder.layers)
    enc_dim = ds.records[0]["hidden_dim"]
    state_dim = ds.records[0]["state_dim"]
    cfg.decoder.state_dim = state_dim
    dec = build_decoder(cfg.decoder, enc_dim, state_dim)
    loss_fn = DecoderLoss(cfg.loss)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    batch = latent_collate([ds[i] for i in range(len(ds))])
    grid = tuple(int(x) for x in batch["grid"])
    latents = {int(k): v for k, v in batch["layers"].items()}

    losses = []
    for _ in range(5):
        out = dec(latents, grid)
        loss, _ = loss_fn(out.frames, batch["frames"], out.state, batch["state"],
                          batch["state_mask"], batch["state_keys"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_checkpoint_roundtrip(tiny_cfg, latent_cache, tmp_path):
    cfg = tiny_cfg.copy()
    ds = LatentDataset(latent_cache, layers=cfg.encoder.layers)
    enc_dim = ds.records[0]["hidden_dim"]
    dec = build_decoder(cfg.decoder, enc_dim, ds.records[0]["state_dim"])
    path = save_checkpoint(tmp_path / "ckpt.pt", dec, step=7)
    dec2 = build_decoder(cfg.decoder, enc_dim, ds.records[0]["state_dim"])
    step = load_checkpoint(path, dec2)
    assert step == 7
    for p1, p2 in zip(dec.parameters(), dec2.parameters(), strict=True):
        assert torch.allclose(p1, p2)
