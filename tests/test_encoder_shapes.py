"""Encoder contract: LatentBundle shapes, layer selection, folding, frozen-ness."""

from __future__ import annotations

import torch

from src.encoders import build_encoder
from src.encoders.base import LatentBundle


def test_mock_encoder_bundle(tiny_cfg):
    enc = build_encoder(tiny_cfg.encoder)
    video = torch.rand(2, tiny_cfg.encoder.num_frames, 3, tiny_cfg.encoder.image_size,
                       tiny_cfg.encoder.image_size)
    bundle = enc.encode(video, layers="all")
    assert isinstance(bundle, LatentBundle)
    assert bundle.hidden_dim == tiny_cfg.encoder.hidden_dim
    assert len(bundle.layer_indices) == tiny_cfg.encoder.num_layers
    for li in bundle.layer_indices:
        tokens = bundle.layers[li]
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == tiny_cfg.encoder.hidden_dim
        assert tokens.shape[1] == bundle.grid.num_tokens


def test_layer_selection(tiny_cfg):
    enc = build_encoder(tiny_cfg.encoder)
    video = torch.rand(1, tiny_cfg.encoder.num_frames, 3, tiny_cfg.encoder.image_size,
                       tiny_cfg.encoder.image_size)
    bundle = enc.encode(video, layers=[0, 2])
    assert bundle.layer_indices == [0, 2]


def test_fold_shape(tiny_cfg):
    enc = build_encoder(tiny_cfg.encoder)
    video = torch.rand(2, tiny_cfg.encoder.num_frames, 3, tiny_cfg.encoder.image_size,
                       tiny_cfg.encoder.image_size)
    bundle = enc.encode(video, layers=[0])
    folded = bundle.fold(0)
    g = bundle.grid
    assert folded.shape == (2, g.temporal, g.height, g.width, bundle.hidden_dim)


def test_encoder_is_frozen(tiny_cfg):
    enc = build_encoder(tiny_cfg.encoder)
    assert all(not p.requires_grad for p in enc.parameters())


def test_determinism(tiny_cfg):
    enc = build_encoder(tiny_cfg.encoder)
    video = torch.rand(1, tiny_cfg.encoder.num_frames, 3, tiny_cfg.encoder.image_size,
                       tiny_cfg.encoder.image_size)
    a = enc.encode(video, layers=[1]).layers[1]
    b = enc.encode(video, layers=[1]).layers[1]
    assert torch.allclose(a, b)


def test_real_wrappers_share_contract():
    """Mock and real wrappers must share the EncoderWrapper / LatentBundle contract (config swap)."""
    from src.encoders.base import EncoderWrapper
    from src.encoders.mock_encoder import MockVJEPAEncoder
    from src.encoders.vjepa2_wrapper import VJEPA2Encoder

    assert issubclass(MockVJEPAEncoder, EncoderWrapper)
    assert issubclass(VJEPA2Encoder, EncoderWrapper)
    for name in ("forward_features", "hidden_dim", "num_layers"):
        assert hasattr(VJEPA2Encoder, name)
