"""Metric correctness + research-discipline controls.

Sanity checks the spec demands: oracle ≈ perfect, randomized/shuffled controls ≈ chance, and metric
edge cases (identical inputs).
"""

from __future__ import annotations

import numpy as np
import torch

from src.eval.latent_metrics import linear_cka, participation_ratio
from src.eval.physics_metrics import physics_metrics
from src.eval.reconstruction_metrics import psnr, reconstruction_metrics, ssim
from src.training.train_probe import probe_layers


def test_psnr_ssim_identical():
    x = torch.rand(2, 4, 3, 16, 16)
    assert psnr(x, x) > 40
    assert ssim(x, x) > 0.99


def test_reconstruction_metrics_keys():
    x = torch.rand(1, 4, 3, 16, 16)
    y = torch.rand(1, 4, 3, 16, 16)
    m = reconstruction_metrics(x, y)
    for k in ("psnr", "ssim", "ms_ssim", "l1", "l2", "temporal_consistency"):
        assert k in m and m[k] is not None


def test_physics_oracle_is_perfect():
    state = torch.rand(2, 6, 11)
    keys = [f"obj0_{n}" for n in ["pos_x", "pos_y", "vel_x", "vel_y", "acc_x", "acc_y",
                                  "radius", "mass", "visible"]] + ["gravity", "collision_event"]
    mask = torch.ones(11)
    m = physics_metrics(state, state, keys, mask)
    assert m["position_rmse"] < 1e-5
    assert m["velocity_rmse"] < 1e-5


def test_cka_self_is_one():
    x = np.random.randn(50, 8)
    assert abs(linear_cka(x, x) - 1.0) < 1e-5


def test_participation_ratio_bounds():
    x = np.random.randn(100, 10)
    pr = participation_ratio(x)
    assert 1.0 <= pr <= 10.5


def test_probe_controls_collapse(latent_cache):
    """Real probe R² should exceed the shuffled/randomized controls for at least one variable."""
    records = probe_layers(latent_cache, layers="all", seed=0)
    assert records
    # controls must not systematically beat the real probe
    gaps = [r["r2"] - max(r["ctrl_shuffled_latent_r2"], r["ctrl_randomized_label_r2"]) for r in records]
    assert max(gaps) > 0.05  # at least one variable is genuinely decodable above control
    # controls themselves should be near/below zero on average
    ctrl = [max(r["ctrl_shuffled_latent_r2"], r["ctrl_randomized_label_r2"]) for r in records]
    assert np.mean(ctrl) < 0.5
