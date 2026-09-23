"""Tests for the Step-2 velocity-first clean moving-ball dataset and analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.ball_tracking import ball_centroids, measured_velocity
from src.data.moving_ball import MovingBall, state_dim


# ---------------------------------------------------------------------------
# Generator correctness
# ---------------------------------------------------------------------------

def test_shapes_and_range():
    gen = MovingBall(image_size=128, num_frames=32, scenario="constant_velocity", seed=0)
    clip = gen.generate(0)
    assert clip.frames.shape == (32, 3, 128, 128)
    assert clip.frames.min() >= 0.0 and clip.frames.max() <= 1.0
    assert clip.state.shape == (32, state_dim())
    assert len(clip.state_keys) == state_dim()


def test_determinism():
    g1 = MovingBall(seed=0)
    g2 = MovingBall(seed=0)
    c1, c2 = g1.generate(5), g2.generate(5)
    assert torch.allclose(c1.frames, c2.frames)
    assert torch.allclose(c1.state, c2.state)


def test_ball_stays_in_frame():
    """The ball must stay fully within the frame for all 32 frames."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="constant_velocity", seed=42)
    for idx in range(10):
        clip = gen.generate(idx)
        keys = clip.state_keys
        px = clip.state[:, keys.index("obj0_pos_x")]
        py = clip.state[:, keys.index("obj0_pos_y")]
        r  = clip.state[0, keys.index("obj0_radius")]
        assert (px >= r).all() and (px <= 1.0 - r).all(), f"clip {idx}: ball x out of frame"
        assert (py >= r).all() and (py <= 1.0 - r).all(), f"clip {idx}: ball y out of frame"


def test_constant_velocity_is_constant():
    """Velocity must be the same in every frame (constant-velocity scenario)."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="constant_velocity", seed=7)
    clip = gen.generate(0)
    keys = clip.state_keys
    vx = clip.state[:, keys.index("obj0_vel_x")]
    vy = clip.state[:, keys.index("obj0_vel_y")]
    assert torch.allclose(vx, vx[0].expand_as(vx))
    assert torch.allclose(vy, vy[0].expand_as(vy))


def test_acceleration_is_zero():
    """Acceleration (vel - prev_vel) must be 0 everywhere for constant-velocity clips."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="constant_velocity", seed=3)
    clip = gen.generate(0)
    keys = clip.state_keys
    ax = clip.state[:, keys.index("obj0_acc_x")]
    ay = clip.state[:, keys.index("obj0_acc_y")]
    assert torch.allclose(ax, torch.zeros_like(ax))
    assert torch.allclose(ay, torch.zeros_like(ay))


def test_white_background():
    """The corners of every frame (well away from the ball) should be white (~1.0)."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="constant_velocity", seed=0)
    clip = gen.generate(0)
    # top-left 10x10 corner should be very close to white unless the ball is there
    corner = clip.frames[:, :, :10, :10]
    assert corner.mean() > 0.95, "background is not white"


def test_occlusion_hidden_frames():
    """Some frames must be hidden (visible==0) in the occlusion scenario."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="occlusion", seed=1)
    for idx in range(5):
        clip = gen.generate(idx)
        keys = clip.state_keys
        vis = clip.state[:, keys.index("obj0_visible")]
        assert (vis == 0).any(), f"occlusion clip {idx}: no hidden frames"
        assert (vis == 1).any(), f"occlusion clip {idx}: no visible frames"


def test_rotated_fixed_speed():
    """All 'rotated' clips should have the same speed (the fixed_speed)."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="rotated", fixed_speed=0.022, seed=2)
    speeds = [gen.generate(i).meta["speed"] for i in range(6)]
    for s in speeds:
        assert abs(s - 0.022) < 1e-6, f"rotated speed not fixed: {s}"


def test_rotated_direction_varies():
    """Direction should vary across 'rotated' clips (golden-ratio coverage)."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="rotated", seed=2)
    angles = [gen.generate(i).meta["angle"] for i in range(8)]
    diffs = [abs(angles[i] - angles[j]) for i in range(len(angles)) for j in range(i+1, len(angles))]
    assert max(diffs) > 0.5, "direction does not vary across rotated clips"


# ---------------------------------------------------------------------------
# Ball tracking
# ---------------------------------------------------------------------------

def test_tracking_matches_gt():
    """Pixel-tracked velocity should match the exact ground-truth velocity to within 1e-3."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="constant_velocity", seed=0)
    for idx in range(4):
        clip = gen.generate(idx)
        meas = measured_velocity(clip.frames)
        assert meas["n_valid"] > 0
        assert abs(meas["vel_x"] - clip.meta["vel_x"]) < 1e-3, \
            f"clip {idx}: tracked vel_x={meas['vel_x']:.4f} vs GT {clip.meta['vel_x']:.4f}"
        assert abs(meas["vel_y"] - clip.meta["vel_y"]) < 1e-3, \
            f"clip {idx}: tracked vel_y={meas['vel_y']:.4f} vs GT {clip.meta['vel_y']:.4f}"


def test_tracking_nan_during_occlusion():
    """ball_centroids should return NaN for hidden (occluded) frames."""
    gen = MovingBall(image_size=128, num_frames=32, scenario="occlusion", seed=1)
    for idx in range(4):
        clip = gen.generate(idx)
        if clip.meta.get("n_hidden_frames", 0) == 0:
            continue
        keys = clip.state_keys
        vis = clip.state[:, keys.index("obj0_visible")].numpy()
        centroids = ball_centroids(clip.frames)
        hidden_idx = np.where(vis == 0)[0]
        assert np.isnan(centroids[hidden_idx]).all(), \
            f"clip {idx}: centroid not NaN on hidden frames {hidden_idx}"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def test_dataset_registry():
    from src.data.dataset_registry import MovingBallDataset
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "name": "moving_ball", "root": None, "image_size": 64, "num_frames": 16, "fps": 8,
        "scenario": "constant_velocity", "num_clips": 8, "categories": "all", "split": "train",
        "speed_range": [0.010, 0.035], "radius_range": [0.07, 0.10], "fixed_speed": 0.022,
        "camera_rotation": False, "seed": 0,
    })
    ds = MovingBallDataset(cfg, encoder_image_size=64, encoder_frames=16)
    assert len(ds) == 8
    item = ds[0]
    assert "frames" in item and "state" in item and "state_keys" in item
    assert item["frames"].shape[0] == 16  # num_frames
    assert item["state"].shape[1] == state_dim()


def test_velocity_probe_module():
    """Smoke-test velocity_probe._build_reps without needing a real latent cache."""
    from src.training.velocity_probe import _clip_targets
    T, D = 16, 4
    state_keys = [
        "obj0_pos_x", "obj0_pos_y", "obj0_vel_x", "obj0_vel_y",
        "obj0_acc_x", "obj0_acc_y", "obj0_radius", "obj0_speed", "obj0_angle", "obj0_visible",
        "gravity", "collision_event",
    ]
    state = np.zeros((T, len(state_keys)), dtype=np.float32)
    state[:, state_keys.index("obj0_vel_x")] = 0.02
    state[:, state_keys.index("obj0_vel_y")] = 0.01
    state[:, state_keys.index("obj0_speed")] = float(np.sqrt(0.02**2 + 0.01**2))
    state[:, state_keys.index("obj0_angle")] = float(np.arctan2(0.01, 0.02))
    tgts = _clip_targets(state, state_keys)
    assert abs(tgts["vel"][0] - 0.02) < 1e-5
    assert abs(tgts["vel"][1] - 0.01) < 1e-5
    assert abs(tgts["speed"][0] - np.sqrt(0.02**2 + 0.01**2)) < 1e-5
