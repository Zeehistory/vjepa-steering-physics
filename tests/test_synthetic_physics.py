"""Synthetic physics generator: determinism, shapes, and physical sanity."""

from __future__ import annotations

import torch

from src.data.synthetic_physics import SyntheticPhysics, state_dim_for


def test_determinism():
    g1 = SyntheticPhysics(image_size=32, num_frames=8, scenarios=["bouncing_ball"], seed=0)
    g2 = SyntheticPhysics(image_size=32, num_frames=8, scenarios=["bouncing_ball"], seed=0)
    c1, c2 = g1.generate(3), g2.generate(3)
    assert torch.allclose(c1.frames, c2.frames)
    assert torch.allclose(c1.state, c2.state)


def test_shapes_and_range():
    gen = SyntheticPhysics(image_size=48, num_frames=12, scenarios=["projectile"], seed=1)
    clip = gen.generate(0)
    assert clip.frames.shape == (12, 3, 48, 48)
    assert clip.frames.min() >= 0 and clip.frames.max() <= 1
    assert clip.state.shape[0] == 12
    assert len(clip.state_keys) == clip.state.shape[1]


def test_projectile_falls_under_gravity():
    """A projectile's vertical position should increase (fall, +y down) on average over the clip."""
    gen = SyntheticPhysics(image_size=64, num_frames=16, scenarios=["projectile"], seed=2)
    clip = gen.generate(0)
    keys = clip.state_keys
    yi = keys.index("obj0_pos_y")
    y = clip.state[:, yi]
    # under downward gravity the object eventually descends relative to its apex
    assert y[-1] > y.min()


def test_collision_event_recorded():
    gen = SyntheticPhysics(image_size=64, num_frames=24, scenarios=["collision"], seed=0)
    clip = gen.generate(0)
    ci = clip.state_keys.index("collision_event")
    assert clip.state[:, ci].max() == 1.0  # the two balls collide


def test_state_dim_helper():
    assert state_dim_for(["bouncing_ball"]) < state_dim_for(["collision"])
