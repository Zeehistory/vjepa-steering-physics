"""Validate the per-frame centroid / position loss against the dataset's own renderer.

These run on CPU in <1s and are the cheap gate before any GPU training: they confirm the soft
centroid uses the SAME (x=width, y=height, origin top-left, [0,1]) convention as the moving-ball
renderer, that the loss is ~0 for a correctly-placed ball and large for a static path-smear, and that
gradients flow back into the rendered pixels (so it can actually move the ball during training).
"""

from __future__ import annotations

import torch

from src.data.moving_ball import MovingBall
from src.decoders.loss_functions import (
    _soft_ball_centroid,
    frame_position_loss,
    frame_spread_loss,
)


def _real_clip():
    """One ground-truth moving-ball clip: frames (1,T,C,H,W), state (1,T,S), keys."""
    gen = MovingBall(image_size=64, num_frames=16, scenario="constant_velocity", seed=1)
    clip = gen.generate(0)
    frames = clip.frames.unsqueeze(0)          # (1,T,C,H,W)
    state = clip.state.unsqueeze(0)            # (1,T,S)
    return frames, state, clip.state_keys


def test_soft_centroid_tracks_gt_position():
    frames, state, keys = _real_clip()
    cen, mass, _ = _soft_ball_centroid(frames)          # (1,T,2)
    xi, yi = keys.index("obj0_pos_x"), keys.index("obj0_pos_y")
    gt = torch.stack([state[..., xi], state[..., yi]], dim=-1)
    err = (cen - gt).abs().max().item()
    # the dark ball dominates the dark^2-weighted centroid; a 64px disk recovers GT pos to ~<2px
    assert err < 0.03, f"centroid deviates from GT position by {err:.4f} (>0.03)"
    assert (mass > 0).all(), "every frame should carry dark mass (the ball is always visible here)"


def test_position_loss_zero_for_correct_ball():
    frames, state, keys = _real_clip()
    loss = frame_position_loss(frames, state, keys).item()
    assert loss < 1e-3, f"position loss should be ~0 for the true render, got {loss:.5f}"


def test_position_loss_large_for_static_smear():
    """A frame-averaged smear (the failure mode) must score MUCH worse than the true moving render."""
    frames, state, keys = _real_clip()
    smear = frames.mean(dim=1, keepdim=True).expand_as(frames).contiguous()  # static path-average
    good = frame_position_loss(frames, state, keys).item()
    bad = frame_position_loss(smear, state, keys).item()
    assert bad > 10 * max(good, 1e-6), f"smear={bad:.5f} not >> good={good:.5f}"


def test_spread_loss_penalizes_smear_not_disk():
    frames, state, keys = _real_clip()
    smear = frames.mean(dim=1, keepdim=True).expand_as(frames).contiguous()
    assert frame_spread_loss(frames).item() <= frame_spread_loss(smear).item()


def test_gradient_flows_to_pixels():
    frames, state, keys = _real_clip()
    pred = frames.clone().requires_grad_(True)
    loss = frame_position_loss(pred, state, keys) + frame_spread_loss(pred)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0, "no gradient reached the pixels"


def test_accepts_both_namings():
    """obj0_pos_* (this dataset) and bare pos_* (synthetic_physics single-object) both resolve."""
    frames, state, keys = _real_clip()                       # keys are obj0_pos_*
    bare = [k.replace("obj0_", "") for k in keys]             # -> pos_*
    a = frame_position_loss(frames, state, keys).item()
    b = frame_position_loss(frames, state, bare).item()
    assert a > 0 and abs(a - b) < 1e-9, "both namings should give the identical (nonzero) result"
