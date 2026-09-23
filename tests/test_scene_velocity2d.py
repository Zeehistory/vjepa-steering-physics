"""Regression tests for the 2D-velocity scene variant (scene_velocity2d).

This is the true-velocity dataset for the velocity-subspace / operator experiments: within a scene all
ranks share ONE initial position, and each rank carries a distinct velocity VECTOR (direction AND speed
both vary). The same-scene difference H_b - H_a then isolates a 2D velocity difference Delta v. We assert
those invariants on the rendered clips + ground-truth state (the latents are tested downstream).
"""
from __future__ import annotations

import numpy as np

from src.data.moving_ball import MovingBall

K = 8


def _scene(scene_idx: int = 0, image_size: int = 128, radius: float = 0.11):
    g = MovingBall(image_size=image_size, num_frames=16, fps=4, scenario="scene_velocity2d",
                   clips_per_scene=K, speed_range=(0.012, 0.026),
                   radius_range=(radius, radius), seed=0)
    return [g.generate(scene_idx * K + r) for r in range(K)], g


def test_shared_initial_position():
    """Frame-0 ball CENTER is identical across all ranks (one start position per scene)."""
    clips, _ = _scene()
    px = {round(float(c.state[0, c.state_keys.index("obj0_pos_x")]), 9) for c in clips}
    py = {round(float(c.state[0, c.state_keys.index("obj0_pos_y")]), 9) for c in clips}
    assert len(px) == 1 and len(py) == 1, f"start position must be shared across ranks: {px}, {py}"


def test_eight_distinct_velocity_vectors():
    """Each rank has a distinct 2D velocity; both direction and speed vary across the scene."""
    clips, _ = _scene()
    vels = {(round(c.meta["vel_x"], 6), round(c.meta["vel_y"], 6)) for c in clips}
    assert len(vels) == K, f"expected {K} distinct velocity vectors, got {len(vels)}"
    angles = {round(c.meta["angle"], 4) for c in clips}
    speeds = {round(c.meta["speed"], 6) for c in clips}
    assert len(angles) == K, f"directions must vary across ranks: {len(angles)} distinct"
    assert len(speeds) == K, f"speeds must vary across ranks: {len(speeds)} distinct"


def test_scene_velocities_metadata_consistent():
    """Every rank carries the same scene-level velocity set, and its own vel matches its rank entry."""
    clips, _ = _scene()
    sv = [c.meta["scene_velocities"] for c in clips]
    assert all(np.allclose(s, sv[0]) for s in sv), "all ranks must share the scene velocity set"
    for r, c in enumerate(clips):
        vx, vy = c.meta["scene_velocities"][r]
        assert np.isclose(vx, c.meta["vel_x"]) and np.isclose(vy, c.meta["vel_y"])


def test_constant_velocity_per_clip():
    """Within a clip velocity is constant (zero acceleration) and matches the per-frame state."""
    clips, _ = _scene()
    for c in clips:
        ax = c.state[:, c.state_keys.index("obj0_acc_x")].numpy()
        ay = c.state[:, c.state_keys.index("obj0_acc_y")].numpy()
        assert np.allclose(ax, 0.0) and np.allclose(ay, 0.0)
        vx = c.state[:, c.state_keys.index("obj0_vel_x")].numpy()
        vy = c.state[:, c.state_keys.index("obj0_vel_y")].numpy()
        assert np.allclose(vx, vx[0]) and np.allclose(vy, vy[0])


def test_ball_stays_in_frame_all_ranks():
    """The whole disk stays inside [r, 1-r] for every rank and every frame (feasible shared start)."""
    clips, g = _scene()
    r = clips[0].meta["radius"]
    for c in clips:
        px = c.state[:, c.state_keys.index("obj0_pos_x")].numpy()
        py = c.state[:, c.state_keys.index("obj0_pos_y")].numpy()
        assert px.min() >= r - 1e-6 and px.max() <= 1 - r + 1e-6, "ball left frame in x"
        assert py.min() >= r - 1e-6 and py.max() <= 1 - r + 1e-6, "ball left frame in y"


def test_ball_stays_trackable():
    """Every rank's ball must be clearly darker than its white background (darkness>0.5 tracker)."""
    clips, _ = _scene()
    for c in clips:
        lum = c.frames[0].numpy().mean(0)  # (H, W) luminance
        assert lum.min() < 0.5, f"ball not dark enough to track (min lum {lum.min():.2f})"
        assert (lum < 0.5).mean() < 0.2, "ball should be a small dark blob on a bright background"


def test_scenes_are_distinct():
    """Different scene ids reproduce different geometry (start position and/or velocity set)."""
    s0, _ = _scene(scene_idx=0)
    s1, _ = _scene(scene_idx=1)
    p0 = (round(float(s0[0].state[0, 0]), 6), round(float(s0[0].state[0, 1]), 6))
    p1 = (round(float(s1[0].state[0, 0]), 6), round(float(s1[0].state[0, 1]), 6))
    v0 = {(round(c.meta["vel_x"], 6), round(c.meta["vel_y"], 6)) for c in s0}
    v1 = {(round(c.meta["vel_x"], 6), round(c.meta["vel_y"], 6)) for c in s1}
    assert p0 != p1 or v0 != v1, "distinct scenes should not be identical"


def test_determinism():
    """Same index + seed reproduces an identical clip (bit-for-bit frames + state)."""
    a, _ = _scene(scene_idx=3)
    b, _ = _scene(scene_idx=3)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca.frames.numpy(), cb.frames.numpy())
        assert np.array_equal(ca.state.numpy(), cb.state.numpy())
