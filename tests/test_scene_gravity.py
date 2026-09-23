"""Regression tests for the gravity / projectile scene variant (scene_gravity, Step 2 "Bravo").

The canonical specialization of scene_accel2d: within a scene every clip shares ONE start AND ONE
roughly-horizontal launch velocity v0; each rank has a distinct constant DOWNWARD acceleration
g = (0, |g|) whose DIRECTION is fixed (+y is down) and only whose MAGNITUDE varies across ranks. So
H_b - H_a isolates Delta g (how much faster the ball falls) on a single vertical axis. We assert that
contract -- shared start/v0, fixed-down direction, distinct magnitudes, horizontal launch -- plus the
usual constant-accel / parabola-recovers / in-frame / determinism guarantees.
"""
from __future__ import annotations

import numpy as np

from src.analysis.ball_tracking import measured_acceleration
from src.data.moving_ball import MovingBall

K = 8


def _gen(seed: int = 0):
    return MovingBall(image_size=128, num_frames=16, fps=4, scenario="scene_gravity",
                      clips_per_scene=K, speed_range=(0.008, 0.016),
                      gravity_range=(0.0015, 0.0040), radius_range=(0.08, 0.12), seed=seed)


def _scene(scene_idx: int = 0, seed: int = 0):
    g = _gen(seed)
    return [g.generate(scene_idx * K + r) for r in range(K)], g


def _col(clip, name):
    return clip.state[:, clip.state_keys.index(name)].numpy()


def test_shared_initial_position():
    """Frame-0 ball CENTER is identical across all ranks (one start position per scene)."""
    clips, _ = _scene()
    px = {round(float(_col(c, "obj0_pos_x")[0]), 9) for c in clips}
    py = {round(float(_col(c, "obj0_pos_y")[0]), 9) for c in clips}
    assert len(px) == 1 and len(py) == 1, f"start position must be shared across ranks: {px}, {py}"


def test_shared_launch_velocity():
    """Frame-0 launch velocity v0 is identical across all ranks (only the gravity varies)."""
    clips, _ = _scene()
    v0 = {(round(float(_col(c, "obj0_vel_x")[0]), 9), round(float(_col(c, "obj0_vel_y")[0]), 9))
          for c in clips}
    assert len(v0) == 1, f"launch velocity must be shared across ranks: {v0}"


def test_launch_is_roughly_horizontal():
    """v0 is a sideways throw: the horizontal component dominates the vertical wobble."""
    clips, _ = _scene()
    vx0, vy0 = float(_col(clips[0], "obj0_vel_x")[0]), float(_col(clips[0], "obj0_vel_y")[0])
    assert abs(vx0) > abs(vy0), f"launch should be roughly horizontal: v0=({vx0},{vy0})"


def test_gravity_is_downward_and_magnitude_only_varies():
    """Every rank's acceleration points straight DOWN (+y, acc_x==0); only |g| varies across ranks."""
    clips, _ = _scene()
    for c in clips:
        assert np.isclose(c.meta["acc_x"], 0.0, atol=1e-12), f"gravity must have no x: {c.meta['acc_x']}"
        assert c.meta["acc_y"] > 0.0, f"gravity must point down (+y): {c.meta['acc_y']}"
    mags = {round(float(c.meta["acc_y"]), 7) for c in clips}
    assert len(mags) == K, f"expected {K} distinct gravity magnitudes, got {len(mags)}"


def test_acceleration_is_constant_within_clip():
    """The (obj0_acc_x, obj0_acc_y) columns are constant over the clip and match the meta value."""
    clips, _ = _scene()
    for c in clips:
        ax, ay = _col(c, "obj0_acc_x"), _col(c, "obj0_acc_y")
        assert np.allclose(ax, ax[0]) and np.allclose(ay, ay[0]), "acceleration must be constant"
        assert np.isclose(ax[0], c.meta["acc_x"]) and np.isclose(ay[0], c.meta["acc_y"])


def test_velocity_evolves_linearly_with_gravity():
    """Per-frame velocity follows v_t = v0 + g*t (the discrete constant-acceleration roll-out)."""
    clips, _ = _scene()
    for c in clips:
        vx, vy = _col(c, "obj0_vel_x"), _col(c, "obj0_vel_y")
        ax, ay = c.meta["acc_x"], c.meta["acc_y"]
        t = np.arange(len(vx))
        assert np.allclose(vx, vx[0] + ax * t, atol=1e-9)
        assert np.allclose(vy, vy[0] + ay * t, atol=1e-9)


def test_position_is_quadratic_and_parabola_recovers_gravity():
    """Fitting a parabola to the GT centroid recovers g = 2*c2 (the metric the tracker uses)."""
    clips, _ = _scene()
    for c in clips:
        px, py = _col(c, "obj0_pos_x"), _col(c, "obj0_pos_y")
        t = np.arange(len(px), dtype=float)
        A = np.stack([np.ones_like(t), t, t ** 2], axis=1)
        cx = np.linalg.lstsq(A, px, rcond=None)[0]
        cy = np.linalg.lstsq(A, py, rcond=None)[0]
        assert np.isclose(2 * cx[2], c.meta["acc_x"], atol=1e-7)
        assert np.isclose(2 * cy[2], c.meta["acc_y"], atol=1e-7)


def test_measured_acceleration_on_rendered_frames():
    """The pixel tracker recovers the (downward) gravity direction from rendered frames (< 5 deg)."""
    clips, _ = _scene()
    for c in clips:
        a = np.array([c.meta["acc_x"], c.meta["acc_y"]])
        m = measured_acceleration(c.frames)
        arec = np.array([m["acc_x"], m["acc_y"]])
        cos = arec @ a / (np.linalg.norm(arec) * np.linalg.norm(a) + 1e-12)
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 5.0


def test_scene_accels_metadata_consistent():
    """Every rank carries the same scene-level gravity set, and its own g matches its rank entry."""
    clips, _ = _scene()
    sa = [c.meta["scene_accels"] for c in clips]
    assert all(np.allclose(s, sa[0]) for s in sa), "all ranks must share the scene gravity set"
    for r, c in enumerate(clips):
        ax, ay = c.meta["scene_accels"][r]
        assert np.isclose(ax, c.meta["acc_x"]) and np.isclose(ay, c.meta["acc_y"])


def test_ball_stays_in_frame_all_ranks():
    """The whole disk stays inside [r, 1-r] for every rank and every frame (feasible shared start)."""
    clips, _ = _scene()
    r = clips[0].state[0, clips[0].state_keys.index("obj0_radius")].item()
    for c in clips:
        px, py = _col(c, "obj0_pos_x"), _col(c, "obj0_pos_y")
        assert px.min() >= r - 1e-6 and px.max() <= 1 - r + 1e-6, "ball left frame in x"
        assert py.min() >= r - 1e-6 and py.max() <= 1 - r + 1e-6, "ball left frame in y"


def test_gravity_varies_across_scenes():
    """Across scenes the gravity set actually changes (not a fixed pattern)."""
    first = {tuple(np.round(np.ravel(_scene(scene_idx=s)[0][0].meta["scene_accels"]), 6))
             for s in range(10)}
    assert len(first) > 1, "gravity sets should vary across scenes"


def test_determinism():
    """Same index + seed reproduces an identical clip (bit-for-bit frames + state)."""
    a, _ = _scene(scene_idx=3)
    b, _ = _scene(scene_idx=3)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca.frames.numpy(), cb.frames.numpy())
        assert np.array_equal(ca.state.numpy(), cb.state.numpy())
