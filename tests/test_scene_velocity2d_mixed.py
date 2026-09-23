"""Regression tests for the appearance-mixed 2D-velocity scene variant (scene_velocity2d_mixed).

Same within-scene velocity contract as scene_velocity2d (one shared start, K distinct velocity VECTORs
so H_b - H_a isolates Delta v), but the appearance — ball radius, ball colour, background shade — is
sampled ONCE per scene, held fixed across the scene's ranks, and VARIES across scenes. We assert the
velocity invariants still hold, the appearance is constant within a scene, it actually varies across
scenes, and every ball stays trackable (clearly darker than its background).
"""
from __future__ import annotations

import numpy as np

from src.data.moving_ball import MovingBall

K = 8


def _gen(seed: int = 0):
    return MovingBall(image_size=128, num_frames=16, fps=4, scenario="scene_velocity2d_mixed",
                      clips_per_scene=K, speed_range=(0.012, 0.024),
                      radius_range=(0.08, 0.13), seed=seed)


def _scene(scene_idx: int = 0, seed: int = 0):
    g = _gen(seed)
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
    assert len({round(c.meta["angle"], 4) for c in clips}) == K, "directions must vary across ranks"
    assert len({round(c.meta["speed"], 6) for c in clips}) == K, "speeds must vary across ranks"


def test_appearance_fixed_within_scene():
    """Radius, ball colour and background are identical across every rank of a scene."""
    clips, _ = _scene()
    radii = {round(c.meta["radius"], 9) for c in clips}
    colors = {tuple(round(v, 9) for v in c.meta["ball_color"]) for c in clips}
    bgs = {tuple(round(v, 9) for v in c.meta["bg_color"]) for c in clips}
    assert len(radii) == 1, f"radius must be fixed within a scene: {radii}"
    assert len(colors) == 1, f"ball colour must be fixed within a scene: {colors}"
    assert len(bgs) == 1, f"background must be fixed within a scene: {bgs}"


def test_appearance_varies_across_scenes():
    """Across scenes the appearance actually changes (radius / colour / background not all constant)."""
    n = 12
    radii, colors, bgs = set(), set(), set()
    for s in range(n):
        c = _scene(scene_idx=s)[0][0]
        radii.add(round(c.meta["radius"], 6))
        colors.add(tuple(round(v, 6) for v in c.meta["ball_color"]))
        bgs.add(tuple(round(v, 6) for v in c.meta["bg_color"]))
    assert len(radii) > 1, "radius should vary across scenes"
    assert len(colors) > 1, "ball colour should vary across scenes"
    assert len(bgs) > 1, "background should vary across scenes"


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
    clips, _ = _scene()
    r = clips[0].meta["radius"]
    for c in clips:
        px = c.state[:, c.state_keys.index("obj0_pos_x")].numpy()
        py = c.state[:, c.state_keys.index("obj0_pos_y")].numpy()
        assert px.min() >= r - 1e-6 and px.max() <= 1 - r + 1e-6, "ball left frame in x"
        assert py.min() >= r - 1e-6 and py.max() <= 1 - r + 1e-6, "ball left frame in y"


def test_ball_stays_trackable_under_mixed_appearance():
    """Every rank's ball stays clearly darker than its (possibly grey) background, across many scenes."""
    for s in range(12):
        clips, _ = _scene(scene_idx=s)
        for c in clips:
            lum = c.frames[0].numpy().mean(0)  # (H, W) luminance
            assert lum.min() < 0.5, f"scene {s}: ball not dark enough to track (min lum {lum.min():.2f})"
            assert (lum < 0.5).mean() < 0.2, "ball should be a small dark blob on a brighter background"


def test_determinism():
    """Same index + seed reproduces an identical clip (bit-for-bit frames + state)."""
    a, _ = _scene(scene_idx=3)
    b, _ = _scene(scene_idx=3)
    for ca, cb in zip(a, b):
        assert np.array_equal(ca.frames.numpy(), cb.frames.numpy())
        assert np.array_equal(ca.state.numpy(), cb.state.numpy())
