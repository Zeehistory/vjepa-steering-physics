"""Regression tests for the per-scene-randomized-appearance 2D-acceleration variant
(scene_accel2d_mixed) -- the apples-to-apples transfer test requested by the PI.

The same-scene contract of scene_accel2d is unchanged (one shared start, one shared initial velocity v0,
K distinct constant acceleration VECTORs), so H_b - H_a still isolates Delta a. The ONE difference is that
appearance (ball radius, ball colour, background shade) is sampled once per scene and held fixed across
ranks, but VARIES across scenes. We assert the physics contract still holds AND that appearance is fixed
within a scene but varies across scenes.
"""
from __future__ import annotations

import numpy as np

from src.analysis.ball_tracking import measured_acceleration
from src.data.moving_ball import MovingBall

K = 8


def _gen(seed: int = 0):
    return MovingBall(image_size=128, num_frames=16, fps=4, scenario="scene_accel2d_mixed",
                      clips_per_scene=K, speed_range=(0.008, 0.016),
                      accel_range=(0.0015, 0.0035), radius_range=(0.08, 0.12), seed=seed)


def _scene(scene_idx: int = 0, seed: int = 0):
    g = _gen(seed)
    return [g.generate(scene_idx * K + r) for r in range(K)], g


def _col(clip, name):
    return clip.state[:, clip.state_keys.index(name)].numpy()


def _ball_hue(clip):
    """Mean R and B of the ball pixels (frame 0). Colour ramps dark-blue (R<B) -> dark-red (R>B),
    so (R - B) is a signed hue that varies across scenes; a plain darkness scalar does not."""
    f0 = np.asarray(clip.frames)[0]              # (C, H, W)
    gray = f0.mean(axis=0)                       # (H, W)
    mask = gray < 0.5                            # ball pixels (darker than any background)
    r = float(f0[0][mask].mean())
    b = float(f0[2][mask].mean())
    return round(r - b, 3)


def test_shared_initial_position_and_velocity():
    """Frame-0 center and initial velocity are shared across ranks (the same-scene pair contract)."""
    clips, _ = _scene()
    px = {round(float(_col(c, "obj0_pos_x")[0]), 9) for c in clips}
    py = {round(float(_col(c, "obj0_pos_y")[0]), 9) for c in clips}
    vx = {round(float(_col(c, "obj0_vel_x")[0]), 9) for c in clips}
    vy = {round(float(_col(c, "obj0_vel_y")[0]), 9) for c in clips}
    assert len(px) == 1 and len(py) == 1, f"start position must be shared: {px}, {py}"
    assert len(vx) == 1 and len(vy) == 1, f"initial velocity must be shared: {vx}, {vy}"


def test_distinct_acceleration_per_rank():
    """Each rank has a distinct constant 2D acceleration vector."""
    clips, _ = _scene()
    accs = {(round(float(_col(c, "obj0_acc_x")[0]), 7),
             round(float(_col(c, "obj0_acc_y")[0]), 7)) for c in clips}
    assert len(accs) == K, f"expected {K} distinct accelerations, got {len(accs)}"


def test_acceleration_constant_and_recoverable():
    """The pixel tracker recovers the acceleration direction from the rendered frames (< 5 deg),
    even with the ball colour / background varying per scene."""
    clips, _ = _scene()
    for c in clips:
        a = np.array([c.meta["acc_x"], c.meta["acc_y"]])
        m = measured_acceleration(c.frames)
        arec = np.array([m["acc_x"], m["acc_y"]])
        cos = arec @ a / (np.linalg.norm(arec) * np.linalg.norm(a) + 1e-12)
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 5.0


def test_in_frame():
    """Every quadratic path stays in frame (no long blank stretches)."""
    clips, _ = _scene()
    for c in clips:
        gray = c.frames.numpy().mean(axis=1)  # (T,H,W)
        dark_present = (gray < 0.5).reshape(gray.shape[0], -1).any(axis=1)
        assert dark_present.all(), "ball must be visible (in-frame) in every frame"


def test_appearance_fixed_within_scene():
    """Radius / colour / background are identical across the ranks of one scene."""
    clips, _ = _scene()
    radii = {round(float(_col(c, "obj0_radius")[0]), 6) for c in clips}
    assert len(radii) == 1, f"radius must be fixed within a scene: {radii}"
    # background = a frame-0 corner pixel; ball colour = darkest pixel of frame 0
    bgs = {round(float(np.asarray(c.frames)[0, :, 0, 0].mean()), 4) for c in clips}
    balls = {_ball_hue(c) for c in clips}
    assert len(bgs) == 1, f"background must be fixed within a scene: {bgs}"
    assert len(balls) == 1, f"ball colour must be fixed within a scene: {balls}"


def test_appearance_varies_across_scenes():
    """Radius / colour / background vary across scenes (the whole point of the mixed variant)."""
    g = _gen()
    radii, bgs, balls = set(), set(), set()
    for s in range(24):
        c = g.generate(s * K)  # rank 0 of scene s
        radii.add(round(float(_col(c, "obj0_radius")[0]), 4))
        bgs.add(round(float(np.asarray(c.frames)[0, :, 0, 0].mean()), 3))
        balls.add(_ball_hue(c))
    assert len(radii) > 5, f"radius should vary across scenes: {len(radii)} distinct"
    assert len(bgs) > 5, f"background should vary across scenes: {len(bgs)} distinct"
    assert len(balls) > 5, f"ball colour should vary across scenes: {len(balls)} distinct"


def test_ball_darker_than_background():
    """Ball stays clearly darker than background so the darkness>0.5 tracker keeps working."""
    g = _gen()
    for s in range(24):
        c = g.generate(s * K)
        f0 = np.asarray(c.frames)[0]
        bg = float(f0[:, 0, 0].mean())
        ball = float(f0.reshape(3, -1).min(axis=1).mean())
        assert ball < 0.5 < bg, f"scene {s}: need ball({ball:.2f})<0.5<bg({bg:.2f})"
