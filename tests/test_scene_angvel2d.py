"""Regression tests for the angular-velocity scene variant (scene_angvel2d, the rotational sibling of velocity).

Within a scene every clip shares ONE rotation centre, ONE initial orientation theta0, and ONE appearance;
each rank has a distinct constant ANGULAR VELOCITY omega (signed scalar). Because frame 0 is bit-identical
across the scene, H_b - H_a isolates Delta omega. We assert that contract, that omega is genuinely constant
and recoverable from the decoded pixels by the marker tracker, and that the whole spinning object stays in
frame at every orientation. The tracker validation is the linchpin: it must recover omega on GT frames to
high precision (the analog of measured_acceleration's 0.05deg check).
"""
from __future__ import annotations

import numpy as np
import torch

from src.analysis.ball_tracking import measured_angvel, rotor_orientation
from src.analysis.velocity_ops import clip_angvel
from src.data.moving_ball import MovingBall, state_dim

K = 8


def _gen(seed: int = 0, scenario: str = "scene_angvel2d", image_size: int = 128):
    return MovingBall(image_size=image_size, num_frames=16, fps=4, scenario=scenario,
                      clips_per_scene=K, omega_range=(0.06, 0.20), radius_range=(0.11, 0.15), seed=seed)


def _scene(scene_idx: int = 0, seed: int = 0, scenario: str = "scene_angvel2d", image_size: int = 128):
    g = _gen(seed, scenario, image_size)
    return [g.generate(scene_idx * K + r) for r in range(K)], g


def _col(clip, name):
    return clip.state[:, clip.state_keys.index(name)].numpy()


def test_state_schema_extended():
    """theta + omega columns are appended; state_dim grows to accommodate them, keys stay aligned."""
    clips, _ = _scene()
    c = clips[0]
    assert c.state.shape == (16, state_dim())
    assert len(c.state_keys) == state_dim()
    assert "obj0_theta" in c.state_keys and "obj0_omega" in c.state_keys


def test_object_does_not_translate():
    """Pure rotation: the centre is fixed and shared, velocity/acceleration are exactly zero."""
    clips, _ = _scene()
    px = {round(float(_col(c, "obj0_pos_x")[0]), 9) for c in clips}
    py = {round(float(_col(c, "obj0_pos_y")[0]), 9) for c in clips}
    assert len(px) == 1 and len(py) == 1, f"rotation centre must be shared+fixed: {px},{py}"
    for c in clips:
        assert np.allclose(_col(c, "obj0_pos_x"), _col(c, "obj0_pos_x")[0])  # constant over time
        assert np.allclose(_col(c, "obj0_vel_x"), 0.0) and np.allclose(_col(c, "obj0_vel_y"), 0.0)
        assert np.allclose(_col(c, "obj0_acc_x"), 0.0) and np.allclose(_col(c, "obj0_acc_y"), 0.0)


def test_shared_initial_orientation():
    """Frame-0 orientation theta0 is identical across ranks (only omega varies)."""
    clips, _ = _scene()
    t0 = {round(float(_col(c, "obj0_theta")[0]), 9) for c in clips}
    assert len(t0) == 1, f"theta0 must be shared across ranks: {t0}"


def test_eight_distinct_signed_omegas():
    """Each rank has a distinct constant angular velocity; signs are balanced (CW and CCW present)."""
    clips, _ = _scene()
    omegas = [float(c.meta["omega"]) for c in clips]
    for c in clips:  # omega constant over the clip
        assert np.allclose(_col(c, "obj0_omega"), c.meta["omega"])
    assert len({round(o, 7) for o in omegas}) == K, f"expected {K} distinct omegas: {omegas}"
    assert any(o < 0 for o in omegas) and any(o > 0 for o in omegas), f"signs not balanced: {omegas}"
    assert all(0.06 - 1e-6 <= abs(o) <= 0.20 + 1e-6 for o in omegas), f"|omega| out of range: {omegas}"


def test_theta_advances_by_omega():
    """theta(t) = theta0 + omega*t exactly (the ground truth the tracker must recover)."""
    clips, _ = _scene()
    for c in clips:
        th = _col(c, "obj0_theta")
        t = np.arange(len(th))
        assert np.allclose(th, th[0] + c.meta["omega"] * t, atol=1e-6)


def test_clip_angvel_reads_omega():
    """velocity_ops.clip_angvel returns [omega, 0] from the packed state (the operator target)."""
    clips, _ = _scene()
    for c in clips:
        sample = {"state": c.state.numpy(), "state_keys": c.state_keys}
        w = clip_angvel(sample)
        assert w.shape == (2,) and abs(w[0] - c.meta["omega"]) < 1e-6 and w[1] == 0.0


def test_all_orientations_in_frame():
    """The whole spinning object stays inside the frame for every orientation (dark mass never clips edges)."""
    clips, _ = _scene(image_size=128)
    for c in clips:
        f = c.frames.numpy()  # (T,C,H,W)
        gray = f.mean(axis=1)  # (T,H,W)
        dark = gray < 0.5
        # no dark object pixels on the 1px border
        assert dark[:, 0, :].sum() == 0 and dark[:, -1, :].sum() == 0
        assert dark[:, :, 0].sum() == 0 and dark[:, :, -1].sum() == 0


def test_tracker_recovers_omega_on_gt():
    """THE linchpin: measured_angvel recovers omega from GT-rendered frames to high precision (<3%)."""
    clips, _ = _scene(image_size=256)  # native resolution -> best tracker precision
    for c in clips:
        m = measured_angvel(c.frames)
        assert m["n_valid"] >= 14, f"tracker lost too many frames: {m}"
        assert abs(m["omega"] - c.meta["omega"]) < 0.03 * abs(c.meta["omega"]) + 0.003, \
            f"tracked omega={m['omega']:.4f} vs GT {c.meta['omega']:.4f}"


def test_marker_separable_from_body():
    """rotor_orientation reads a finite per-frame angle from the RED marker (colour-separable)."""
    clips, _ = _scene(image_size=256)
    phi = rotor_orientation(clips[0].frames)
    assert np.isfinite(phi).mean() >= 0.9, "marker must be readable on almost every frame"


# ---- mixed-appearance variant ------------------------------------------------------------------------
def test_mixed_appearance_varies_across_scenes_fixed_within():
    """scene_angvel2d_mixed: bar colour + bg fixed within a scene, varying across scenes; marker stays red."""
    g = _gen(seed=0, scenario="scene_angvel2d_mixed", image_size=128)
    s0 = [g.generate(0 * K + r) for r in range(K)]
    s1 = [g.generate(1 * K + r) for r in range(K)]
    col0 = {tuple(c.meta["ball_color"]) for c in s0}
    bg0 = {tuple(c.meta["bg_color"]) for c in s0}
    assert len(col0) == 1 and len(bg0) == 1, "appearance must be fixed within a scene"
    assert tuple(s0[0].meta["ball_color"]) != tuple(s1[0].meta["ball_color"]) or \
        tuple(s0[0].meta["bg_color"]) != tuple(s1[0].meta["bg_color"]), "appearance must vary across scenes"
    # low-red body so the red marker stays separable
    r, gg, b = s1[0].meta["ball_color"]
    assert r <= max(gg, b) + 0.05, f"mixed body colour must stay low-red: {s1[0].meta['ball_color']}"


def test_mixed_preserves_omega_contract_and_tracking():
    """The mixed variant keeps the same-scene omega contract and stays trackable under appearance change."""
    g = _gen(seed=0, scenario="scene_angvel2d_mixed", image_size=256)
    clips = [g.generate(3 * K + r) for r in range(K)]
    t0 = {round(float(_col(c, "obj0_theta")[0]), 9) for c in clips}
    assert len(t0) == 1, "theta0 shared within scene"
    assert len({round(float(c.meta["omega"]), 7) for c in clips}) == K
    for c in clips[:3]:
        m = measured_angvel(c.frames)
        assert abs(m["omega"] - c.meta["omega"]) < 0.05 * abs(c.meta["omega"]) + 0.004, \
            f"mixed tracked omega={m['omega']:.4f} vs GT {c.meta['omega']:.4f}"
