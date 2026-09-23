"""Regression tests for the angular-ACCELERATION scene (scene_angaccel2d) -- the rotational sibling of
scene_accel2d and the second-order sibling of scene_angvel2d.

Within a scene every clip shares ONE rotation centre, ONE initial orientation theta0, ONE initial spin rate
omega0 and ONE appearance; each rank has a distinct constant ANGULAR ACCELERATION alpha (signed: spin-up vs
spin-down). Because theta(0)=theta0 and omega(0)=omega0 for every rank, frame 0 is bit-identical across the
scene and H_b - H_a isolates Delta alpha.

The linchpin is the tracker check: measured_angaccel must recover alpha from GROUND-TRUTH pixels, since
every steering number downstream is read off decoded pixels the same way. We also pin that angular
acceleration is a strict GENERALIZATION -- with alpha=0 the rotor renders exactly the constant-spin
scene_angvel2d clip, so adding it cannot have disturbed the solved angular-velocity result.
"""
from __future__ import annotations

import numpy as np
import torch

from src.analysis.ball_tracking import measured_angaccel, measured_angvel, rotor_orientation
from src.analysis.velocity_ops import clip_angaccel, clip_angvel0
from src.data.moving_ball import MovingBall, state_dim

K = 8
DARK, RED = 0.25, 0.08          # tight thresholds, as the big-object rotor decode requires


def _gen(seed: int = 0, scenario: str = "scene_angaccel2d", image_size: int = 128):
    return MovingBall(image_size=image_size, num_frames=16, fps=4, scenario=scenario,
                      clips_per_scene=K, omega0_range=(-0.06, 0.06), alpha_range=(0.005, 0.014),
                      radius_range=(0.32, 0.42), seed=seed)


def _scene(scene_idx: int = 0, seed: int = 0, scenario: str = "scene_angaccel2d"):
    g = _gen(seed, scenario)
    return [g.generate(scene_idx * K + r) for r in range(K)], g


def _col(clip, name):
    return np.asarray(clip.state)[:, list(clip.state_keys).index(name)]


# ---- the scene contract that makes H_b - H_a isolate Delta alpha --------------------------------------
def test_first_frame_identical_across_ranks():
    clips, _ = _scene()
    for c in clips[1:]:
        assert torch.equal(clips[0].frames[0], c.frames[0])


def test_scene_shares_centre_theta0_and_initial_rate_but_varies_alpha():
    clips, _ = _scene()
    centres = {(round(float(_col(c, "obj0_pos_x")[0]), 9), round(float(_col(c, "obj0_pos_y")[0]), 9))
               for c in clips}
    assert len(centres) == 1, "rotation centre must be shared across the scene"
    assert len({round(float(_col(c, "obj0_theta")[0]), 9) for c in clips} ) == 1, "theta0 must be shared"
    assert len({round(float(clip_angvel0(_as_sample(c))), 9) for c in clips}) == 1, "omega0 must be shared"
    alphas = [float(clip_angaccel(_as_sample(c))[0]) for c in clips]
    assert len({round(a, 9) for a in alphas}) == K, "every rank must have a distinct alpha"


def test_alpha_signs_are_balanced_and_decorrelated_from_rank():
    """Balanced signs stop a steer from scoring by exploiting a rank->sign shortcut."""
    pos = neg = 0
    for s in range(12):
        clips, _ = _scene(scene_idx=s)
        a = [float(clip_angaccel(_as_sample(c))[0]) for c in clips]
        assert sum(1 for x in a if x > 0) == K // 2
        pos += sum(1 for x in a[: K // 2] if x > 0)
        neg += sum(1 for x in a[: K // 2] if x < 0)
    assert pos > 0 and neg > 0, "sign must not be determined by rank ordering"


def _as_sample(clip):
    return {"state": np.asarray(clip.state), "state_keys": list(clip.state_keys)}


# ---- the state schema ---------------------------------------------------------------------------------
def test_state_schema_has_alpha_and_omega_ramps():
    clips, _ = _scene()
    c = clips[0]
    assert np.asarray(c.state).shape[1] == state_dim()
    assert "obj0_alpha" in c.state_keys
    alpha = _col(c, "obj0_alpha")
    assert np.allclose(alpha, alpha[0]), "alpha is the constant command"
    omega, theta = _col(c, "obj0_omega"), _col(c, "obj0_theta")
    t = np.arange(len(omega))
    # omega(t) = omega0 + alpha*t exactly, and theta is its integral
    np.testing.assert_allclose(omega, omega[0] + alpha[0] * t, atol=1e-5)
    np.testing.assert_allclose(theta, theta[0] + omega[0] * t + 0.5 * alpha[0] * t ** 2, atol=1e-4)


def test_translation_scenarios_leave_the_rotational_columns_at_zero():
    g = MovingBall(image_size=64, num_frames=16, scenario="scene_velocity2d", clips_per_scene=4, seed=0)
    c = g.generate(0)
    assert np.asarray(c.state).shape[1] == state_dim()
    for k in ("obj0_theta", "obj0_omega", "obj0_alpha"):
        assert np.allclose(_col(c, k), 0.0)


# ---- angular acceleration is a strict generalization of angular velocity ------------------------------
def test_alpha_zero_reproduces_the_constant_spin_rotor_exactly():
    """The angvel result must be untouched: with alpha=0 the rotor roll-out is bit-identical."""
    g = _gen()
    centre, theta0, omega0 = np.array([0.5, 0.5]), 0.7, 0.13
    kw = dict(body_color=(0.2, 0.2, 0.2), bg_color=(1.0, 1.0, 1.0), scenario="x",
              index=0, scene=0, rank=0, scene_omegas=[0.13])
    spin = g._roll_out_rotor(centre, theta0, omega0, 0.35, **kw)                 # alpha defaults to 0
    spin_explicit = g._roll_out_rotor(centre, theta0, omega0, 0.35, alpha=0.0, **kw)
    assert torch.equal(spin.frames, spin_explicit.frames)
    theta = np.asarray(spin.state)[:, list(spin.state_keys).index("obj0_theta")]
    np.testing.assert_allclose(theta, theta0 + omega0 * np.arange(len(theta)), atol=1e-5)
    m = measured_angvel(spin.frames, darkness_thresh=DARK, red_thresh=RED)
    assert abs(m["omega"] - omega0) / abs(omega0) < 0.01


# ---- THE LINCHPIN: the tracker recovers alpha from ground-truth pixels --------------------------------
def test_measured_angaccel_recovers_alpha_on_ground_truth_frames():
    gt, hat, nvalid = [], [], []
    for s in range(4):
        clips, _ = _scene(scene_idx=s)
        for c in clips:
            m = measured_angaccel(c.frames, darkness_thresh=DARK, red_thresh=RED)
            gt.append(float(clip_angaccel(_as_sample(c))[0]))
            hat.append(m["alpha"])
            nvalid.append(m["n_valid"])
    gt, hat = np.array(gt), np.array(hat)
    assert np.isfinite(hat).all(), "tracker must read every GT clip"
    assert min(nvalid) == 16, "every frame of a GT clip must be readable"
    rel = np.abs(hat - gt) / np.abs(gt)
    assert rel.max() < 0.02, f"max relative alpha error {rel.max():.4f}"
    assert np.corrcoef(gt, hat)[0, 1] > 0.999
    assert np.all(np.sign(hat) == np.sign(gt)), "sign convention must never invert"


def test_measured_angaccel_also_recovers_the_initial_rate():
    """Guards a parabola fit that trades curvature against slope (alpha right for the wrong reason)."""
    clips, _ = _scene()
    for c in clips:
        m = measured_angaccel(c.frames, darkness_thresh=DARK, red_thresh=RED)
        w0 = float(clip_angvel0(_as_sample(c)))
        assert abs(m["omega0"] - w0) < 0.005


def test_measured_angaccel_is_nan_without_enough_readable_frames():
    blank = torch.ones(3, 3, 64, 64)
    m = measured_angaccel(blank, darkness_thresh=DARK, red_thresh=RED)
    assert np.isnan(m["alpha"])


def test_per_frame_rotation_stays_far_below_pi_so_unwrap_is_safe():
    """The tracker unwraps phase; the ranges must keep every step well under pi or alpha is unidentifiable."""
    worst = 0.0
    for s in range(8):
        clips, _ = _scene(scene_idx=s)
        for c in clips:
            worst = max(worst, float(np.abs(np.diff(_col(c, "obj0_theta"))).max()))
    assert worst < np.pi / 4, f"max per-frame rotation {worst:.3f} rad leaves too little unwrap margin"


def test_object_stays_in_frame_at_every_orientation():
    clips, _ = _scene()
    for c in clips:
        phi = rotor_orientation(c.frames, darkness_thresh=DARK, red_thresh=RED)
        assert not np.isnan(phi).any(), "rotor must be fully visible in every frame"


# ---- the appearance-mixed variant ---------------------------------------------------------------------
def test_mixed_variant_varies_appearance_across_scenes_but_keeps_the_contract():
    a, _ = _scene(scene_idx=0, scenario="scene_angaccel2d_mixed")
    b, _ = _scene(scene_idx=1, scenario="scene_angaccel2d_mixed")
    for c in a[1:]:
        assert torch.equal(a[0].frames[0], c.frames[0]), "within-scene contract must survive mixing"
    assert a[0].meta["ball_color"] != b[0].meta["ball_color"] or a[0].meta["bg_color"] != b[0].meta["bg_color"]
    # the marker must stay colour-separable so the tracker still fires
    for c in a:
        m = measured_angaccel(c.frames, darkness_thresh=DARK, red_thresh=RED)
        assert np.isfinite(m["alpha"])
