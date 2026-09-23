"""Tests for the Franka render variant (``src/data/franka_strike.py``).

The variant exists to hold physics EXACTLY fixed while changing appearance, so that a probe or
operator fitted on one rendering and applied to the other isolates visual transfer. These tests pin
the two properties that claim rests on, each of which failed at some point while building it:

1. **The two renderings are the same physics.** Not "similar" -- the measured outcome must agree to
   solver precision. It did not, three separate times: the Panda's ``<option>`` overriding the
   integrator and ``impratio``; and MuJoCo's default solver tolerance being too loose once the arm
   added 7 DOFs to the constraint solve, which roughened the strike map 32x.
2. **The ball is still the only dark object in frame.** The stock Panda's dark trim sits below the
   darkness-centroid tracker's 0.5 threshold and pulls the measured centroid ~38 px off the ball.
"""

from __future__ import annotations

import os

# MUST precede the mujoco import -- see the note in tests/test_paddle_strike.py.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("mujoco", reason="paddle-strike dataset needs mujoco")

from src.analysis.ball_tracking import ball_centroids, measured_velocity  # noqa: E402
from src.data.paddle_strike import PaddleStrike, build_striker  # noqa: E402

franka_strike = pytest.importorskip(
    "src.data.franka_strike", reason="needs mujoco_menagerie (experiments/threads/paddle-robotics/01_data/fetch_menagerie.sh)")

try:
    franka_strike._menagerie_dir()
except FileNotFoundError:
    pytest.skip("mujoco_menagerie Panda not vendored", allow_module_level=True)

ACTIONS = (0.25, 0.0, -0.3, -0.5, -0.8, -1.105, -1.45)


@pytest.fixture(scope="module")
def paddle() -> PaddleStrike:
    return PaddleStrike(image_size=256, num_frames=16, seed=0)


@pytest.fixture(scope="module")
def franka():
    return franka_strike.FrankaStrike(image_size=256, num_frames=16, seed=0)


_RENDER_OK: bool | None = None


def _can_render(gen) -> bool:
    global _RENDER_OK
    if _RENDER_OK is None:
        try:
            gen._lazy_sim(render=True)
            _RENDER_OK = True
        except Exception:
            _RENDER_OK = False
    return _RENDER_OK


# -- 1. the two renderings are the same physics ----------------------------------------------------

def test_outcomes_match_the_abstract_paddle(paddle, franka) -> None:
    """The whole point of the variant: appearance changes, physics does not."""
    for v_p in ACTIONS:
        a = paddle.simulate(1.0, v_p, render=False)
        b = franka.simulate(1.0, v_p, render=False)
        assert abs(a["v_out_world"] - b["v_out_world"]) < 1e-9, (v_p, a["v_out_world"],
                                                                 b["v_out_world"])
        assert abs(a["contact_frame"] - b["contact_frame"]) < 1e-9
        assert abs(a["post_start_x"] - b["post_start_x"]) < 1e-9


def test_solver_settings_match_the_abstract_scene(paddle, franka) -> None:
    """Pinned because the Panda's own <option> silently overrode both, and the resulting difference
    in how the contact was solved is invisible until you diff the outcomes."""
    a = paddle._lazy_sim(render=False)["model"].opt
    b = franka._lazy_sim(render=False)["model"].opt
    assert a.timestep == b.timestep
    assert a.tolerance == b.tolerance
    assert a.impratio == b.impratio
    assert a.integrator == b.integrator


def test_arm_never_touches_anything(franka) -> None:
    """The arm is a visual carrier; if any arm geom could collide, it would change the physics."""
    sim = franka._lazy_sim(render=False)
    model, ids = sim["model"], sim["ids"]
    arm = np.isin(model.geom_bodyid, ids["arm_bodies"])
    assert arm.sum() > 0, "no arm geoms found -- the body-name list is stale"
    assert not model.geom_contype[arm].any()
    assert not model.geom_conaffinity[arm].any()


def test_pre_window_still_identical_across_actions(franka) -> None:
    """The load-bearing invariant must survive the arm being added to the scene."""
    sp = franka.scene_params(0)
    ref = None
    for ratio in sp["ratios"]:
        v_p = franka.action_for(sp["v_in"], -ratio * sp["v_in"])
        r = franka.simulate(sp["v_in"], v_p, render=False)
        pre = r["ball_x"][:16].copy(), r["pad_x"][:16].copy()
        if ref is None:
            ref = pre
        else:
            assert np.array_equal(ref[0], pre[0])
            assert np.array_equal(ref[1], pre[1])


# -- 2. the arm must not break the pixel tracker ---------------------------------------------------

def test_ball_is_still_the_only_dark_object(franka) -> None:
    if not _can_render(franka):
        pytest.skip("no working GL context for offscreen rendering")
    r = franka.simulate(1.0, -1.105, render=True)
    rad_px = r["ball_img_radius"] * franka.image_size
    T = franka.num_frames
    for t in (0, T // 2, T, 2 * T - 1):
        gray = (r["frames"][t].astype(np.float32) / 255.0).mean(axis=2)
        ys, xs = np.nonzero(gray < 0.5)
        assert len(xs) > 0, f"nothing dark in frame {t}"
        assert xs.max() - xs.min() <= 2.6 * rad_px, f"dark blob too wide in frame {t} (arm trim?)"
        assert ys.max() - ys.min() <= 2.6 * rad_px, f"dark blob too tall in frame {t} (arm trim?)"
        assert abs(xs.mean() / franka.image_size - r["img_pos"][t, 0]) * franka.image_size < 2.0


def test_tracker_agrees_with_analytic_projection(franka) -> None:
    if not _can_render(franka):
        pytest.skip("no working GL context for offscreen rendering")
    r = franka.simulate(1.0, -1.105, render=True)
    fr = (r["frames"].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    for sl in (r["pre_slice"], r["post_slice"]):
        tracked = measured_velocity(torch.from_numpy(fr[sl]))["vel_x"]
        analytic = float(np.diff(r["img_pos"][sl], axis=0)[:, 0].mean())
        assert abs(tracked - analytic) / abs(analytic) < 0.02
        cen = ball_centroids(torch.from_numpy(fr[sl]))
        assert np.abs(cen - r["img_pos"][sl]).max() * franka.image_size < 2.0


def test_arm_is_actually_visible(paddle, franka) -> None:
    """A 'render variant' that renders the same pixels would be a silent no-op."""
    if not _can_render(franka):
        pytest.skip("no working GL context for offscreen rendering")
    a = paddle.simulate(1.0, -1.105, render=True)["frames"][0].astype(np.float32)
    b = franka.simulate(1.0, -1.105, render=True)["frames"][0].astype(np.float32)
    changed = (np.abs(a - b).mean(axis=2) > 8).mean()
    assert changed > 0.02, f"only {changed:.1%} of pixels differ -- is the arm in frame?"


# -- 3. the factory --------------------------------------------------------------------------------

def test_build_striker_dispatch() -> None:
    assert isinstance(build_striker("paddle"), PaddleStrike)
    assert isinstance(build_striker("franka"), franka_strike.FrankaStrike)
    assert build_striker("franka").embodiment == "franka"
    with pytest.raises(ValueError):
        build_striker("nonesuch")
