"""Tests for the paddle-strike dataset (``scene_paddle_strike``).

These pin the properties the latent-control experiment actually depends on. Each was a real failure
mode found while building the scene (see ``src/data/paddle_strike.py``):

1. **No rendered frame straddles the collision.** Aiming contact at the post-window's first frame
   left that frame carrying the *pre*-contact velocity, and a compliant contact then rang on for two
   more frames -- both smeared the post-window velocity label into a blend.
2. **The pre-window is bit-identical across actions.** This is load-bearing, not cosmetic: if the
   context clip revealed which action was taken, "select among candidate actions" would be vacuous.
   An earlier design solved the paddle's REST POSITION per action so contact landed on a fixed
   frame; that leaked the action as ~17 px of paddle offset in a supposedly action-agnostic context.
   The fix -- fixed rest position, event-aligned post window -- is what these tests pin.
3. **The action -> outcome map is monotone, linear and invertible**, which is what makes a commanded
   speed reachable at all.
4. **The ball is the only dark object in frame.** The whole pipeline tracks it with a darkness
   centroid, so a paddle face or shadow crossing the 0.5 threshold silently biases every measured
   velocity -- which is exactly what happened at the first camera/lighting setting (30 px of error).
"""

from __future__ import annotations

import os

# MUST precede the mujoco import: mujoco resolves its GL backend from MUJOCO_GL at IMPORT time, and
# importorskip below would otherwise lock in the default (GLFW/X11), which has no DISPLAY on a
# cluster node. The data module sets this too, but it is imported after importorskip.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("mujoco", reason="paddle-strike dataset needs mujoco")

from src.analysis.ball_tracking import ball_centroids, measured_velocity  # noqa: E402
from src.control.strike_inverse import StrikeInverse, sweep_strikes  # noqa: E402
from src.data.paddle_strike import (  # noqa: E402
    BALL_MASS,
    BALL_R,
    BALL_VALUE,
    BALL_VALUE_RANGE,
    NOMINAL_CONTACT_FRAME,
    PADDLE_HALF,
    PADDLE_MASS,
    X_PADDLE_REST,
    X_POST,
    PaddleStrike,
    ball_start_x,
)

K = 8


@pytest.fixture(scope="module")
def gen() -> PaddleStrike:
    return PaddleStrike(image_size=256, num_frames=16, clips_per_scene=K, seed=0)


_RENDER_OK: bool | None = None


def _can_render(gen: PaddleStrike) -> bool:
    """Probe once and cache. A second attempt after a failed EGL init aborts the process, so this
    must never retry -- which is exactly what happened before the MUJOCO_GL fix above."""
    global _RENDER_OK
    if _RENDER_OK is None:
        try:
            gen._lazy_sim(render=True)
            _RENDER_OK = True
        except Exception:
            _RENDER_OK = False
    return _RENDER_OK


# -- 1. contact timing -----------------------------------------------------------------------------

def test_collision_falls_inside_the_unrendered_swing_gap(gen: PaddleStrike) -> None:
    """No RENDERED frame may straddle the collision, or its velocity label is a blend.

    Contact time is deliberately free (the paddle's rest position is fixed so the pre-window cannot
    leak the action), so the invariant is about clearance, not about an exact frame index.
    """
    for v_in in (0.85, 1.0, 1.15):
        for v_p in (0.2, 0.0, -0.4, -0.9, -1.4):
            r = gen.simulate(v_in, v_p, render=False)
            assert r["margin_after_pre"] > 0, (v_in, v_p, r["margin_after_pre"])
            assert r["margin_before_post"] > 0, (v_in, v_p, r["margin_before_post"])
            assert r["n_touches"] == 1, (v_in, v_p, r["n_touches"])


def test_post_window_opens_at_a_shared_ball_position(gen: PaddleStrike) -> None:
    """Every action's post clip must open with the ball in the same place, so that within a scene
    the clips differ only in VELOCITY -- the contract H_b - H_a relies on."""
    xs = [gen.simulate(1.0, v_p, render=False)["post_start_x"]
          for v_p in (0.2, 0.0, -0.4, -0.9, -1.4)]
    assert max(xs) - min(xs) < 0.005, xs
    assert all(abs(x - X_POST) < 0.005 for x in xs), xs


def test_collision_finishes_before_the_post_window(gen: PaddleStrike) -> None:
    """Post-window velocity must be constant -- i.e. the collision is fully over by POST_START.

    A softer contact (the first ``solref`` tried) rang through frames 24-25 and left this std at
    ~0.7 m/s, which would have smeared every post-clip velocity label.
    """
    for v_p in (-0.3, -0.9, -1.5):
        r = gen.simulate(1.0, v_p, render=False)
        assert r["v_out_world_std"] < 1e-3, (v_p, r["v_out_world_std"])


def test_pre_window_velocity_is_the_commanded_inflow(gen: PaddleStrike) -> None:
    for v_in in (0.85, 1.0, 1.15):
        r = gen.simulate(v_in, -1.0, render=False)
        assert abs(r["v_in_world"] - v_in) < 1e-3


def test_ball_start_places_nominal_contact(gen: PaddleStrike) -> None:
    """With a STATIONARY paddle the ball must arrive at NOMINAL_CONTACT_FRAME -- that is what the
    ball's start position is solved for, and it depends on v_in alone, never on the action."""
    for v_in in (0.85, 1.0, 1.15):
        t_c = NOMINAL_CONTACT_FRAME / gen.fps
        surface = ball_start_x(v_in) + v_in * t_c + BALL_R
        face = X_PADDLE_REST - PADDLE_HALF[0]
        assert abs(face - surface) < 1e-12
        # and the simulator agrees, to within the contact's own compliance
        r = gen.simulate(v_in, 0.0, render=False)
        assert abs(r["contact_frame"] - NOMINAL_CONTACT_FRAME) < 0.5, r["contact_frame"]


# -- 2. the load-bearing invariant -----------------------------------------------------------------

def test_pre_window_is_identical_across_actions(gen: PaddleStrike) -> None:
    """Every action in a scene must share a bit-identical pre-contact window.

    If it did not, the context latent would leak the action and choosing among candidate actions
    would be circular. The paddle is at rest for the whole pre-window precisely to guarantee this.
    """
    sp = gen.scene_params(0)
    ref = None
    for ratio in sp["ratios"]:
        v_p = gen.action_for(sp["v_in"], -ratio * sp["v_in"])
        r = gen.simulate(sp["v_in"], v_p, render=False)
        pre = r["ball_x"][:16].copy(), r["pad_x"][:16].copy()
        if ref is None:
            ref = pre
        else:
            assert np.array_equal(ref[0], pre[0]), "ball track differs in the pre-window"
            assert np.array_equal(ref[1], pre[1]), "paddle track differs in the pre-window"


def test_determinism(gen: PaddleStrike) -> None:
    a = gen.simulate(1.0, -1.0, render=False)
    b = gen.simulate(1.0, -1.0, render=False)
    assert np.array_equal(a["ball_x"], b["ball_x"])
    assert a["v_out_world"] == b["v_out_world"]


# -- 3. controllability ----------------------------------------------------------------------------

def test_map_is_monotone_and_linear(gen: PaddleStrike) -> None:
    sweep = sweep_strikes(gen, [0.9, 1.0, 1.1], np.linspace(0.3, -1.5, 12))
    for v in (0.9, 1.0, 1.1):
        m = np.isclose(sweep["v_in"], v, atol=1e-3)
        d = np.diff(sweep["v_out"][m])
        assert np.all(d < 0), "v_out must decrease strictly as the paddle drives harder leftward"
    inv = StrikeInverse.fit(sweep, BALL_MASS, PADDLE_MASS)
    assert 100.0 * inv.forward_max_resid / inv.forward_range < 1.0


def test_commanded_speed_is_achieved_within_tolerance(gen: PaddleStrike) -> None:
    """The headline bar, in miniature: fit on one set of v_in, command on a disjoint set."""
    sweep = sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15], np.linspace(0.35, -1.75, 16))
    inv = StrikeInverse.fit(sweep, BALL_MASS, PADDLE_MASS)
    rng = np.random.default_rng(0)
    v_in = rng.choice([0.9, 1.0, 1.1], size=24)
    target = -rng.uniform(0.5, 3.0, size=24) * v_in
    v_p = np.asarray(inv.action_for(v_in, target, mode="quadratic")).ravel()
    rel = []
    for vi, vp, tg in zip(v_in, v_p, target):
        r = gen.simulate(float(vi), float(vp), render=False)
        rel.append(abs(r["v_out_world"] - tg) / abs(tg))
    assert (np.asarray(rel) <= 0.05).mean() >= 0.95, f"max rel err {max(rel):.4f}"


def test_effective_restitution_is_stable_across_inflow(gen: PaddleStrike) -> None:
    es = []
    for v in (0.85, 1.0, 1.15):
        s = sweep_strikes(gen, [v], np.linspace(0.3, -1.5, 8))
        c = np.polyfit(s["v_p"], s["v_out"], 1)
        es.append(c[0] * (BALL_MASS + PADDLE_MASS) / PADDLE_MASS - 1.0)
    es = np.asarray(es)
    assert (es.max() - es.min()) / abs(es.mean()) < 0.02


# -- 4. pixels -------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ball_gray,table_shade",
    [
        (BALL_VALUE, 1.0),                     # the default appearance
        # The CORNERS of the per-scene nuisance ranges scene_params actually samples
        # (ball_gray ~ U(0.22, 0.33), table_shade ~ U(0.88, 1.06)). Testing only the default left the
        # dataset's own distribution unchecked, and it is not a safe omission: table_shade scales the
        # tabletop albedo, so at 0.88 the darker wood square sits at mean 0.655 * 0.88 = 0.576, BELOW
        # the 0.610 albedo floor that is supposed to keep every non-ball material out of the darkness
        # mask. Brightest ball against darkest table is the worst case for the ordering the tracker
        # depends on, and it was the one corner nothing covered.
        (BALL_VALUE_RANGE[1], 0.88),
        (BALL_VALUE_RANGE[0], 0.88),
        (BALL_VALUE_RANGE[1], 1.06),
    ],
)
def test_ball_is_the_only_dark_object(gen: PaddleStrike, ball_gray: float,
                                     table_shade: float) -> None:
    """The darkness-centroid tracker assumes the ball is the sole sub-0.5 object in frame.

    At the first camera/lighting setting the paddle's shaded face fell below the threshold and
    dragged the centroid 30 px off the ball.
    """
    if not _can_render(gen):
        pytest.skip("no working GL context for offscreen rendering")
    r = gen.simulate(1.0, -1.1, render=True, ball_gray=ball_gray, table_shade=table_shade)
    rad_px = r["ball_img_radius"] * gen.image_size
    for t in (0, 8, gen.num_frames, 2 * gen.num_frames - 1):
        gray = (r["frames"][t].astype(np.float32) / 255.0).mean(axis=2)
        ys, xs = np.nonzero(gray < 0.5)
        assert len(xs) > 0, f"nothing dark in frame {t}"
        # the dark region must be no bigger than the ball and centred on its projected position
        assert xs.max() - xs.min() <= 2.6 * rad_px, f"dark blob too wide in frame {t}"
        assert ys.max() - ys.min() <= 2.6 * rad_px, f"dark blob too tall in frame {t}"
        cx = xs.mean() / gen.image_size
        assert abs(cx - r["img_pos"][t, 0]) * gen.image_size < 2.0


def test_tracker_agrees_with_analytic_projection(gen: PaddleStrike) -> None:
    """Labels are image-plane, so what the pixel tracker reads must be what we wrote down."""
    if not _can_render(gen):
        pytest.skip("no working GL context for offscreen rendering")
    for v_p in (-1.1, -0.5, 0.2):
        r = gen.simulate(1.0, v_p, render=True)
        fr = (r["frames"].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        for sl in (r["pre_slice"], r["post_slice"]):
            tracked = measured_velocity(torch.from_numpy(fr[sl]))["vel_x"]
            analytic = float(np.diff(r["img_pos"][sl], axis=0)[:, 0].mean())
            assert abs(tracked - analytic) / abs(analytic) < 0.01, (v_p, sl)
            cen = ball_centroids(torch.from_numpy(fr[sl]))
            assert np.abs(cen - r["img_pos"][sl]).max() * gen.image_size < 1.5


# -- 5. the BallClip contract ----------------------------------------------------------------------

def test_clip_contract(gen: PaddleStrike) -> None:
    if not _can_render(gen):
        pytest.skip("no working GL context for offscreen rendering")
    clip = gen.generate(0)
    assert clip.frames.shape == (16, 3, 256, 256)
    assert clip.state.shape == (16, len(clip.state_keys))
    assert float(clip.frames.min()) >= 0.0 and float(clip.frames.max()) <= 1.0
    for key in ("v_in_world", "v_out_world", "v_p_cmd", "contact_frame", "scene", "rank",
                "episode", "embodiment", "ratio"):
        assert key in clip.meta, key
    assert clip.meta["margin_after_pre"] > 0 and clip.meta["margin_before_post"] > 0
    # the constant-velocity contract: vel_x column is constant and equals the mean displacement
    vx = clip.state[:, clip.state_keys.index("obj0_vel_x")]
    assert torch.allclose(vx, vx[0])


def test_scene_rank_structure(gen: PaddleStrike) -> None:
    """Ranks within a scene share v_in and differ only in outcome -- what H_b - H_a pairs on."""
    sp0 = gen.scene_params(0)
    assert len(sp0["ratios"]) == K
    assert np.all(np.diff(sp0["ratios"]) > 0), "ranks must be ordered by outcome speed"
    assert gen.scene_params(0) == gen.scene_params(0)
    assert gen.scene_params(1)["v_in"] != sp0["v_in"]


def test_pre_scenario_is_one_clip_per_scene(gen: PaddleStrike) -> None:
    if not _can_render(gen):
        pytest.skip("no working GL context for offscreen rendering")
    pre = PaddleStrike(image_size=256, num_frames=16, clips_per_scene=K, seed=0,
                       scenario="scene_paddle_strike_pre")
    c0, c1 = pre.generate(0), pre.generate(1)
    assert c0.meta["scene"] == 0 and c1.meta["scene"] == 1
    assert c0.meta["window"] == "pre"
    # a pre clip must carry the inflow, and its ball must be moving rightward (toward the paddle)
    assert c0.meta["v_in_world"] > 0
    assert c0.state[:, c0.state_keys.index("obj0_vel_x")][0] > 0


# -- 6. the inverse model --------------------------------------------------------------------------

def test_inverse_round_trip(gen: PaddleStrike, tmp_path) -> None:
    sweep = sweep_strikes(gen, [0.9, 1.1], np.linspace(0.3, -1.5, 10))
    inv = StrikeInverse.fit(sweep, BALL_MASS, PADDLE_MASS)
    p = tmp_path / "inv.json"
    inv.save(p)
    back = StrikeInverse.load(p)
    assert back.alpha == inv.alpha and back.beta == inv.beta
    v_p = inv.action_for(1.0, -2.0, mode="linear")
    assert abs(float(inv.forward(1.0, v_p)) + 2.0) < 1e-9
