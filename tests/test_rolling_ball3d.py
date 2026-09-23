"""Tests for the 3D MuJoCo rolling-ball dataset (``scene_rollingball3d``).

These pin the three properties the steering experiment actually depends on, each of which was a real
failure mode found while tuning the scene (see ``src/data/rolling_ball3d.py``):

1. **The scene contract** matches ``scene_velocity2d`` — one shared start per scene, K distinct velocity
   vectors, deterministic, every roll stays on the table and in frame.
2. **The physics is constant-velocity** — the rolling-without-slipping launch means world speed does not
   decay, so "the clip's velocity" is well defined exactly as in 2D.
3. **The labels match what the pixel tracker measures** — the state columns are image-plane quantities
   agreeing with the darkness-centroid tracker to sub-pixel accuracy. If this regresses (e.g. the shadow
   darkens past the tracker threshold) every steering number silently gains a bias, so it is pinned.
"""

from __future__ import annotations

import os

# MUST precede the mujoco import. MuJoCo resolves its GL backend from MUJOCO_GL at IMPORT time, and
# importorskip below imports it before ``src.data.rolling_ball3d`` gets the chance to set that
# variable -- so the backend falls back to GLFW/X11, which ABORTS the process (not a catchable
# exception) on a cluster node with no DISPLAY.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("mujoco", reason="3D dataset needs mujoco")

from src.analysis import velocity_ops as vo  # noqa: E402
from src.analysis.ball_tracking import ball_centroids, measured_velocity  # noqa: E402
from src.data.rolling_ball3d import (  # noqa: E402
    BALL_R,
    TABLE_H,
    TABLE_HALF_X,
    TABLE_HALF_Y,
    RollingBall3D,
)

K = 8


@pytest.fixture(scope="module")
def gen() -> RollingBall3D:
    return RollingBall3D(image_size=256, num_frames=16, fps=4, clips_per_scene=K, seed=0)


@pytest.fixture(scope="module")
def scene0(gen: RollingBall3D) -> list:
    """The K clips of scene 0 — generated once, they are the expensive part of this module."""
    return [gen.generate(r) for r in range(K)]


# -- 1. scene contract ---------------------------------------------------------------------------
def test_deterministic(gen: RollingBall3D) -> None:
    other = RollingBall3D(image_size=256, num_frames=16, fps=4, clips_per_scene=K, seed=0)
    a, b = gen.generate(11), other.generate(11)
    assert torch.equal(a.frames, b.frames)
    assert torch.equal(a.state, b.state)


def test_shape_and_range(scene0: list) -> None:
    c = scene0[0]
    assert c.frames.shape == (16, 3, 256, 256)
    assert float(c.frames.min()) >= 0.0 and float(c.frames.max()) <= 1.0
    assert c.state.shape[0] == 16


def test_scene_shares_one_start_with_distinct_velocities(scene0: list) -> None:
    starts = np.stack([np.array(c.meta["world_pos0"]) for c in scene0])
    assert np.allclose(starts, starts[0]), "all ranks of a scene must share ONE start position"

    # distinct world velocity VECTORs: both heading and speed vary across ranks
    wv = np.stack([[c.meta["world_vel_x"], c.meta["world_vel_y"]] for c in scene0])
    headings = np.degrees(np.arctan2(wv[:, 1], wv[:, 0])) % 360
    gaps = np.diff(np.sort(headings))
    assert len(np.unique(np.round(headings, 3))) == K
    assert gaps.min() > 5.0, f"headings must stay distinct, got gaps {gaps}"
    speeds = np.linalg.norm(wv, axis=1)
    assert speeds.max() / speeds.min() > 1.3, "speeds must span a real range, not just direction"


def test_ball_stays_on_table_and_in_frame(gen: RollingBall3D) -> None:
    for i in range(3 * K):
        c = gen.generate(i)
        keys = c.state_keys
        st = c.state.numpy()
        px = st[:, keys.index("obj0_pos_x")]
        py = st[:, keys.index("obj0_pos_y")]
        assert px.min() > 0.02 and px.max() < 0.98, f"clip {i} left the frame in x"
        assert py.min() > 0.02 and py.max() < 0.98, f"clip {i} left the frame in y"
        # and on the tabletop, in world coords (a ball that rolls off would free-fall)
        p0 = np.array(c.meta["world_pos0"])
        v = np.array([c.meta["world_vel_x"], c.meta["world_vel_y"]])
        end = p0 + v * (1.0 / gen.fps) * (gen.num_frames - 1)
        assert abs(end[0]) < TABLE_HALF_X - BALL_R and abs(end[1]) < TABLE_HALF_Y - BALL_R


def test_pipeline_pairs_scenes_like_the_2d_dataset(scene0: list) -> None:
    """velocity_ops reads velocity/positions straight off the state columns — check its view is sane."""
    sa = {"state": scene0[0].state.numpy(), "state_keys": scene0[0].state_keys}
    sb = {"state": scene0[-1].state.numpy(), "state_keys": scene0[-1].state_keys}
    va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb)
    assert va.shape == (2,) and np.isfinite(va).all()
    assert not np.allclose(va, vb), "distinct ranks must give distinct commands"
    # shared start => the pairing that cancels appearance
    assert np.allclose(vo.clip_start_pos(sa), vo.clip_start_pos(sb), atol=1e-6)
    assert vo.command_features(va, vb).shape == (13,)


# -- 2. physics ------------------------------------------------------------------------------------
def test_rolling_is_constant_velocity(gen: RollingBall3D) -> None:
    """Rolling without slipping + zero rolling friction => world speed must not decay over the clip.

    Checked in WORLD space (image speed legitimately varies under perspective). The whole
    ``scene_velocity2d`` contract assumes one velocity per clip; a ball that decelerated would make the
    label a lie.
    """
    c = gen.generate(0)
    p0 = np.array(c.meta["world_pos0"])
    v = np.array([c.meta["world_vel_x"], c.meta["world_vel_y"]])
    # re-simulate and read the true world track back out of mujoco
    mujoco, model, data, _r, _cam = gen._lazy_sim()
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [p0[0], p0[1], TABLE_H + BALL_R]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:3] = [v[0], v[1], 0.0]
    data.qvel[3:6] = [-v[1] / BALL_R, v[0] / BALL_R, 0.0]
    mujoco.mj_forward(model, data)
    steps = int(round((1.0 / gen.fps) / model.opt.timestep))
    speeds = []
    for _ in range(gen.num_frames):
        speeds.append(float(np.linalg.norm(data.qvel[:2])))
        for _ in range(steps):
            mujoco.mj_step(model, data)
    speeds = np.array(speeds)
    drift = (speeds.max() - speeds.min()) / speeds.mean()
    assert drift < 0.01, f"speed drifted {drift:.4f}: {speeds.min()}..{speeds.max()}"
    # and it really is ROLLING (spin locked to translation), not sliding
    assert abs(float(np.linalg.norm(data.qvel[3:5])) - speeds[-1] / BALL_R) / (speeds[-1] / BALL_R) < 0.01


def test_perspective_is_real_but_not_degenerate(gen: RollingBall3D) -> None:
    """The camera must actually foreshorten (else it is not a 3D test) but not so much that the
    velocity command collapses onto a squashed ellipse. Both bounds were tuned; pin them."""
    speeds = [np.linalg.norm(gen._image_velocity(np.zeros(2), 0.08 * np.array([np.cos(a), np.sin(a)])))
              for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    aniso = max(speeds) / min(speeds)
    assert 1.05 < aniso < 1.6, f"heading anisotropy {aniso:.3f} outside the tuned band"

    # image speed also depends on WHERE the ball is (pure perspective; impossible in the 2D dataset)
    at = [np.linalg.norm(gen._image_velocity(np.array([px, py]), np.array([0.08, 0.0])))
          for px in (-0.3, 0.0, 0.3) for py in (-0.3, 0.0, 0.3)]
    assert max(at) / min(at) > 1.05, "no position dependence => camera is effectively orthographic"


def test_image_speeds_match_the_2d_reference_band(gen: RollingBall3D) -> None:
    """The world speed_range is tuned so image speeds land in the 2D dataset's [0.012, 0.024] band —
    that is what makes this an apples-to-apples transfer test rather than a different task."""
    sp = []
    for s in range(12):
        srng = np.random.default_rng(gen.seed * 100_003 + 7919 * (s + 1))
        pos0, vels = gen._velocity3d_set(srng)
        sp += [float(np.linalg.norm(gen._image_velocity(pos0, v))) for v in vels]
    sp = np.array(sp)
    assert 0.008 < np.percentile(sp, 2) and np.percentile(sp, 98) < 0.028


def test_shared_start_shrinks_speeds_only_when_infeasible(gen: RollingBall3D) -> None:
    srng = np.random.default_rng(0)
    slow = np.stack([0.02 * np.array([np.cos(a), np.sin(a)])
                     for a in np.linspace(0, 2 * np.pi, K, endpoint=False)])
    _pos, scale = gen._shared_start(srng, slow)
    assert scale == 1.0, "feasible scene must not be shrunk"
    fast = slow * 40.0  # cannot possibly fit on the table from any single start
    _pos, scale = gen._shared_start(srng, fast)
    assert scale < 1.0


# -- 3. labels match the pixel tracker -------------------------------------------------------------
def test_state_positions_match_the_tracker(scene0: list) -> None:
    """State ``obj0_pos_*`` is the analytic camera projection; the pipeline measures a darkness centroid.
    They must agree, i.e. the contact shadow must stay lighter than the tracker's threshold."""
    for c in scene0[:3]:
        keys = c.state_keys
        st = c.state.numpy()
        gt = np.stack([st[:, keys.index("obj0_pos_x")], st[:, keys.index("obj0_pos_y")]], axis=1)
        tracked = ball_centroids(c.frames)
        assert np.isfinite(tracked).all(), "tracker lost the ball"
        err_px = np.linalg.norm(gt - tracked, axis=1).mean() * 256
        assert err_px < 2.0, f"tracker/label disagree by {err_px:.2f}px"


def test_state_velocity_is_what_the_tracker_measures(scene0: list) -> None:
    """``clip_velocity`` (mean of the state column) must equal ``measured_velocity`` (mean tracked
    displacement) — this equality is what makes the steering angle error meaningful."""
    for c in scene0[:3]:
        keys = c.state_keys
        st = c.state.numpy()
        label = np.array([st[:, keys.index("obj0_vel_x")].mean(), st[:, keys.index("obj0_vel_y")].mean()])
        m = measured_velocity(c.frames)
        meas = np.array([m["vel_x"], m["vel_y"]])
        cos = label @ meas / (np.linalg.norm(label) * np.linalg.norm(meas))
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 2.0
        assert abs(np.linalg.norm(meas) / np.linalg.norm(label) - 1.0) < 0.10


def test_ball_is_the_only_dark_object(scene0: list) -> None:
    """The darkness tracker assumes the ball is the sole sub-threshold blob. Guards against a future
    lighting/texture change (a dark tabletop square, a hard shadow) silently biasing every centroid."""
    fr = scene0[0].frames
    gray = fr.mean(dim=1)
    dark_frac = float((gray < 0.5).float().mean())
    # the ball is ~16px radius out of 256^2 -> ~1.2% of pixels; anything much larger means the shadow
    # or the table is being picked up too
    assert 0.003 < dark_frac < 0.030, f"dark pixel fraction {dark_frac:.4f} — something else is dark"
