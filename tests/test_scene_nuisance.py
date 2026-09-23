"""Regression tests for the nuisance-control scene variants (scene_size/color/background).

These are the controls for the velocity-disentanglement experiment: within a scene, velocity and
trajectory must be held FIXED across ranks while exactly one appearance factor ramps. The same-scene
difference H_b - H_a then isolates that single factor. We assert those invariants on the rendered
clips + ground-truth state (the latents are tested downstream).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.moving_ball import MovingBall

SCENARIOS = ["scene_size", "scene_color", "scene_background"]


def _scene(scenario: str, K: int = 4):
    g = MovingBall(image_size=128, num_frames=16, scenario=scenario, clips_per_scene=K,
                   radius_range=(0.05, 0.12), seed=0)
    return [g.generate(i) for i in range(K)], g


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_velocity_fixed_across_ranks(scenario):
    """The nuisance factor varies; the motion must not."""
    clips, _ = _scene(scenario)
    vels = {(round(c.meta["vel_x"], 9), round(c.meta["vel_y"], 9)) for c in clips}
    assert len(vels) == 1, f"{scenario}: velocity changed across ranks: {vels}"
    # acceleration is exactly zero (constant velocity) for every rank/frame
    for c in clips:
        ax = c.state[:, c.state_keys.index("obj0_acc_x")]
        ay = c.state[:, c.state_keys.index("obj0_acc_y")]
        assert np.allclose(ax.numpy(), 0.0) and np.allclose(ay.numpy(), 0.0)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_factor_ramps_monotonically(scenario):
    """Exactly the target factor changes across ranks, and it is an ordered ramp (rank 0 .. K-1)."""
    clips, _ = _scene(scenario)
    fv = [c.meta["factor_values"] for c in clips]
    assert all(f == fv[0] for f in fv), "every rank should carry the same scene-level ramp"
    ramp = np.asarray(fv[0])
    assert len(ramp) == len(clips)
    diffs = np.diff(ramp)
    assert np.all(diffs > 0) or np.all(diffs < 0), f"{scenario}: factor not monotone: {ramp}"

    if scenario == "scene_size":
        radii = [c.meta["radius"] for c in clips]
        assert radii == sorted(radii) and radii[0] < radii[-1]
        # colour/background held fixed
        assert len({tuple(c.meta["ball_color"]) for c in clips}) == 1
        assert len({tuple(c.meta["bg_color"]) for c in clips}) == 1
    elif scenario == "scene_color":
        assert len({tuple(c.meta["ball_color"]) for c in clips}) == len(clips)
        assert len({c.meta["radius"] for c in clips}) == 1
        assert len({tuple(c.meta["bg_color"]) for c in clips}) == 1
    elif scenario == "scene_background":
        assert len({tuple(c.meta["bg_color"]) for c in clips}) == len(clips)
        assert len({c.meta["radius"] for c in clips}) == 1
        assert len({tuple(c.meta["ball_color"]) for c in clips}) == 1


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_first_frame_position_shared(scenario):
    """Frame-0 ball CENTER is identical across ranks (only the factor differs, not the geometry)."""
    clips, _ = _scene(scenario)
    px = [round(float(c.state[0, c.state_keys.index("obj0_pos_x")]), 9) for c in clips]
    py = [round(float(c.state[0, c.state_keys.index("obj0_pos_y")]), 9) for c in clips]
    assert len(set(px)) == 1 and len(set(py)) == 1, "start position must be shared across ranks"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_ball_stays_trackable(scenario):
    """Every rank's ball must be clearly darker than its background (darkness>0.5 tracker works)."""
    clips, _ = _scene(scenario)
    for c in clips:
        lum = c.frames[0].numpy().mean(0)  # (H, W) luminance
        assert lum.min() < 0.5, f"{scenario}: ball not dark enough to track (min lum {lum.min():.2f})"
        # background is the bright majority; ball is a small dark blob
        assert (lum < 0.5).mean() < 0.2


def test_factor_is_the_only_change():
    """Cross-check: for scene_size the only metadata that differs across ranks is radius+factor."""
    clips, _ = _scene("scene_size")
    assert len({c.meta["angle"] for c in clips}) == 1
    assert len({round(c.meta["speed"], 9) for c in clips}) == 1
