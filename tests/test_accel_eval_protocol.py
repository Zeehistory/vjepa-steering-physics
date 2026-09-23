"""Protocol invariants for the acceleration steering evaluation (2026-08-26).

These pin the two fixes that made the operator comparison meaningful, both of which are the kind of
thing that silently reverts:

  * the val/test split is by SCENE, so the 7 rank-pairs of one scene -- which share an anchor clip and
    are therefore not independent samples -- can never straddle it;
  * every arm is scored on the SAME rows, so an arm that fails to track on the hard rows cannot be
    flattered by being scored only on the easy ones;
  * the arm-name pattern matches any ``<family>_s<gain>``, so a newly swept arm gets val-selected
    instead of dropping into the ungained bucket where every gain would be printed on the test half.

The last one is the dangerous one: the old pattern enumerated only ``spline_K*``/``shufT_K*``, so
gain-sweeping the oracle arms would have quietly published a best-of-sweep number read off the
reported column.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _script(name: str) -> Path:
    """Locate an experiment script by bare name, wherever it sits under experiments/."""
    hits = sorted(_EXPERIMENTS.rglob(f"{name}.py"))
    if not hits:
        raise FileNotFoundError(f"no experiment script named {name}.py under {_EXPERIMENTS}")
    return hits[0]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _script(name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


calib = _load("calibrate_spline_gain")


@pytest.mark.parametrize("name,fam,gain", [
    ("spline_K3_s2.5", "spline_K3", "2.5"),
    ("shufT_K8_s6", "shufT_K8", "6"),
    ("prof_full_s3", "prof_full", "3"),          # the oracle arms are swept now
    ("full_delta_s1.5", "full_delta", "1.5"),
    ("proj_K8_s2", "proj_K8", "2"),
    ("richappc_K3_s2.5", "richappc_K3", "2.5"),  # richer-conditioning operators
])
def test_arm_pattern_matches_every_swept_family(name, fam, gain):
    m = calib.ARM_RE.match(name)
    assert m is not None, f"{name} would fall into the ungained bucket and be reported unselected"
    assert m.group("fam") == fam
    assert m.group("gain") == gain


@pytest.mark.parametrize("name", ["noop", "full_delta", "prof_full", "proj_K4", "v_b", "a_a"])
def test_arm_pattern_leaves_ungained_names_alone(name):
    assert calib.ARM_RE.match(name) is None


def test_angle_err_is_the_angle_between_the_two_vectors():
    # tolerance is 1e-3 deg, not exact: angle_err divides by (|d||t| + 1e-12), so a perfectly aligned
    # pair reads ~8e-5 deg rather than 0. That guard is what keeps a zero-norm decode from raising.
    assert calib.angle_err([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-3)
    assert calib.angle_err([0.0, 1.0], [1.0, 0.0]) == pytest.approx(90.0, abs=1e-3)
    assert calib.angle_err([-1.0, 0.0], [1.0, 0.0]) == pytest.approx(180.0, abs=1e-3)
    assert np.isnan(calib.angle_err([float("nan"), 0.0], [1.0, 0.0]))
    assert np.isnan(calib.angle_err([0.0, 0.0], [1.0, 0.0]))       # degenerate decode, not a 0deg win


def test_all_pairs_rows_carry_their_scene_so_the_split_can_group_them():
    """A --pairs all row is keyed scene#####_r{rank} and carries `scene`; --pairs extreme keys by scene.

    Without the `scene` field the calibrator would split 700 rows by name and put pairs sharing an
    anchor clip on both sides of the val/test boundary.
    """
    rows = {f"scene{s:05d}_r{r}": {"scene": s, "rank": r} for s in range(3) for r in range(1, 8)}
    groups: dict[str, list[str]] = {}
    for k, row in rows.items():
        groups.setdefault(f"scene{row['scene']:05d}", []).append(k)
    assert len(groups) == 3
    assert all(len(v) == 7 for v in groups.values())
    names = sorted(groups)
    val, test = names[:1], names[1:]
    assert not (set(val) & set(test))
    val_rows = {k for s in val for k in groups[s]}
    test_rows = {k for s in test for k in groups[s]}
    assert not (val_rows & test_rows)
    assert len(val_rows) + len(test_rows) == 21


def test_common_finite_mask_scores_every_arm_on_the_same_rows():
    """An arm that fails to track on a row must remove that row for ALL arms, not just itself."""
    per = {
        "r0": {"v_b": [1.0, 0.0], "good": [1.0, 0.0], "flaky": [1.0, 0.0]},
        "r1": {"v_b": [1.0, 0.0], "good": [0.0, 1.0], "flaky": [float("nan"), 0.0]},
    }
    methods = ["good", "flaky"]
    usable = [k for k in per
              if all(m in per[k] and np.isfinite(calib.angle_err(per[k][m], per[k]["v_b"]))
                     for m in methods)]
    assert usable == ["r0"]
    # scored on the common row both arms read 0 deg; the naive per-arm drop would have given
    # good = 45 deg (both rows) and flaky = 0 deg (its one tracked row), i.e. a spurious flaky win
    for m in methods:
        errs = [calib.angle_err(per[k][m], per[k]["v_b"]) for k in usable]
        assert float(np.mean(errs)) == pytest.approx(0.0, abs=1e-3)


def test_trajectory_target_matches_the_discrete_generator_rollout():
    """decopt's anchor path must be da*f(f-1)/2, the generator's discrete integral -- not 0.5*da*f^2.

    ``_accel_rel_positions`` (src/data/moving_ball.py) integrates vel_t = v0 + acc*t,
    pos_{t+1} = pos_t + vel_t. The continuous form overshoots by da*f/2 at every frame.
    """
    F = 16
    da = np.array([0.002, -0.001])
    pos, v0 = np.zeros(2), np.array([0.01, 0.005])
    rel = [pos.copy()]
    for t in range(F - 1):
        pos = pos + (v0 + da * t)
        rel.append(pos.copy())
    rollout = np.stack(rel) - np.outer(np.arange(F), v0)      # strip the shared v0 term

    ff = np.arange(F, dtype=np.float64)
    assert np.allclose(np.outer(ff * (ff - 1.0) / 2.0, da), rollout, atol=1e-12)
    continuous = 0.5 * np.outer(ff * ff, da)
    assert not np.allclose(continuous, rollout, atol=1e-6)
    assert np.allclose(continuous - rollout, np.outer(ff / 2.0, da), atol=1e-12)
