"""Tests for the inverse action predictor (``src/control/action_predictor.py``).

Deliberately a separate file from ``test_paddle_strike.py``: the predictor is pure numpy/torch with
no mujoco dependency, so these run in seconds on any node instead of behind an ``importorskip`` and
an EGL context. Each test pins a failure mode that either did occur or would be invisible if it did.

1. **Scene-level splitting.** A clip-level split leaks: ranks within a scene are the same episode at
   different commands, so val error would measure interpolation across a memorised scene.
2. **Linear recovery.** The true inverse is linear to ~0.3% of range, so a predictor that cannot
   recover an exactly-linear map is broken regardless of what it scores on real data.
3. **Save/load is exact.** A round-trip that silently perturbs weights would show up only as a
   mysteriously worse closed-loop number in a later run.
4. **The target actually matters.** If predictions barely move when the commanded target changes,
   the policy is reading the action off the context -- which is the vacuous result the shuffled
   control exists to catch, and it is worth catching in unit tests too.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.control.action_predictor import (
    ActionPredictor,
    pair_features,
    scene_holdout,
)

K = 32
N_SCENES = 120
N_RANK = 8
ALPHA, BETA = 1.888, 0.06


def synthetic(seed: int = 0, noise: float = 0.05):
    """A toy world with the real one's structure: a scene fixes v_in, each rank picks a command.

    The action follows the certified linear law ``v_out = alpha*v_p + beta*v_in``, so the exact
    inverse exists and any competent predictor must find it.
    """
    rng = np.random.default_rng(seed)
    v_in = rng.uniform(0.85, 1.15, N_SCENES)
    ratios = rng.uniform(0.75, 2.9, (N_SCENES, N_RANK))
    b_pre, b_out = rng.normal(size=K), rng.normal(size=K)
    H, Z, A, S = [], [], [], []
    for s in range(N_SCENES):
        h = b_pre * v_in[s] + noise * rng.normal(size=K)
        for r in range(N_RANK):
            v_star = -ratios[s, r] * v_in[s]
            H.append(h)
            Z.append(b_pre * v_in[s] + b_out * v_star + noise * rng.normal(size=K))
            A.append((v_star - BETA * v_in[s]) / ALPHA)
            S.append(s)
    return (np.asarray(H), np.asarray(Z), np.asarray(A), np.asarray(S))


@pytest.fixture(scope="module")
def data():
    return synthetic()


def test_scene_holdout_is_disjoint_and_covers_everything() -> None:
    _, _, _, S = synthetic()
    fit, val = scene_holdout(S, 0.2, seed=0)
    assert not (set(S[fit]) & set(S[val])), "a scene appears on both sides -- the split leaks"
    assert (fit | val).all() and not (fit & val).any(), "masks must partition the rows"
    # every row of a held-out scene must be held out, not just some of its ranks
    for s in set(S[val]):
        assert val[S == s].all()
    assert 0.1 < val.mean() < 0.35


def test_pair_features_shapes_and_difference_term() -> None:
    rng = np.random.default_rng(0)
    h, z = rng.normal(size=(5, K)), rng.normal(size=(5, K))
    assert pair_features(h, z).shape == (5, 3 * K)
    assert pair_features(h, z, use_diff=False).shape == (5, 2 * K)
    assert pair_features(h, z, use_pre=False, use_diff=False).shape == (5, K)
    # the third block must literally be the difference, in that order
    np.testing.assert_allclose(pair_features(h, z)[:, 2 * K:], z - h)
    with pytest.raises(ValueError):
        pair_features(rng.normal(size=(5, K)), rng.normal(size=(4, K)))


def test_unknown_family_is_rejected(data) -> None:
    H, Z, A, S = data
    with pytest.raises(ValueError):
        ActionPredictor.fit(H, Z, A, scenes=S, family="transformer")


@pytest.mark.parametrize("family", ["ridge", "mlp"])
def test_recovers_an_exactly_linear_inverse(data, family: str) -> None:
    """Held-out-SCENE action error must be a small fraction of the action range.

    The bar is 1% of range for both families. Ridge should be far inside it; the MLP is bounded by
    its linear skip, which is precisely why the skip is there -- an unskipped MLP scored 3.1% here
    and that regression is what this test would catch.
    """
    H, Z, A, S = data
    fit, val = scene_holdout(S, 0.2, seed=0)
    pi = ActionPredictor.fit(H[fit], Z[fit], A[fit], scenes=S[fit], family=family,
                             n_members=2, cfg={"epochs": 150, "patience": 30})
    mae = float(np.abs(pi.action_for(H[val], Z[val]) - A[val]).mean())
    rng_a = float(A.max() - A.min())
    assert mae / rng_a < 0.01, f"{family}: MAE {mae:.4f} = {100 * mae / rng_a:.2f}% of range"


@pytest.mark.parametrize("family", ["ridge", "mlp"])
def test_save_load_round_trip_is_exact(data, family: str, tmp_path) -> None:
    H, Z, A, S = data
    fit, val = scene_holdout(S, 0.2, seed=0)
    pi = ActionPredictor.fit(H[fit], Z[fit], A[fit], scenes=S[fit], family=family,
                             n_members=2, cfg={"epochs": 40, "patience": 10})
    before = pi.action_for(H[val], Z[val])
    pi.save(tmp_path / "pol")
    after = ActionPredictor.load(tmp_path / "pol").action_for(H[val], Z[val])
    np.testing.assert_allclose(before, after, atol=1e-9)


def test_prediction_tracks_the_TARGET_not_just_the_context(data) -> None:
    """Hold the context fixed, sweep the target: the action must move, monotonically and with gain.

    This is the unit-test form of the shuffled control. A policy that ignored ``z_target`` would
    still post a respectable MAE (the context alone predicts v_in, which explains part of the
    action) while being completely useless for commanding an outcome.
    """
    H, Z, A, S = data
    fit, _ = scene_holdout(S, 0.2, seed=0)
    pi = ActionPredictor.fit(H[fit], Z[fit], A[fit], scenes=S[fit], family="ridge")
    # Rebuild the basis `synthetic()` used, drawing in the SAME order so the swept targets land on
    # the training distribution rather than in a random direction the policy never saw.
    r = np.random.default_rng(0)
    r.uniform(0.85, 1.15, N_SCENES)
    r.uniform(0.75, 2.9, (N_SCENES, N_RANK))
    b_pre, b_out = r.normal(size=K), r.normal(size=K)
    v_in = 1.0
    h = np.repeat((b_pre * v_in)[None], 8, axis=0)
    v_stars = np.linspace(-0.8, -3.0, 8)
    z = b_pre * v_in + np.outer(v_stars, b_out)
    a = pi.action_for(h, z)
    assert np.all(np.diff(a) < 0), f"action must decrease as the commanded speed grows: {a}"
    slope = float(np.polyfit(v_stars, a, 1)[0])
    assert 0.3 < slope * ALPHA < 1.7, f"gain on the command axis is {slope * ALPHA:.3f}, expected ~1"


def test_ensemble_spread_is_zero_for_ridge_and_positive_for_mlp(data) -> None:
    H, Z, A, S = data
    fit, val = scene_holdout(S, 0.2, seed=0)
    r = ActionPredictor.fit(H[fit], Z[fit], A[fit], scenes=S[fit], family="ridge")
    assert np.allclose(r.action_spread(H[val], Z[val]), 0.0)
    m = ActionPredictor.fit(H[fit], Z[fit], A[fit], scenes=S[fit], family="mlp",
                            n_members=3, cfg={"epochs": 40, "patience": 10})
    assert (m.action_spread(H[val], Z[val]) > 0).mean() > 0.9


def test_predictions_are_finite_and_inside_a_sane_envelope(data) -> None:
    H, Z, A, S = data
    fit, val = scene_holdout(S, 0.2, seed=0)
    for family in ("ridge", "mlp"):
        pi = ActionPredictor.fit(H[fit], Z[fit], A[fit], scenes=S[fit], family=family,
                                 n_members=2, cfg={"epochs": 40, "patience": 10})
        a = pi.action_for(H[val], Z[val])
        assert np.isfinite(a).all()
        # the striker envelope is roughly [0.3, -1.6]; nothing should leave it by more than a margin
        assert a.min() > -3.0 and a.max() < 1.0, f"{family}: {a.min():.2f}..{a.max():.2f}"


def test_too_few_scenes_is_an_error_not_a_silent_empty_split() -> None:
    rng = np.random.default_rng(0)
    H, Z = rng.normal(size=(4, K)), rng.normal(size=(4, K))
    with pytest.raises(ValueError):
        ActionPredictor.fit(H, Z, rng.normal(size=4), scenes=np.zeros(4, dtype=int), family="ridge")
