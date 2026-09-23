"""Tests for the position-aware command features (the 3D perspective lever).

``command_features_pos`` exists because on perspective data the image velocity a latent edit must
produce depends on WHERE the object is, which the velocity-only base features cannot express. These
pin the two properties that make it a valid experiment: it strictly extends the base features (so it
can never do worse in-sample), and its position columns actually carry the position-by-velocity
interaction rather than a mere offset.
"""

from __future__ import annotations

import numpy as np

from src.analysis import velocity_ops as vo


def test_extends_base_features_exactly() -> None:
    va, vb, p = np.array([0.01, -0.02]), np.array([-0.015, 0.005]), np.array([0.3, 0.7])
    phi = vo.command_features_pos(va, vb, p)
    assert phi.shape == (vo.COMMAND_FEATURE_DIM_POS,) == (27,)
    # the first 13 columns ARE the base features -> the richer map can always fall back to the base one
    assert np.allclose(phi[:13], vo.command_features(va, vb))


def test_position_columns_are_interactions_not_just_an_offset() -> None:
    """A bare position column could only shift the edit; the outer products let the map RESCALE the
    velocity command per position, which is what J(p)^-1 modulation needs."""
    va, vb = np.array([0.01, -0.02]), np.array([-0.015, 0.005])
    p1, p2 = np.array([0.2, 0.5]), np.array([0.8, 0.5])
    f1, f2 = vo.command_features_pos(va, vb, p1), vo.command_features_pos(va, vb, p2)
    assert np.allclose(f1[:13], f2[:13]), "same command -> identical base block"
    assert not np.allclose(f1[13:], f2[13:]), "different position must change the position block"
    # doubling the command must scale the interaction columns (they are bilinear in q and v), while the
    # pure-position columns stay put
    f3 = vo.command_features_pos(2 * va, 2 * vb, p1)
    assert np.allclose(f3[13:15], f1[13:15]), "q columns are command-independent"
    assert np.allclose(f3[15:27], 2 * f1[15:27]), "outer(q, v) columns are linear in the command"


def test_centred_at_image_centre() -> None:
    """q = pos - 0.5, so a ball at the image centre contributes no position terms -- the operator then
    degenerates to the base features rather than to something arbitrary."""
    va, vb = np.array([0.01, -0.02]), np.array([-0.015, 0.005])
    phi = vo.command_features_pos(va, vb, np.array([0.5, 0.5]))
    assert np.allclose(phi[13:], 0.0)


def test_recovers_a_position_dependent_rescaling() -> None:
    """The point of the features: a LINEAR map on them can express a target that the base features
    cannot -- namely one whose velocity gain varies with position (a stand-in for perspective)."""
    rng = np.random.default_rng(0)
    # ground truth: edit = (1 + 2*q_x) * vb  -- gain depends on where the ball is
    def target(vb, p):
        return (1.0 + 2.0 * (p[0] - 0.5)) * vb

    def fit(feat_fn, dim):
        X, Y = [], []
        for _ in range(400):
            p = rng.uniform(0.1, 0.9, 2)
            va, vb = rng.normal(0, 0.02, 2), rng.normal(0, 0.02, 2)
            X.append(feat_fn(va, vb, p)); Y.append(target(vb, p))
        X, Y = np.array(X), np.array(Y)
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        return float(np.mean((X @ W - Y) ** 2))

    err_pos = fit(vo.command_features_pos, 27)
    err_base = fit(lambda va, vb, p: vo.command_features(va, vb), 13)
    assert err_pos < 0.02 * err_base, f"pos features must capture it: {err_pos:.2e} vs base {err_base:.2e}"
