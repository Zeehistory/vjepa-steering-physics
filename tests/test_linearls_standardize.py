"""Regression tests for `LinearLS.solve(standardize=...)`.

Guards the scale bug that made the velocity command operator steer heading and ignore speed:
`ridge * I` applied to raw `command_features` penalises the ~0.02-RMS speed columns ~47x harder than
the ~1-RMS direction columns, so the fit drops magnitude.
"""
import numpy as np
import pytest

from src.analysis.velocity_ops import LinearLS, command_features


def _fit(X, y, ridge=1.0, standardize=False):
    op = LinearLS(X.shape[1], y.shape[1], ridge=ridge)
    op.add(X, y)
    return op.solve(standardize=standardize)


def test_standardize_recovers_small_scale_coefficient():
    """A column 50x smaller in scale must not be shrunk 50x harder."""
    rng = np.random.default_rng(0)
    n = 400
    X = np.c_[rng.normal(size=n), 0.02 * rng.normal(size=n)]   # scales 1 and 0.02
    beta = np.array([[1.0], [50.0]])                            # equal contribution to y
    y = X @ beta

    raw = _fit(X, y, standardize=False).ravel()
    std = _fit(X, y, standardize=True).ravel()

    assert raw[1] < 0.25 * beta[1, 0], "raw ridge is expected to crush the small-scale column"
    assert std == pytest.approx(beta.ravel(), rel=0.02), "standardized ridge must recover both"


def test_standardize_is_a_noop_when_columns_share_scale():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = X @ rng.normal(size=(4, 2))
    assert _fit(X, y, standardize=True) == pytest.approx(_fit(X, y, standardize=False), rel=0.05)


def test_default_is_unchanged():
    """solve() with no argument must stay byte-identical to the pre-fix behaviour."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3)); y = rng.normal(size=(50, 2))
    op = LinearLS(3, 2, ridge=1.0); op.add(X, y)
    expected = np.linalg.solve(X.T @ X + np.eye(3), X.T @ y)
    assert op.solve() == pytest.approx(expected)


def test_command_features_scale_gap_is_real():
    """The gap this flag exists for: speed columns ~0.02 RMS, direction columns ~1 RMS."""
    rng = np.random.default_rng(3)
    P = np.array([command_features(0.02 * rng.normal(size=2), 0.02 * rng.normal(size=2))
                  for _ in range(500)])
    rms = np.sqrt((P ** 2).mean(0))
    speed_cols = [1, 2, 3, 4, 5, 6, 7, 8]      # v_b, v_a, dv, |v_b|, |v_a|
    dir_cols = [0, 9, 10, 11, 12]              # bias, u_b, u_a
    assert rms[speed_cols].max() < 0.1
    assert rms[dir_cols].min() > 0.5
