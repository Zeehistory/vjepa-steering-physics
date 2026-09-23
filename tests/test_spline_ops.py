"""CPU validation for the temporal spline basis used by acceleration steering.

The claims that matter downstream, each pinned here: the basis is a partition of unity (so a constant
edit is exactly representable and K=1 IS the classical global-vector operator); K=2 is exactly linear in
t; K=T reproduces any profile exactly (so it IS the unconstrained cmd_prof operator); the pooling and
broadcast round-trip is exact; and Catmull-Rom interpolates its control points while extrapolating a
genuinely curved path where the straight-line control cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import spline_ops as so

T = 8
D = 5


def test_basis_shape_and_partition_of_unity():
    for K in (1, 2, 3, 4, 6, 8):
        B = so.spline_basis(T, K)
        assert B.shape == (T, K)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(T), atol=1e-12)
        assert (B >= -1e-12).all(), "B-spline basis must be non-negative"


def test_k1_is_the_constant_global_vector_operator():
    """K=1 collapses to one vector applied identically at every temporal token."""
    B = so.spline_basis(T, 1)
    np.testing.assert_allclose(B, np.ones((T, 1)), atol=1e-12)
    C = np.arange(D, dtype=float).reshape(1, D)
    prof = so.reconstruct_profile(C, B)
    assert prof.shape == (T, D)
    for t in range(T):
        np.testing.assert_allclose(prof[t], C[0], atol=1e-12)


def test_k2_is_exactly_linear_in_t():
    B = so.spline_basis(T, 2)
    ramp = np.linspace(0.0, 1.0, T)
    # the second basis column is the normalized ramp; first is its complement
    np.testing.assert_allclose(B[:, 1], ramp, atol=1e-12)
    np.testing.assert_allclose(B[:, 0], 1.0 - ramp, atol=1e-12)
    # any affine-in-t profile is reproduced with zero error
    prof = np.outer(3.0 + 2.0 * ramp, np.arange(D, dtype=float))
    np.testing.assert_allclose(so.smooth_profile(prof, B), prof, atol=1e-10)


def test_full_rank_basis_reproduces_any_profile():
    """K=T is the unconstrained per-token profile — the existing cmd_prof parametrization."""
    B = so.spline_basis(T, T)
    assert np.linalg.matrix_rank(B) == T
    rng = np.random.default_rng(0)
    prof = rng.standard_normal((T, D))
    np.testing.assert_allclose(so.smooth_profile(prof, B), prof, atol=1e-8)


def test_smoothing_error_decreases_with_more_knots():
    """More control points can only fit a fixed profile better (nested least-squares subspaces)."""
    rng = np.random.default_rng(1)
    ramp = np.linspace(0, 1, T)
    # a curved (quadratic-ish) profile plus mild noise, i.e. the shape acceleration actually induces
    prof = np.outer(0.3 + ramp**2, rng.standard_normal(D)) + 0.01 * rng.standard_normal((T, D))
    errs = []
    for K in (1, 2, 3, 4, 6, 8):
        approx = so.smooth_profile(prof, so.spline_basis(T, K))
        errs.append(float(np.linalg.norm(approx - prof)))
    assert all(errs[i] >= errs[i + 1] - 1e-9 for i in range(len(errs) - 1)), errs
    assert errs[-1] < 1e-8, "K=T must be exact"
    assert errs[0] > errs[2], "a curved profile must be poorly served by a constant edit"


def test_growing_magnitude_profile_needs_more_than_a_constant():
    """The measured shape: |edit| grows monotonically along the clip (curvature_summary dH_norm_per_t).

    A constant edit captures the mean but none of the growth; a 2-knot ramp captures nearly all of it.
    """
    measured_l12 = np.array([448.79, 440.12, 602.92, 670.09, 747.89, 775.28, 838.14, 877.65])
    direction = np.zeros(D); direction[0] = 1.0
    prof = np.outer(measured_l12, direction)
    rel = {}
    for K in (1, 2, 3, 8):
        approx = so.smooth_profile(prof, so.spline_basis(T, K))
        rel[K] = float(np.linalg.norm(approx - prof) / np.linalg.norm(prof))
    assert rel[1] > 0.15, f"constant edit should badly miss a growing profile, got {rel[1]}"
    assert rel[2] < 0.05, f"a linear ramp should nearly capture it, got {rel[2]}"
    assert rel[8] < 1e-9


def test_pool_broadcast_roundtrip_is_exact():
    grid = (T, 4, 3)
    rng = np.random.default_rng(2)
    prof = rng.standard_normal((T, D))
    flat = so.broadcast_profile(prof, grid)
    assert flat.size == T * 4 * 3 * D
    np.testing.assert_allclose(so.temporal_profile(flat, grid), prof, atol=1e-12)


def test_catmull_rom_interpolates_control_points():
    rng = np.random.default_rng(3)
    P = rng.standard_normal((6, D))
    for i in range(6):
        np.testing.assert_allclose(so.catmull_rom(P, float(i)), P[i], atol=1e-10)


def test_catmull_rom_matches_line_on_collinear_points_but_not_on_curved_ones():
    # collinear control points: the spline must degenerate to the straight line
    base = np.arange(6, dtype=float)
    P_line = np.stack([base * 2.0, base * -1.0], axis=1)
    for s in (0.3, 1.7, 3.5, 4.9):
        np.testing.assert_allclose(so.catmull_rom(P_line, s), so.linear_eval(P_line, s), atol=1e-10)
    # curved control points: spline and line must disagree (that gap is the whole hypothesis)
    P_curve = np.stack([base, base**2], axis=1)
    gaps = [np.linalg.norm(so.catmull_rom(P_curve, s) - so.linear_eval(P_curve, s))
            for s in (0.5, 2.5, 4.5)]
    assert min(gaps) > 1e-3, gaps


def test_catmull_rom_beats_the_line_when_INTERPOLATING_a_curved_family():
    """The paper's claim, in miniature: on a curved latent family, spline interp beats linear interp.

    This is the arm we actually run (leave-one-rank-out), so it is the property worth pinning.
    """
    base = np.arange(6, dtype=float)
    P = np.stack([base, base**2], axis=1)          # curved family
    for s in (1.5, 2.5, 3.5):
        truth = np.array([s, s**2])
        spline_err = np.linalg.norm(so.catmull_rom(P, s) - truth)
        line_err = np.linalg.norm(so.linear_eval(P, s) - truth)
        assert spline_err < line_err, f"s={s}: spline {spline_err} vs line {line_err}"


def test_catmull_rom_extrapolation_is_NOT_better_than_a_line():
    """Documented negative — the reason the family arm interpolates instead of extrapolating.

    Standard Catmull-Rom reflects a phantom point at the ends, which forces the boundary tangent toward
    the last chord and FLATTENS the curve just where extrapolation needs it to keep bending. Past the
    final control point the spline therefore undershoots a convex family by more than the straight line
    does. Pinned so nobody 'fixes' the family arm by asking the spline to extrapolate.
    """
    base = np.arange(5, dtype=float)
    P = np.stack([base, base**2], axis=1)
    truth = np.array([4.5, 4.5**2])
    spline_err = np.linalg.norm(so.catmull_rom(P, 4.5) - truth)
    line_err = np.linalg.norm(so.linear_eval(P, 4.5) - truth)
    assert spline_err > line_err


def test_closed_catmull_rom_interpolates_and_wraps():
    """The acceleration family is a LOOP (directions sweep a full turn), so the spline must be periodic."""
    M = 8
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    P = np.stack([np.cos(ang), np.sin(ang)], axis=1)      # unit circle control points
    for i in range(M):
        np.testing.assert_allclose(so.catmull_rom_closed(P, float(i)), P[i], atol=1e-10)
    # wrapping is seamless: index M is index 0
    np.testing.assert_allclose(so.catmull_rom_closed(P, float(M)), P[0], atol=1e-10)
    # on a circle the periodic spline must sit much closer to the circle than the chord does
    for s in (0.5, 3.5, 7.5):                              # 7.5 straddles the wrap
        r_spline = np.linalg.norm(so.catmull_rom_closed(P, s))
        r_line = np.linalg.norm(so.linear_eval_closed(P, s))
        assert abs(r_spline - 1.0) < abs(r_line - 1.0), f"s={s}: {r_spline} vs {r_line}"


def test_angular_index_places_queries_around_the_loop():
    M = 8
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    for i in range(M):
        assert so.angular_index(ang, ang[i]) == pytest.approx(float(i), abs=1e-9)
    # midway between control points 2 and 3
    mid = 0.5 * (ang[2] + ang[3])
    assert so.angular_index(ang, mid) == pytest.approx(2.5, abs=1e-9)
    # a query past the last control point falls in the closing segment [M-1, M)
    assert 7.0 <= so.angular_index(ang, ang[7] + 0.5 * (ang[1] - ang[0])) < 8.0
    # negative / >2pi queries are wrapped
    assert so.angular_index(ang, ang[3] + 2 * np.pi) == pytest.approx(3.0, abs=1e-9)


def test_interp_index_inverts_the_parametrization():
    vals = np.array([0.0, 1.0, 2.0, 4.0])
    for j, v in enumerate(vals):
        assert so.interp_index(vals, v) == pytest.approx(float(j), abs=1e-9)
    assert so.interp_index(vals, 3.0) == pytest.approx(2.5, abs=1e-9)   # midway in [2, 4]
    assert so.interp_index(vals, 6.0) == pytest.approx(4.0, abs=1e-9)   # extrapolates past the end
    assert so.interp_index(vals, -1.0) == pytest.approx(-1.0, abs=1e-9)


def test_project_reconstruct_shapes():
    B = so.spline_basis(T, 4)
    rng = np.random.default_rng(4)
    prof = rng.standard_normal((T, D))
    C = so.project_profile(prof, B)
    assert C.shape == (4, D)
    assert so.reconstruct_profile(C, B).shape == (T, D)
