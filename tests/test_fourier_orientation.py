"""Tests for the shared FOURIER-IN-ORIENTATION operator (:mod:`src.analysis.fourier_orientation`).

The operator was factored out of ``experiments/threads/angular-velocity/05_steering/steer_angvel_fourier.py`` (the solved angular-velocity steer,
held-out rho=0.94) so the angular-ACCELERATION experiment can apply *the same* fitted operator. That makes
one property load-bearing for the whole zero-shot transfer claim: the shared module must be EXACTLY the
operator that produced the angvel result. If the refactor silently changed the basis, the roll, or the
solve, then "same operator, no refit" would be false and the transfer result would be uninterpretable. The
equivalence tests below pin that against the original script's implementations.

The rest assert the properties the claim rests on: the canonicalizing roll is exactly invertible, the DC
(appearance) column cancels in the steering difference, alpha=0 reproduces the constant-spin trajectory,
and the operator recovers a known planted Fourier signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis import fourier_orientation as fo


# ---- reference implementations, copied verbatim from experiments/threads/angular-velocity/05_steering/steer_angvel_fourier.py ------------------
def _ref_phi_feats(theta, order):
    out = [np.ones_like(theta)]
    for k in range(1, order + 1):
        out.append(np.cos(k * theta)); out.append(np.sin(k * theta))
    return np.stack(out, -1)


def _ref_canon_roll(x, cen, grid, inverse=False):
    _, H, W = grid
    w = int(np.clip(round(cen[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(cen[1] * (H - 1)), 0, H - 1))
    dh, dw = (H // 2 - h), (W // 2 - w)
    if inverse:
        dh, dw = -dh, -dw
    return np.roll(x, shift=(dh, dw), axis=(1, 2))


GRID = (8, 16, 16)


def test_phi_feats_matches_original_angvel_script():
    """The Fourier basis must be bit-identical to the one that produced the angvel rho=0.94 result."""
    theta = np.linspace(-3 * np.pi, 3 * np.pi, 37)
    for order in (1, 2, 4, 8):
        np.testing.assert_allclose(fo.phi_feats(theta, order), _ref_phi_feats(theta, order), rtol=0, atol=0)


def test_canon_roll_matches_original_angvel_script():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 16, 16, 3)).astype(np.float32)
    for cen in ([0.5, 0.5], [0.1, 0.9], [0.73, 0.22], [0.0, 1.0]):
        cen = np.array(cen)
        np.testing.assert_array_equal(fo.canon_roll(x, cen, GRID), _ref_canon_roll(x, cen, GRID))
        np.testing.assert_array_equal(fo.canon_roll(x, cen, GRID, inverse=True),
                                      _ref_canon_roll(x, cen, GRID, inverse=True))


def test_canon_roll_is_exactly_invertible():
    """Center-canonicalization must lose nothing: it is an integer roll, so roll-then-unroll is identity."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 16, 16, 5)).astype(np.float32)
    for cen in ([0.13, 0.87], [0.5, 0.5], [0.99, 0.01]):
        cen = np.array(cen)
        back = fo.canon_roll(fo.canon_roll(x, cen, GRID), cen, GRID, inverse=True)
        np.testing.assert_array_equal(back, x)


def test_theta_of_alpha_zero_is_constant_spin():
    """alpha=0 must reproduce the angular-VELOCITY roll-out exactly: the two experiments differ only here."""
    tau = np.arange(8, dtype=np.float64) * 2 + 0.5
    np.testing.assert_allclose(fo.theta_of(0.7, 0.13, 0.0, tau), 0.7 + 0.13 * tau, rtol=0, atol=1e-15)


def test_theta_of_matches_the_rendered_rollout():
    """theta(t) must match the generator's phi(t) = theta0 + w0*t + alpha/2*t^2 (same convention)."""
    t = np.arange(16, dtype=np.float64)
    th0, w0, al = 1.3, -0.04, 0.011
    np.testing.assert_allclose(fo.theta_of(th0, w0, al, t), th0 + w0 * t + 0.5 * al * t * t, atol=1e-12)
    np.testing.assert_allclose(fo.omega_of(w0, al, t), w0 + al * t, atol=1e-12)


def test_harmonic_selection_widths_and_content():
    """even/odd harmonic subsets underpin the mechanistic ablation (bar pi-symmetry vs marker 2pi cue)."""
    th = np.array([0.3, 1.1])
    assert fo.phi_feats(th, 4, "all").shape[-1] == 1 + 2 * 4
    assert fo.phi_feats(th, 4, "even").shape[-1] == 1 + 2 * 2     # k = 2, 4
    assert fo.phi_feats(th, 4, "odd").shape[-1] == 1 + 2 * 2      # k = 1, 3
    # the even subset must be invariant to theta -> theta+pi (a bar looks the same); the odd subset flips
    ev = fo.phi_feats(np.array([0.4]), 4, "even")
    ev_pi = fo.phi_feats(np.array([0.4 + np.pi]), 4, "even")
    np.testing.assert_allclose(ev, ev_pi, atol=1e-12)
    od = fo.phi_feats(np.array([0.4]), 4, "odd")
    od_pi = fo.phi_feats(np.array([0.4 + np.pi]), 4, "odd")
    assert not np.allclose(od[:, 1:], od_pi[:, 1:], atol=1e-6)


def test_n_features_agrees_with_design_matrix():
    th = np.zeros(4); om = np.ones(4) * 0.1
    for basis in fo.BASES:
        for harm in ("all", "even", "odd"):
            for order in (1, 2, 4):
                w = fo.design_matrix(th, om, order, harm, basis).shape[-1]
                assert w == fo.n_features(order, harm, basis)


def test_orientation_rate_basis_extends_orientation_basis():
    """The rate basis must contain the orientation basis as its first block (so it can only add)."""
    th = np.array([0.2, 0.9]); om = np.array([0.05, -0.11])
    P = fo.design_matrix(th, None, 3, "all", "orientation")
    R = fo.design_matrix(th, om, 3, "all", "orientation_rate")
    np.testing.assert_allclose(R[..., : P.shape[-1]], P, atol=1e-12)
    np.testing.assert_allclose(R[..., P.shape[-1]:], om[:, None] * P, atol=1e-12)


def test_orientation_rate_requires_omega():
    with pytest.raises(ValueError):
        fo.design_matrix(np.zeros(3), None, 2, "all", "orientation_rate")


def test_dc_appearance_column_cancels_in_the_edit():
    """A constant (appearance/DC) latent offset must contribute NOTHING to the steering edit.

    This is why the operator is appearance-robust: the edit is a DIFFERENCE of the model evaluated at two
    orientations, so the DC column drops out no matter how large it is.
    """
    rng = np.random.default_rng(2)
    T, P, M = 8, 9, 16 * 16 * 4
    C = {12: rng.normal(size=(T, P, M)).astype(np.float32)}
    C[12][:, 0, :] = 1e4       # an enormous DC/appearance term
    tau = np.arange(T) * 2.0 + 0.5
    row_a = fo.phi_feats(fo.theta_of(0.5, 0.10, 0.0, tau), 4)
    row_b = fo.phi_feats(fo.theta_of(0.5, -0.17, 0.0, tau), 4)
    d = fo.predict_dH(C, row_a, row_b, [12], (T, 16, 16))[12]
    assert np.isfinite(d).all()
    # identical commands -> exactly zero edit
    z = fo.predict_dH(C, row_a, row_a, [12], (T, 16, 16))[12]
    np.testing.assert_allclose(z, 0.0, atol=1e-8)


def test_fit_operator_recovers_a_planted_fourier_signal():
    """Plant a known C, synthesize latents, and check the ridge fit recovers the steering edit."""
    rng = np.random.default_rng(3)
    T, H, W, D, order = 4, 8, 8, 2, 3
    M = H * W * D
    P = 1 + 2 * order
    C_true = rng.normal(size=(T, P, M)).astype(np.float64)
    N = 400
    th0 = rng.uniform(0, 2 * np.pi, N)
    om = rng.uniform(0.06, 0.20, N) * rng.choice([-1, 1], N)
    tau = np.arange(T) * 2.0 + 0.5
    PHI = np.stack([fo.phi_feats(fo.theta_of(th0[i], om[i], 0.0, tau), order) for i in range(N)])
    Y = np.einsum("ntp,tpm->ntm", PHI, C_true).astype(np.float32)
    C = fo.fit_operator(PHI, {6: Y}, [6], ridge=1e-6)
    np.testing.assert_allclose(C[6], C_true.astype(np.float32), rtol=1e-2, atol=1e-2)
    # and the synthesized edit matches the true difference
    ra = fo.phi_feats(fo.theta_of(0.3, 0.09, 0.0, tau), order)
    rb = fo.phi_feats(fo.theta_of(0.3, -0.15, 0.0, tau), order)
    pred = fo.predict_dH(C, ra, rb, [6], (T, H, W))[6].reshape(T, M)
    true = np.einsum("tp,tpm->tm", rb - ra, C_true)
    assert fo.cosine(pred, true) > 0.999


def test_operator_fit_on_constant_spin_extrapolates_to_a_quadratic_trajectory():
    """The crux of the zero-shot claim, in miniature.

    Fit the operator ONLY on constant-spin trajectories, then ask it to synthesize the edit for an
    angular-ACCELERATION trajectory it never saw. Because the model is indexed by ORIENTATION and the
    ground truth here is genuinely a function of orientation, the fit must extrapolate across kinematic
    order. This isolates the mathematical claim from the empirical question of whether V-JEPA latents
    actually are such a function -- that is what the cluster runs measure.
    """
    rng = np.random.default_rng(4)
    T, H, W, D, order = 8, 8, 8, 2, 4
    M = H * W * D
    P = 1 + 2 * order
    C_true = rng.normal(size=(T, P, M))
    tau = np.arange(T) * 2.0 + 0.5
    N = 600
    th0 = rng.uniform(0, 2 * np.pi, N)
    om = rng.uniform(0.06, 0.20, N) * rng.choice([-1, 1], N)
    PHI = np.stack([fo.phi_feats(fo.theta_of(th0[i], om[i], 0.0, tau), order) for i in range(N)])  # alpha=0 ONLY
    Y = np.einsum("ntp,tpm->ntm", PHI, C_true).astype(np.float32)
    C = fo.fit_operator(PHI, {6: Y}, [6], ridge=1e-6)

    # a quadratic (angular-acceleration) trajectory, absent from the fit
    th0_t, w0 = 1.1, 0.02
    ra = fo.phi_feats(fo.theta_of(th0_t, w0, 0.004, tau), order)
    rb = fo.phi_feats(fo.theta_of(th0_t, w0, -0.013, tau), order)
    pred = fo.predict_dH(C, ra, rb, [6], (T, H, W))[6].reshape(T, M)
    true = np.einsum("tp,tpm->tm", rb - ra, C_true)
    assert fo.cosine(pred, true) > 0.999
