"""CPU validation for concept-space manifold steering (the arXiv:2605.05115 port).

The claims that matter downstream, each pinned here: the thin-plate design matrix is well formed and
its ``linear_only`` variant really is the affine sub-model; the fitted surface recovers a genuinely
CURVED concept->activation map that its own zero-curvature control cannot (this is the whole
experiment -- if the control matched, ``man`` would be measuring nothing); the steering edit is exactly
the difference of two points on the surface and is therefore antisymmetric and appearance-free; and
``s^{-1}`` returns the concept coordinate it was given.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import manifold_ops as mo

SCALE = 0.002          # the acceleration magnitudes this is actually used on
DIM = 120


def _curved_world(n, seed=0, scale=SCALE):
    """A concept->activation map with genuine second-order structure in the concept."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(scale=scale, size=(n, 2))
    A = np.random.default_rng(99).normal(size=(4, DIM))
    u = Z / scale
    F = np.stack([u[:, 0], u[:, 1], u[:, 0] ** 2 - u[:, 1] ** 2, u[:, 0] * u[:, 1]], axis=1)
    return Z, F @ A


def test_tps_kernel_is_zero_at_zero_and_matches_r2logr():
    r = np.array([0.0, 0.5, 1.0, 2.0])
    k = mo.tps_kernel(r)
    assert k[0] == 0.0
    assert k[2] == pytest.approx(0.0)                 # 1^2 log 1 = 0
    np.testing.assert_allclose(k[1:], np.array([0.5, 1.0, 2.0]) ** 2 * np.log([0.5, 1.0, 2.0]))


def test_design_matrix_shapes_and_linear_only_is_the_affine_block():
    Z = np.random.default_rng(0).normal(size=(7, 2))
    C = np.random.default_rng(1).normal(size=(5, 2))
    A = mo.tps_design(Z, C)
    assert A.shape == (7, 5 + 3)
    Alin = mo.tps_design(Z, C, linear_only=True)
    assert Alin.shape == (7, 3)
    np.testing.assert_allclose(A[:, -3:], Alin)       # the affine columns are identical


def test_curvature_control_is_a_real_control():
    """The paper's mechanism only means something if the straight-manifold fit is genuinely worse."""
    Z, Y = _curved_world(1200, seed=0)
    mean = Y.mean(axis=0)
    m = mo.fit_manifold(Z, Y - mean, n_centers=48, k_pca=24, ridge=1e-4)
    mlin = mo.fit_manifold(Z, Y - mean, n_centers=48, k_pca=24, ridge=1e-4, linear_only=True)
    Zt, Yt = _curved_world(300, seed=7)
    Yt = Yt - mean

    def r2(mf):
        P = np.stack([mf(z) for z in Zt])
        return 1.0 - ((P - Yt) ** 2).sum() / ((Yt - Yt.mean(axis=0)) ** 2).sum()

    assert r2(m) > 0.95, "thin-plate surface must recover a smooth curved map"
    assert r2(m) - r2(mlin) > 0.5, "the zero-curvature control must be clearly worse"


def test_edit_is_the_difference_of_two_points_on_the_surface():
    """The steering edit must be exactly s(a_b) - s(a_a): antisymmetric, and free of any constant."""
    Z, Y = _curved_world(600, seed=3)
    m = mo.fit_manifold(Z, Y - Y.mean(axis=0), n_centers=32, k_pca=16, ridge=1e-4)
    a, b = Z[0], Z[5]
    np.testing.assert_allclose(m.edit(a, b), m(b) - m(a), atol=1e-9)
    np.testing.assert_allclose(m.edit(a, b), -m.edit(b, a), atol=1e-9)
    np.testing.assert_allclose(m.edit(a, a), np.zeros(DIM), atol=1e-9)
    # a constant appearance offset added to the surface's mean cannot change the edit
    shifted = mo.ConceptManifold(m.centers, m.coef, m.basis, m.mean + 3.0, m.zscale, m.linear_only)
    np.testing.assert_allclose(shifted.edit(a, b), m.edit(a, b), atol=1e-9)


def test_inverse_recovers_the_concept_coordinate():
    """s^{-1}(s(a)) == a, to the resolution of the projection search."""
    Z, Y = _curved_world(800, seed=11)
    m = mo.fit_manifold(Z, Y - Y.mean(axis=0), n_centers=48, k_pca=24, ridge=1e-5)
    for a in (Z[3], Z[40], Z[100]):
        rel = np.linalg.norm(m.invert(m(a)) - a) / np.linalg.norm(a)
        assert rel < 0.05, f"inverse map missed {a} by {rel:.3f} relative"


def test_kmeans_centers_stay_inside_the_data_hull():
    Z = np.random.default_rng(0).normal(size=(500, 2))
    C = mo.kmeans_centers(Z, 20)
    assert C.shape == (20, 2)
    assert (C.min(axis=0) >= Z.min(axis=0) - 1e-9).all()
    assert (C.max(axis=0) <= Z.max(axis=0) + 1e-9).all()


def test_randomized_pca_matches_the_dense_svd_subspace():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(200, 60)) @ rng.normal(size=(60, 60))
    Y = Y - Y.mean(axis=0)
    B, _ = mo.randomized_pca(Y, 10)
    Vt = np.linalg.svd(Y, full_matrices=False)[2][:10]
    # compare the SUBSPACES, not the individual vectors (signs/ordering within ties are free)
    ang = np.linalg.svd(B @ Vt.T)[1]
    # randomized range-finding is approximate by construction; the tail direction is the loose one
    np.testing.assert_allclose(ang, np.ones(10), atol=5e-3)


def test_save_load_round_trip(tmp_path):
    Z, Y = _curved_world(400, seed=5)
    m = mo.fit_manifold(Z, Y - Y.mean(axis=0), n_centers=24, k_pca=12, ridge=1e-4)
    mo.save_manifold(m, tmp_path / "man_L18")
    m2 = mo.load_manifold(tmp_path / "man_L18")
    np.testing.assert_allclose(m2.edit(Z[0], Z[1]), m.edit(Z[0], Z[1]), rtol=1e-5, atol=1e-6)


def test_project_block_matches_the_naive_projection():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(8, 300)).astype(np.float32)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    np.testing.assert_allclose(mo.project_block(U, X), (U @ X).T, rtol=1e-5, atol=1e-5)
