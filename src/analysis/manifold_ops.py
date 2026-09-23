"""Concept-space MANIFOLD steering (port of arXiv:2605.05115, Wurgaft et al. 2026).

**What the paper actually does, and how it differs from what this repo already calls "spline
steering".** The repo's spline arm (``src/analysis/spline_ops.py``) puts a B-spline over the
*temporal token axis*: the edit is a smooth curve in ``t``. The paper puts a spline over the
*concept axis*: it fits a smooth surface through the activation centroids of each concept value,
calls that surface the activation manifold ``M_h``, and then steers by interpolating in the
manifold's INTRINSIC coordinates rather than straight through activation space,

    pi_m(t) = s((1 - t) u_0 + t u_1),      u_i = s^{-1}(h_i*)

where ``s : R^k -> A`` maps intrinsic (concept) coordinates to the manifold. For 1-D concept
domains they fit cubic splines through the centroids; for 2-D domains they fit **thin-plate
splines**, after reducing activations to 64 PCA dimensions. The claim under test is that the
concept->activation map is CURVED, so a straight line in activation space leaves the manifold and
produces off-distribution behaviour, while the curved path does not.

Here the concept domain is the 2-D acceleration vector ``a = (a_x, a_y)``, so this is exactly the
paper's thin-plate-spline case. The estimator is

    mu(a) = E[ R - mean_over_scene(R) | a ]

i.e. the expected SCENE-CENTRED spatially-pooled temporal profile at acceleration ``a``. Centring
within a scene at fit time is what removes appearance (the ``_mixed`` dataset randomizes colour and
background per scene); nothing at steer time needs it, because the edit is a DIFFERENCE

    edit(a_a -> a_b) = mu(a_b) - mu(a_a)

and any constant offset cancels. That difference is command-only: it reads the two acceleration
labels and never ``H_b``.

The linear special case ``manlin`` (drop the RBF block, keep only the affine part) is the control
that isolates curvature: same estimator, same data, same regularization, straight manifold. If
``man`` beats ``manlin`` the curvature is doing the work; if it does not, the paper's mechanism
does not transfer to this quantity and we say so.

Everything is pure numpy and small: the fitted artifacts are a ``(M+3, k)`` coefficient block and a
``(k, T*D)`` PCA basis per layer, a few MB total.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------------
# thin-plate spline basis over a 2-D concept domain
# ----------------------------------------------------------------------------------------------------
def tps_kernel(r: np.ndarray) -> np.ndarray:
    """Thin-plate radial basis ``phi(r) = r^2 log r`` (Duchon 1977), with ``phi(0) = 0``."""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    nz = r > 1e-12
    out[nz] = r[nz] ** 2 * np.log(r[nz])
    return out


def tps_design(Z: np.ndarray, centers: np.ndarray, linear_only: bool = False) -> np.ndarray:
    """Design matrix ``(N, M+3)`` = ``[phi(||z_i - c_j||) | 1 | z_x | z_y]``.

    ``linear_only=True`` drops the RBF block and returns just ``(N, 3)`` — the STRAIGHT-manifold
    control, identical in every other respect (same targets, same ridge, same fit routine).
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    aff = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    if linear_only:
        return aff
    C = np.asarray(centers, dtype=np.float64)
    r = np.linalg.norm(Z[:, None, :] - C[None, :, :], axis=2)
    return np.concatenate([tps_kernel(r), aff], axis=1)


def kmeans_centers(Z: np.ndarray, n_centers: int, iters: int = 60, seed: int = 0) -> np.ndarray:
    """Lloyd k-means on the 2-D concept points — where to put the thin-plate knots.

    The accelerations are not uniform on a disc (magnitude is drawn from a band and rescaled per
    scene), so a fixed grid would waste knots on empty regions and starve the populated annulus.
    """
    Z = np.asarray(Z, dtype=np.float64)
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), size=min(n_centers, len(Z)), replace=False)].copy()
    for _ in range(iters):
        d = np.linalg.norm(Z[:, None, :] - C[None, :, :], axis=2)
        lab = d.argmin(axis=1)
        newC = C.copy()
        for j in range(len(C)):
            m = lab == j
            if m.any():
                newC[j] = Z[m].mean(axis=0)
        if np.allclose(newC, C):
            break
        C = newC
    return C


# ----------------------------------------------------------------------------------------------------
# randomized PCA (the paper's "reduce activations to 64 dimensions before fitting the manifold")
# ----------------------------------------------------------------------------------------------------
def randomized_pca(Y: np.ndarray, k: int, oversample: int = 12, n_iter: int = 4,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` right singular vectors of ``Y (N, D)`` (already centred) and their singular values.

    Returns ``(basis (k, D), svals (k,))``. Randomized because ``D = T*D_model = 8192`` with
    ``N = 4000`` makes a full SVD wasteful when only 64 directions are wanted.
    """
    Y = np.asarray(Y, dtype=np.float64)
    N, D = Y.shape
    k = int(min(k, N, D))
    rng = np.random.default_rng(seed)
    Om = rng.standard_normal((D, k + oversample))
    Q = Y @ Om
    for _ in range(n_iter):                       # power iterations sharpen the spectrum decay
        Q, _ = np.linalg.qr(Q)
        Q = Y @ (Y.T @ Q)
    Q, _ = np.linalg.qr(Q)
    Bsmall = Q.T @ Y                              # (k+p, D)
    Ub, s, Vt = np.linalg.svd(Bsmall, full_matrices=False)
    return Vt[:k], s[:k]


# ----------------------------------------------------------------------------------------------------
# the fitted manifold
# ----------------------------------------------------------------------------------------------------
class ConceptManifold:
    """``s : concept -> activation profile``, fitted as a thin-plate spline in a PCA subspace.

    ``coef (M+3, k)`` maps the TPS design row to PCA coordinates; ``basis (k, D)`` lifts back to the
    profile space; ``mean (D,)`` is the training mean of the (scene-centred) profiles. Evaluation is

        s(a) = mean + design(a / zscale) @ coef @ basis
    """

    def __init__(self, centers, coef, basis, mean, zscale, linear_only=False):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.zscale = float(zscale)
        self.linear_only = bool(linear_only)

    # -- evaluation ----------------------------------------------------------------------------
    def coords(self, a) -> np.ndarray:
        """Intrinsic concept point -> PCA coordinates ``(k,)``."""
        Z = np.atleast_2d(np.asarray(a, dtype=np.float64)) / self.zscale
        return (tps_design(Z, self.centers, self.linear_only) @ self.coef)[0]

    def __call__(self, a) -> np.ndarray:
        """Concept point -> profile ``(D,)`` on the manifold."""
        return self.mean + self.coords(a) @ self.basis

    def edit(self, a_a, a_b) -> np.ndarray:
        """``s(a_b) - s(a_a)`` — the manifold-transport edit. The mean cancels; command-only."""
        return (self.coords(a_b) - self.coords(a_a)) @ self.basis

    # -- the paper's s^{-1}: nearest point on the manifold ---------------------------------------
    def invert(self, profile, a_init=None, grid: int = 41, refine: int = 3) -> np.ndarray:
        """``s^{-1}(h)`` by orthogonal projection: the concept point whose manifold image is closest.

        Search is in the PCA subspace (the paper inverts inside the same 64-dim space it fitted in),
        which is also what makes it survivable here: the raw profile is dominated by per-scene
        appearance, and the accel PCA subspace is where the concept actually varies.

        Coarse grid over the box spanned by the training centres, then ``refine`` local zoom-ins.
        """
        y = (np.asarray(profile, dtype=np.float64) - self.mean) @ self.basis.T   # (k,)
        C = self.centers * self.zscale
        lo, hi = C.min(axis=0), C.max(axis=0)
        if a_init is not None:
            a0 = np.asarray(a_init, dtype=np.float64).reshape(2)
            span = 0.5 * (hi - lo)
            lo, hi = a0 - span, a0 + span
        best = None
        for _ in range(max(1, refine)):
            gx = np.linspace(lo[0], hi[0], grid)
            gy = np.linspace(lo[1], hi[1], grid)
            G = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
            P = tps_design(G / self.zscale, self.centers, self.linear_only) @ self.coef   # (N,k)
            d = np.linalg.norm(P - y[None, :], axis=1)
            j = int(d.argmin())
            best = G[j]
            sx = (gx[1] - gx[0]); sy = (gy[1] - gy[0])
            lo = best - np.array([sx, sy]); hi = best + np.array([sx, sy])
        return np.asarray(best, dtype=np.float64)


def fit_manifold(Z: np.ndarray, Y: np.ndarray, n_centers: int = 64, k_pca: int = 64,
                 ridge: float = 1e-3, linear_only: bool = False, seed: int = 0) -> ConceptManifold:
    """Fit ``s`` from concept points ``Z (N, 2)`` to profiles ``Y (N, D)``.

    ``Y`` is expected SCENE-CENTRED (appearance already removed at fit time). The concept is scaled
    to unit RMS so ``ridge`` means the same thing across layers and datasets, and the thin-plate
    kernel is evaluated at a sane radius.

    Ridge is applied to the RBF block only; the affine block ``[1, z_x, z_y]`` is left unpenalized,
    which is the usual smoothing-spline convention and makes ``linear_only`` the exact zero-curvature
    limit of the same estimator rather than a differently-regularized model.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    zscale = float(np.sqrt((Z ** 2).sum(axis=1).mean())) or 1.0
    Zs = Z / zscale
    centers = np.zeros((0, 2)) if linear_only else kmeans_centers(Zs, n_centers, seed=seed)

    mean = Y.mean(axis=0)
    Yc = Y - mean
    basis, _ = randomized_pca(Yc, k_pca, seed=seed)          # (k, D)
    coords = Yc @ basis.T                                     # (N, k)

    A = tps_design(Zs, centers, linear_only)                  # (N, M+3)
    M = A.shape[1] - 3
    pen = np.zeros(A.shape[1])
    pen[:M] = ridge
    G = A.T @ A + np.diag(pen) * max(1.0, len(Z))
    coef = np.linalg.solve(G, A.T @ coords)                   # (M+3, k)
    return ConceptManifold(centers, coef, basis, mean, zscale, linear_only)


def project_block(U: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``U (k, Dfull) @ X (Dfull, n) -> (n, k)`` — block projection onto a big saved basis.

    Batched on purpose: ``U`` is ~1 GB per layer, so projecting one clip at a time is bound by
    re-streaming the basis. A block of one scene (8 clips) amortizes that read 8x.
    """
    return (np.asarray(U, dtype=np.float32) @ np.asarray(X, dtype=np.float32)).T.astype(np.float64)


def save_manifold(mf: ConceptManifold, path_prefix) -> None:
    from pathlib import Path
    p = Path(str(path_prefix))
    np.savez(p.with_suffix(".npz"), centers=mf.centers, coef=mf.coef,
             basis=mf.basis.astype(np.float32), mean=mf.mean.astype(np.float32),
             zscale=np.array([mf.zscale]), linear_only=np.array([int(mf.linear_only)]))


def load_manifold(path_prefix) -> ConceptManifold:
    from pathlib import Path
    d = np.load(Path(str(path_prefix)).with_suffix(".npz"))
    return ConceptManifold(d["centers"], d["coef"], d["basis"].astype(np.float64),
                           d["mean"].astype(np.float64), float(d["zscale"][0]),
                           bool(int(d["linear_only"][0])))


# ----------------------------------------------------------------------------------------------------
# spatial placement of a profile edit
# ----------------------------------------------------------------------------------------------------
def place_profile(profile_td: np.ndarray, mask_thw: np.ndarray, grid) -> np.ndarray:
    """Profile ``(T, D)`` x soft mask ``(T, H, W)`` -> flat ``(T*H*W*D,)``, ENERGY-MATCHED to broadcast.

    ``spline_ops.broadcast_profile`` puts the same vector on every spatial token; this puts it where a
    mask says the content is. The mask is renormalized to mean 1 over the spatial tokens of each time
    step, so ``place_profile`` and ``broadcast_profile`` deposit the same total edit energy and any
    difference between them is PLACEMENT, not magnitude -- which matters because the global gain is
    calibrated per arm and would otherwise absorb the difference.
    """
    T, H, W = grid
    m = np.asarray(mask_thw, dtype=np.float64)
    denom = m.mean(axis=(1, 2), keepdims=True)
    m = m / np.where(denom > 1e-12, denom, 1.0)
    P = np.asarray(profile_td, dtype=np.float64)
    return (m[:, :, :, None] * P[:, None, None, :]).reshape(-1)


def accel_trajectory(pos0: np.ndarray, v0: np.ndarray, accel: np.ndarray,
                     n_frames: int) -> np.ndarray:
    """Constant-acceleration roll-out ``pos[f] = pos0 + v0*f + a*f*(f-1)/2``, clamped to [0, 1].

    The ``f*(f-1)/2`` (not ``f^2/2``) matches how the generator integrates -- the same discrepancy that
    was already found and fixed once in the TTO anchor. Command-only: ``pos0`` and ``v0`` are observables
    of the ANCHOR clip and ``accel`` is the target command, so no part of ``H_b`` is read.
    """
    f = np.arange(n_frames, dtype=np.float64).reshape(-1, 1)
    return np.clip(np.asarray(pos0).reshape(1, 2) + f * np.asarray(v0).reshape(1, 2)
                   + (f * (f - 1) / 2.0) * np.asarray(accel).reshape(1, 2), 0.0, 1.0)
