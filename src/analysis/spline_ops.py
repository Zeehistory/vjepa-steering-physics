"""Spline bases over the TEMPORAL token axis, for acceleration steering.

Motivation (Step 2 "Bravo", spline direction 2026-07-27). Every command-only acceleration operator tried
so far synthesizes ONE global edit vector that is applied with the same weight at every temporal token —
a *constant-in-t* edit. But acceleration is by definition a time-VARYING effect, and the curvature
measurement (``outputs/analysis/moving_ball_accel2d_mixed/curvature/curvature_summary.json``) shows the
true edit's magnitude grows monotonically along the clip: ``dH_norm_per_t`` rises 137 -> 789 at L6 and
449 -> 878 at L12 from ``t=0`` to ``t=7``. A constant edit has the wrong SHAPE in time, which is a
plausible reason the linear operators plateau at ~13-15deg while the per-pair oracle reaches 10.6deg.

This module parametrizes the edit as a smooth curve in ``t`` instead:

    Delta R(t) = sum_j  C_j * B_j(t),      B in R^{T x K},  C in R^{K x D}

with ``B`` a clamped uniform B-spline basis over the ``T=8`` temporal tokens and ``K`` control points.
``K`` interpolates between the two regimes that bracket the problem:

    K = 1   constant in t          == the classical "one global steering vector" operator
    K = 2   linear ramp in t
    K = 3,4 the piecewise-linear / low-order curved regime (the useful middle)
    K = T   unconstrained per-token profile  == the existing ``cmd_prof`` operator

so a sweep over ``K`` is a clean ablation of *temporal smoothness alone*: same features, same ridge, same
training pairs, only the number of temporal degrees of freedom changes.

Also provides a uniform Catmull-Rom evaluator, used for the "latent family" arm: a scene's clips form a
sequence of latents ordered by acceleration, and the question is whether that sequence is curved enough
that a spline through it extrapolates to an unseen acceleration better than a straight line does.

Everything here is pure numpy on the spatially-pooled profile ``r(H) in R^{T x D}`` (``T=8``, ``D=1024``),
which is the representation the acceleration probe reads and is spatial-roll invariant.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------------
# clamped uniform B-spline basis (Cox-de Boor)
# ----------------------------------------------------------------------------------------------------
def knot_vector(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped uniform knot vector for ``n_ctrl`` control points of the given degree.

    Length is ``n_ctrl + degree + 1``: ``degree+1`` repeated 0s, evenly spaced interior knots, then
    ``degree+1`` repeated 1s. Clamping makes the curve interpolate its first and last control point,
    which is what we want at the clip boundaries.
    """
    n_interior = n_ctrl - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_ctrl={n_ctrl} too small for degree={degree} (need n_ctrl >= degree+1)")
    interior = [(i + 1) / (n_interior + 1) for i in range(n_interior)]
    return np.asarray([0.0] * (degree + 1) + interior + [1.0] * (degree + 1), dtype=np.float64)


def _basis_at(u: float, n_ctrl: int, degree: int, knots: np.ndarray) -> np.ndarray:
    """Cox-de Boor evaluation of all ``n_ctrl`` basis functions at parameter ``u`` in [0, 1]."""
    u = float(np.clip(u, 0.0, 1.0))
    n_span = len(knots) - 1
    # degree-0 indicator. The half-open convention leaves u == 1 with no live span, so the final
    # non-degenerate span is closed explicitly (else the last control point would never be reached).
    N = np.zeros(n_span, dtype=np.float64)
    for i in range(n_span):
        lo, hi = knots[i], knots[i + 1]
        if (lo <= u < hi) or (u >= 1.0 and hi >= 1.0 and lo < hi):
            N[i] = 1.0
            if u >= 1.0:
                N[:i] = 0.0
                break
    for d in range(1, degree + 1):
        Nd = np.zeros(n_span - d, dtype=np.float64)
        for i in range(n_span - d):
            den1 = knots[i + d] - knots[i]
            den2 = knots[i + d + 1] - knots[i + 1]
            if den1 > 0:
                Nd[i] += (u - knots[i]) / den1 * N[i]
            if den2 > 0:
                Nd[i] += (knots[i + d + 1] - u) / den2 * N[i + 1]
        N = Nd
    return N[:n_ctrl]


def spline_basis(n_t: int, n_ctrl: int, degree: int = 3) -> np.ndarray:
    """Design matrix ``B`` of shape ``(n_t, n_ctrl)`` sampling a clamped B-spline at the token times.

    The ``n_t`` temporal tokens are placed at evenly spaced parameters in ``[0, 1]``. The degree is
    lowered automatically when there are too few control points to support it, so ``n_ctrl=1`` gives a
    constant basis and ``n_ctrl=2`` a linear ramp — the degenerate cases we want as baselines.

    Rows sum to 1 (partition of unity), so a constant profile is always exactly representable and the
    ``n_ctrl=1`` column is literally the "one global vector applied at every token" operator.
    """
    if n_ctrl < 1:
        raise ValueError("n_ctrl must be >= 1")
    deg = int(min(degree, n_ctrl - 1))
    knots = knot_vector(n_ctrl, deg)
    ts = np.linspace(0.0, 1.0, n_t)
    B = np.stack([_basis_at(u, n_ctrl, deg, knots) for u in ts], axis=0)
    # guard against tiny Cox-de Boor round-off breaking partition-of-unity
    rows = B.sum(axis=1, keepdims=True)
    return B / np.where(rows > 0, rows, 1.0)


# ----------------------------------------------------------------------------------------------------
# projection / reconstruction of a temporal profile
# ----------------------------------------------------------------------------------------------------
def project_profile(profile_td: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Least-squares control points ``C (K, D)`` for a profile ``(T, D)`` in the basis ``B (T, K)``."""
    return np.linalg.lstsq(B, np.asarray(profile_td, dtype=np.float64), rcond=None)[0]


def reconstruct_profile(coef_kd: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Control points ``(K, D)`` -> profile ``(T, D)``."""
    return B @ np.asarray(coef_kd, dtype=np.float64)


def smooth_profile(profile_td: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Project a profile onto the spline basis and back — the best ``K``-knot approximation of it."""
    return reconstruct_profile(project_profile(profile_td, B), B)


# ----------------------------------------------------------------------------------------------------
# temporal pooling helpers (match experiments/threads/acceleration/05_steering/steer_accel2d.py conventions exactly)
# ----------------------------------------------------------------------------------------------------
def temporal_profile(flat: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Flat layer tokens ``(T*H*W*D,)`` -> spatially pooled profile ``(T, D)``."""
    T, H, W = grid
    D = flat.size // (T * H * W)
    return np.asarray(flat, dtype=np.float64).reshape(T, H, W, D).mean(axis=(1, 2))


def broadcast_profile(profile_td: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Profile ``(T, D)`` -> flat ``(T*H*W*D,)``, constant across spatial tokens.

    Spatial placement is deliberately uniform: the transport experiment established that *where* the
    edit lands is inert at the decoder (correct trajectory tokens scored no better than a random other
    scene's), so the profile carries the whole signal and broadcasting is the honest parametrization.
    """
    T, H, W = grid
    D = profile_td.shape[1]
    return np.broadcast_to(np.asarray(profile_td, dtype=np.float64)[:, None, None, :],
                           (T, H, W, D)).reshape(-1).copy()


# ----------------------------------------------------------------------------------------------------
# Catmull-Rom, for splines through a family of latents (paper-style interpolation arm)
# ----------------------------------------------------------------------------------------------------
def catmull_rom(points: np.ndarray, s: float) -> np.ndarray:
    """Uniform Catmull-Rom spline through ``points (M, D)``, evaluated at ``s`` in units of index.

    ``s = i`` returns ``points[i]`` exactly (the spline interpolates its control points). Endpoints are
    handled by reflecting a phantom point, so the tangent at the ends is well defined. ``s`` outside
    ``[0, M-1]`` EXTRAPOLATES along the end segment's cubic — which is the point of the acceleration
    arm: we fit the curve on the accelerations we have and ask it for one we have never seen.
    """
    P = np.asarray(points, dtype=np.float64)
    M = P.shape[0]
    if M < 2:
        raise ValueError("need at least 2 control points")
    i = int(np.floor(s))
    i = int(np.clip(i, 0, M - 2))
    u = float(s - i)

    def _pt(j: int) -> np.ndarray:
        if j < 0:
            return 2.0 * P[0] - P[1]          # reflected phantom start
        if j > M - 1:
            return 2.0 * P[M - 1] - P[M - 2]  # reflected phantom end
        return P[j]

    p0, p1, p2, p3 = _pt(i - 1), _pt(i), _pt(i + 1), _pt(i + 2)
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3)


def catmull_rom_closed(points: np.ndarray, s: float) -> np.ndarray:
    """Periodic Catmull-Rom through ``points (M, D)`` treated as a CLOSED loop, evaluated at index ``s``.

    The acceleration family is a loop, not a ramp: within a scene the clips sweep the acceleration
    DIRECTION through a full turn at roughly fixed magnitude, so control point ``M-1`` is adjacent to
    control point ``0``. Neighbours are taken mod ``M``, which removes the endpoint problem entirely —
    there are no ends — and makes the curve smooth across the wrap.
    """
    P = np.asarray(points, dtype=np.float64)
    M = P.shape[0]
    if M < 3:
        raise ValueError("closed Catmull-Rom needs at least 3 control points")
    i = int(np.floor(s))
    u = float(s - i)
    p0, p1, p2, p3 = P[(i - 1) % M], P[i % M], P[(i + 1) % M], P[(i + 2) % M]
    u2, u3 = u * u, u * u * u
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3)


def linear_eval_closed(points: np.ndarray, s: float) -> np.ndarray:
    """Piecewise-linear counterpart of :func:`catmull_rom_closed` — the chord across the same loop."""
    P = np.asarray(points, dtype=np.float64)
    M = P.shape[0]
    i = int(np.floor(s))
    u = float(s - i)
    return P[i % M] + u * (P[(i + 1) % M] - P[i % M])


def angular_index(angles_rad: np.ndarray, query_rad: float) -> float:
    """Index coordinate of ``query_rad`` within control points placed at ``angles_rad`` around a loop.

    Angles are unwrapped into a monotone increasing sequence, then the query is placed by linear
    interpolation between the two bracketing control points (wrapping through the closing segment).
    Returns a float in ``[0, M)`` suitable for :func:`catmull_rom_closed`.
    """
    a = np.mod(np.asarray(angles_rad, dtype=np.float64), 2 * np.pi)
    M = len(a)
    q = float(np.mod(query_rad, 2 * np.pi))
    for i in range(M):
        lo = a[i]
        hi = a[(i + 1) % M]
        span = np.mod(hi - lo, 2 * np.pi)
        off = np.mod(q - lo, 2 * np.pi)
        if span > 1e-12 and off <= span + 1e-12:
            return float(i + off / span)
    return 0.0


def linear_eval(points: np.ndarray, s: float) -> np.ndarray:
    """Piecewise-linear counterpart of :func:`catmull_rom` (same parametrization, straight segments).

    Extrapolates along the first/last segment when ``s`` falls outside ``[0, M-1]``, so it is the exact
    like-for-like control for the spline: identical control points, identical query, no curvature.
    """
    P = np.asarray(points, dtype=np.float64)
    M = P.shape[0]
    i = int(np.clip(int(np.floor(s)), 0, M - 2))
    u = float(s - i)
    return P[i] + u * (P[i + 1] - P[i])


def fit_scalar_parametrization(values: np.ndarray) -> np.ndarray:
    """Map physical scalars (e.g. per-rank acceleration magnitudes) to spline index coordinates.

    Returns the index positions ``0..M-1`` of the SORTED values; querying a new scalar is then a matter
    of :func:`interp_index`. Kept explicit so the parametrization used at fit and query time is the same.
    """
    return np.arange(len(np.asarray(values)), dtype=np.float64)


def interp_index(values_sorted: np.ndarray, query: float) -> float:
    """Index coordinate of ``query`` within monotone ``values_sorted``, linearly extrapolating outside.

    This is what lets the spline be asked for an acceleration that is not one of the observed ranks.
    """
    v = np.asarray(values_sorted, dtype=np.float64)
    M = len(v)
    if M < 2:
        return 0.0
    if query <= v[0]:
        denom = v[1] - v[0]
        return 0.0 if abs(denom) < 1e-12 else float((query - v[0]) / denom)
    if query >= v[-1]:
        denom = v[-1] - v[-2]
        return float(M - 1) if abs(denom) < 1e-12 else float((M - 1) + (query - v[-1]) / denom)
    j = int(np.searchsorted(v, query) - 1)
    j = int(np.clip(j, 0, M - 2))
    denom = v[j + 1] - v[j]
    return float(j) if abs(denom) < 1e-12 else float(j + (query - v[j]) / denom)
