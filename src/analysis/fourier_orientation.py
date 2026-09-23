"""The FOURIER-IN-ORIENTATION latent operator: a command-only model of the SO(2) group action.

This is the machinery behind the solved angular-velocity steer (held-out rho=0.94), factored out of
``experiments/threads/angular-velocity/05_steering/steer_angvel_fourier.py`` so that the angular-ACCELERATION experiment can apply *the same fitted
operator* rather than a lookalike reimplementation -- which is what makes the zero-shot transfer claim a
clean test.

THE IDEA. A rotation acts on the latent NONLINEARLY in the angle: no linear map in ``delta omega`` (in
Cartesian or in polar) can express it, which is why every linear attempt failed (cart-axis rho 0.005, polar
cmd-U8 0.14, rotation-transport 0.0). The basis that LINEARIZES the action is Fourier in the orientation.
After center-canonicalization each latent token value is a periodic function of the object's orientation, so

    H_canon[t, cell, d] ~= sum_k C[t, cell, d, k] * phi_k(theta(t))

with ``phi = [1, cos theta, sin theta, cos 2theta, sin 2theta, ...]``. The bar's pi-symmetry lives in the
EVEN harmonics and the marker's 2pi cue in the FUNDAMENTAL. ``C`` is a tiny per-token ridge fit on real
clips. To steer, RE-EVALUATE the model at the target orientation and difference:

    dH_canon[t] = C[t] . (phi(theta_b(t)) - phi(theta_a(t)))

The DC/appearance column cancels in the difference, leaving a pure orientation edit.

WHY IT SHOULD GENERALIZE ACROSS KINEMATIC ORDER. ``C`` is indexed by ORIENTATION, never by the command.
Nothing in the fit knows whether ``theta(t)`` came from a constant spin, a ramp, or an arbitrary profile --
the command enters only through the trajectory ``theta(t)`` at evaluation time. So an operator fit purely on
constant-omega clips (``theta = theta0 + omega*t``) should steer angular ACCELERATION (``theta = theta0 +
omega0*t + alpha/2*t^2``) with NO refit. That is a falsifiable prediction about the operator being a model
of the group action rather than of the training command, and it is what :mod:`scripts.steer_angaccel_fourier`
tests.

THE RATE CAVEAT. A V-JEPA temporal token pools several frames, so the latent there plausibly encodes not
only the orientation theta(t) but the RATE omega(t) at which it is sweeping (motion blur / local dynamics).
The orientation-only basis ignores rate; it nonetheless carried the angular-velocity steer, so orientation
dominates. :func:`design_matrix` therefore supports both an ``orientation`` basis and an
``orientation_rate`` basis (which adds rate-interacted harmonics), letting the experiment measure how much
of the residual the rate term explains instead of assuming an answer.
"""
from __future__ import annotations

import numpy as np

BASES = ("orientation", "orientation_rate")


def to_grid(arr, grid):
    """Flat latent -> ``(T, H, W, D)``."""
    T, H, W = grid
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(T, H, W, a.size // (T * H * W))


def canon_roll(x, cen, grid, inverse=False):
    """Roll ``(T,H,W,D)`` so the rotation centre cell -> grid middle (or back).

    An exact integer roll, hence invertible and information-preserving. This removes the per-scene centre
    confound: without it, two scenes rotating about different points place the same orientation change at
    different grid cells, so no shared operator exists (the original "88 deg, no shared axis" reading was
    largely this artifact).
    """
    _, H, W = grid
    w = int(np.clip(round(cen[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(cen[1] * (H - 1)), 0, H - 1))
    dh, dw = (H // 2 - h), (W // 2 - w)
    if inverse:
        dh, dw = -dh, -dw
    return np.roll(x, shift=(dh, dw), axis=(1, 2))


def phi_feats(theta, order, harmonics="all"):
    """Fourier design row(s) for orientation ``theta`` -> ``[1, cos, sin, cos2, sin2, ...]``.

    ``harmonics`` selects which harmonics may be used, for the mechanistic ablation:
      * ``"all"``  -- every harmonic k = 1..order (the default operator);
      * ``"even"`` -- only k even, the bar's pi-symmetric content (a bar at theta and theta+pi renders
                      identically apart from the marker);
      * ``"odd"``  -- only k odd, which carries the marker's 2pi (360 deg-unambiguous) cue.
    The DC column is always present; it cancels in the steering difference regardless.
    """
    theta = np.asarray(theta, dtype=np.float64)
    out = [np.ones_like(theta)]
    for k in range(1, order + 1):
        if harmonics == "even" and k % 2 != 0:
            continue
        if harmonics == "odd" and k % 2 == 0:
            continue
        out.append(np.cos(k * theta))
        out.append(np.sin(k * theta))
    return np.stack(out, -1)


def n_features(order, harmonics="all", basis="orientation"):
    """Width of the design row produced by :func:`design_matrix` for these settings."""
    p = phi_feats(np.zeros(1), order, harmonics).shape[-1]
    return p * (2 if basis == "orientation_rate" else 1)


def design_matrix(theta, omega=None, order=4, harmonics="all", basis="orientation"):
    """Design row(s) for the latent model at orientation ``theta`` (and rate ``omega``).

    ``orientation``      -> ``phi(theta)``: the latent as a function of pose alone.
    ``orientation_rate`` -> ``[phi(theta), omega * phi(theta)]``: pose plus rate-interacted harmonics, i.e.
    a first-order expansion of the latent in the local kinematic STATE ``(theta, omega)``. Fit on
    constant-spin clips (where ``omega`` is constant per clip but varies across clips, so both blocks are
    identifiable), this can then be evaluated along a novel ``omega(t)`` ramp.
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    P = phi_feats(theta, order, harmonics)
    if basis == "orientation":
        return P
    if omega is None:
        raise ValueError("basis='orientation_rate' requires omega")
    w = np.asarray(omega, dtype=np.float64)[..., None]
    return np.concatenate([P, w * P], axis=-1)


def theta_of(theta0, omega0, alpha, tau):
    """Orientation trajectory ``theta(t) = theta0 + omega0*tau + 0.5*alpha*tau^2`` at token times ``tau``.

    The single point where a command enters the operator. ``alpha = 0`` recovers the constant-spin roll-out,
    so the angular-velocity and angular-acceleration experiments differ ONLY in this trajectory.
    """
    tau = np.asarray(tau, dtype=np.float64)
    return theta0 + omega0 * tau + 0.5 * alpha * tau * tau


def omega_of(omega0, alpha, tau):
    """Instantaneous rate ``omega(t) = omega0 + alpha*tau`` at token times ``tau``."""
    tau = np.asarray(tau, dtype=np.float64)
    return omega0 + alpha * tau


def fit_operator(PHI, Ys, layers, ridge):
    """Per (layer, temporal token) ridge: ``C[t] = (X'X + lam I)^-1 X'Y``.

    ``PHI`` is ``(N, T, P)`` design rows, ``Ys[L]`` is ``(N, T, M)`` center-canonicalized latents flattened
    over ``(H, W, D)``. Returns ``{L: C (T, P, M)}``. Solved independently per token because a V-JEPA
    temporal token has its own receptive field over the clip.
    """
    N, T, P = PHI.shape
    C = {}
    for L in layers:
        Y = Ys[L]
        M = Y.shape[2]
        CL = np.empty((T, P, M), dtype=np.float32)
        for t in range(T):
            X = PHI[:, t, :]
            A = X.T @ X + ridge * np.eye(P)
            B = X.T @ Y[:, t, :]
            CL[t] = np.linalg.solve(A, B).astype(np.float32)
        C[L] = CL
    return C


def predict_dH(C, row_a, row_b, layers, grid):
    """Synthesize the edit ``dH_canon[t] = C[t] . (row_b[t] - row_a[t])`` -> ``{L: (T,H,W,D)}``.

    ``row_a`` / ``row_b`` are ``(T, P)`` design rows for the source and target commands. Because the model
    is differenced, the DC/appearance column drops out and what remains is a pure kinematic edit.
    """
    T, H, W = grid
    df = np.asarray(row_b, dtype=np.float64) - np.asarray(row_a, dtype=np.float64)
    out = {}
    for L in layers:
        CL = C[L]
        M = CL.shape[2]
        dH = np.einsum("tp,tpm->tm", df, CL).astype(np.float32)
        out[L] = dH.reshape(T, H, W, M // (H * W))
    return out


def cosine(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
