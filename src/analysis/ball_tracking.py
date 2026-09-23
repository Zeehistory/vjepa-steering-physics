"""Measure the ball's motion directly from rendered/decoded pixels (Step 2 velocity verification).

The clean moving-ball dataset is a dark disk on a white background, so we can recover the ball's
per-frame center by an intensity-weighted centroid of the *dark* pixels — no learned tracker needed.
From the centroid track we read off the empirical **velocity** (mean inter-frame displacement) and
**speed**. This is the objective, pixel-level evidence for velocity steering: after we add ``alpha*d_v``
to the latents and decode, does the *decoded ball actually move faster/slower/in a new direction*?

These functions take ``(T, C, H, W)`` float frames in ``[0, 1]`` (the decoder's output) and return
normalized-coordinate quantities (units of image width per frame), matching the dataset's state units.
"""

from __future__ import annotations

import numpy as np
import torch


def ball_centroids(frames: torch.Tensor, darkness_thresh: float = 0.5) -> np.ndarray:
    """Per-frame ball center ``(T, 2)`` in normalized [0,1] coords (x, y), NaN if the ball is absent.

    Works on a dark-ball / light-background scene: weight each pixel by how *dark* it is relative to the
    background and take the weighted centroid. Frames with negligible dark mass (e.g. the ball fully
    occluded) yield NaN so callers can skip them.
    """
    x = frames.detach().cpu().float()
    if x.dim() != 4:
        raise ValueError(f"expected (T,C,H,W), got {tuple(x.shape)}")
    t, _c, h, w = x.shape
    gray = x.mean(dim=1)  # (T, H, W), ~1 background, ~0 ball
    dark = (1.0 - gray).clamp(min=0.0)
    dark = torch.where(dark > (1.0 - darkness_thresh), dark, torch.zeros_like(dark))
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    mass = dark.sum(dim=(1, 2))  # (T,)
    cx = (dark * xs).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    cy = (dark * ys).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    out = torch.stack([cx, cy], dim=1).numpy()
    out[mass.numpy() < 1e-3] = np.nan
    return out


def ball_centroids_marker_invariant(
    frames: torch.Tensor, darkness_thresh: float = 0.5, red_thresh: float = 0.12,
) -> np.ndarray:
    """Ball center ``(T, 2)`` that a rotating marker cannot move. Use this on MARKED balls.

    ``ball_centroids`` weights each pixel by its darkness. That is exact for a uniformly dark ball and
    biased for ``spin_ball3d``, whose amber marker (1.00, 0.75, 0.10) has luma 0.617 -> darkness 0.383,
    against the ball body's ~0.90. The marker therefore sits INSIDE the mask (0.383 clears no threshold
    it is tested against, but it is non-zero mass) carrying a different weight from the body it covers,
    so the weighted centroid is pulled toward or away from the marker as the marker orbits. The readout
    then oscillates at the SPIN frequency and contaminates the measured velocity -- which is fatal here,
    because spin-to-velocity crosstalk is the exact quantity this study reports.

    ``rotor_orientation`` already excludes red from its body mask for precisely this reason; that
    exclusion was never applied to the velocity path.

    The fix is a BINARY mask over "ball material" -- dark body OR coloured marker, weighted equally --
    so the centroid is the geometric center of the disc regardless of where on it the marker sits.
    Equal weighting is the essential part; simply dropping marker pixels would leave a rotating HOLE
    and reintroduce the same wobble.
    """
    x = frames.detach().cpu().float()
    if x.dim() != 4:
        raise ValueError(f"expected (T,C,H,W), got {tuple(x.shape)}")
    _t, c, h, w = x.shape
    if c < 3:
        raise ValueError("marker-invariant tracking needs RGB frames")
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    gray = x.mean(dim=1)
    body = (1.0 - gray) > (1.0 - darkness_thresh)
    marker = (r - torch.maximum(g, b)).clamp(min=0.0) > red_thresh
    mask = (body | marker).float()
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    mass = mask.sum(dim=(1, 2))
    cx = (mask * xs).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    cy = (mask * ys).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    out = torch.stack([cx, cy], dim=1).numpy()
    out[mass.numpy() < 1.0] = np.nan
    return out


def measured_velocity_marker_invariant(
    frames: torch.Tensor, darkness_thresh: float = 0.5, red_thresh: float = 0.12,
) -> dict[str, float]:
    """:func:`measured_velocity` on the marker-invariant centroid. Same return contract."""
    c = ball_centroids_marker_invariant(frames, darkness_thresh, red_thresh)
    disp = np.diff(c, axis=0)
    valid = ~np.isnan(disp).any(axis=1)
    if valid.sum() == 0:
        return {"vel_x": float("nan"), "vel_y": float("nan"), "speed": float("nan"), "n_valid": 0}
    v = disp[valid].mean(axis=0)
    return {"vel_x": float(v[0]), "vel_y": float(v[1]),
            "speed": float(np.linalg.norm(v)), "n_valid": int(valid.sum())}


def measured_velocity(frames: torch.Tensor, darkness_thresh: float = 0.5) -> dict[str, float]:
    """Empirical velocity from the decoded frames.

    Returns ``{vel_x, vel_y, speed, n_valid}`` where velocity is the mean inter-frame displacement of
    the centroid (normalized units per frame) over frames where the ball is visible. ``speed`` is its
    magnitude. ``n_valid`` is how many consecutive-visible frame pairs contributed.
    """
    c = ball_centroids(frames, darkness_thresh)
    disp = np.diff(c, axis=0)  # (T-1, 2)
    valid = ~np.isnan(disp).any(axis=1)
    if valid.sum() == 0:
        return {"vel_x": float("nan"), "vel_y": float("nan"), "speed": float("nan"), "n_valid": 0}
    v = disp[valid].mean(axis=0)
    return {
        "vel_x": float(v[0]), "vel_y": float(v[1]),
        "speed": float(np.linalg.norm(v)), "n_valid": int(valid.sum()),
    }


def rotor_orientation(frames: torch.Tensor, darkness_thresh: float = 0.5,
                      red_thresh: float = 0.25) -> np.ndarray:
    """Per-frame orientation ``phi`` (radians, (T,)) of the spinning bar+marker, NaN where unreadable.

    The rotation object is a dark bar with a fixed RED end-marker. We recover the orientation without any
    learned tracker, purely from pixels:
      * BODY mask = dark pixels that are NOT red -> its weighted centroid is the rotation CENTRE (the bar is
        symmetric about the centre, so this is orientation-invariant);
      * MARKER mask = "red" pixels (``R - max(G, B) > red_thresh``) -> its weighted centroid is the leading
        end;
      * ``phi = atan2(marker_y - centre_y, marker_x - centre_x)`` in the same x-right / y-down convention the
        renderer uses. Frames whose marker or body mass is negligible yield NaN.
    """
    x = frames.detach().cpu().float()
    if x.dim() != 4:
        raise ValueError(f"expected (T,C,H,W), got {tuple(x.shape)}")
    t, c, h, w = x.shape
    if c < 3:
        raise ValueError("rotor_orientation needs RGB frames (marker is colour-separated)")
    r, g, b = x[:, 0], x[:, 1], x[:, 2]                     # (T,H,W) each
    gray = x.mean(dim=1)                                     # ~1 background, <1 object
    dark = (1.0 - gray).clamp(min=0.0)
    dark = torch.where(dark > (1.0 - darkness_thresh), dark, torch.zeros_like(dark))
    redness = (r - torch.maximum(g, b)).clamp(min=0.0)       # marker >> body/bg
    marker_w = torch.where(redness > red_thresh, redness, torch.zeros_like(redness))
    body_w = torch.where(redness <= red_thresh, dark, torch.zeros_like(dark))
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)

    def _centroid(weight):
        m = weight.sum(dim=(1, 2))                           # (T,)
        cx = (weight * xs).sum(dim=(1, 2)) / m.clamp(min=1e-6)
        cy = (weight * ys).sum(dim=(1, 2)) / m.clamp(min=1e-6)
        return cx, cy, m

    bx, by, bm = _centroid(body_w)
    mx, my, mm = _centroid(marker_w)
    phi = torch.atan2(my - by, mx - bx).numpy()
    bad = (bm.numpy() < 1e-3) | (mm.numpy() < 1e-5)
    phi[bad] = np.nan
    return phi


def measured_angvel(frames: torch.Tensor, darkness_thresh: float = 0.5,
                    red_thresh: float = 0.25) -> dict[str, float]:
    """Empirical ANGULAR velocity ``omega`` (rad/frame) from the decoded frames.

    The angular-velocity analog of :func:`measured_velocity`. Reads the marker orientation ``phi(t)`` every
    frame (:func:`rotor_orientation`), unwraps it (per-frame rotation is well below pi, so unwrapping is
    unambiguous), and fits the line ``phi(t) = phi0 + omega*t`` by least squares over the readable frames;
    ``omega`` (the slope) is the signed angular velocity, directly comparable to the GT ``obj0_omega`` and to
    a commanded angular velocity. Returns ``{omega, phi0, resid, n_valid}``; needs >= 3 readable frames.
    """
    phi = rotor_orientation(frames, darkness_thresh, red_thresh)
    t = np.arange(phi.shape[0], dtype=np.float64)
    valid = ~np.isnan(phi)
    if valid.sum() < 3:
        return {"omega": float("nan"), "phi0": float("nan"), "resid": float("nan"),
                "n_valid": int(valid.sum())}
    tt = t[valid]
    pu = np.unwrap(phi[valid])                               # unwrap only over readable frames
    A = np.stack([np.ones_like(tt), tt], axis=1)            # (n, 2)
    coef, *_ = np.linalg.lstsq(A, pu, rcond=None)
    phi0, omega = float(coef[0]), float(coef[1])
    resid = float(np.sqrt(np.mean((A @ coef - pu) ** 2)))
    return {"omega": omega, "phi0": phi0, "resid": resid, "n_valid": int(valid.sum())}


def measured_angaccel(frames: torch.Tensor, darkness_thresh: float = 0.5,
                      red_thresh: float = 0.25) -> dict[str, float]:
    """Empirical constant ANGULAR acceleration ``alpha`` (rad/frame^2) from the decoded frames.

    The rotational analog of :func:`measured_acceleration`, and the second-order sibling of
    :func:`measured_angvel`: read the marker orientation ``phi(t)`` every frame
    (:func:`rotor_orientation`), unwrap it, then fit the PARABOLA ``phi(t) = c0 + c1*t + c2*t^2`` by least
    squares over the readable frames. The rotor roll-out is exactly ``theta(t) = theta0 + omega0*t +
    0.5*alpha*t^2``, so ``c2 = alpha/2`` and the recovered angular acceleration is ``alpha = 2*c2`` --
    directly comparable to the GT ``obj0_alpha`` column and to a commanded angular acceleration. ``c1`` is
    the recovered INITIAL spin rate ``omega0`` (returned as ``omega0`` for the anti-gaming check that a
    steer changed the curvature rather than the base rate).

    Needs >= 4 readable frames: three identify a parabola exactly, so a fourth is the minimum that leaves a
    residual capable of exposing a bad fit. Returns ``{alpha, omega0, phi0, resid, n_valid}``.
    """
    phi = rotor_orientation(frames, darkness_thresh, red_thresh)
    t = np.arange(phi.shape[0], dtype=np.float64)
    valid = ~np.isnan(phi)
    if valid.sum() < 4:
        return {"alpha": float("nan"), "omega0": float("nan"), "phi0": float("nan"),
                "resid": float("nan"), "n_valid": int(valid.sum())}
    tt = t[valid]
    pu = np.unwrap(phi[valid])                               # unwrap only over readable frames
    A = np.stack([np.ones_like(tt), tt, tt ** 2], axis=1)    # (n, 3)
    coef, *_ = np.linalg.lstsq(A, pu, rcond=None)
    resid = float(np.sqrt(np.mean((A @ coef - pu) ** 2)))
    return {"alpha": float(2.0 * coef[2]), "omega0": float(coef[1]), "phi0": float(coef[0]),
            "resid": resid, "n_valid": int(valid.sum())}


def measured_acceleration(frames: torch.Tensor, darkness_thresh: float = 0.5) -> dict[str, float]:
    """Empirical constant 2D acceleration from the decoded frames (Step 2 "Bravo").

    The acceleration analog of :func:`measured_velocity`: track the ball's per-frame centroid, then fit a
    quadratic ``pos(t) = c0 + c1*t + c2*t^2`` per axis by least squares over the visible frames. The
    discrete constant-acceleration roll-out (``pos_t = pos0 + (v0 - a/2)*t + (a/2)*t^2``) makes ``c2 =
    a/2``, so the recovered acceleration is ``a = 2*c2`` — directly comparable to the GT
    ``(obj0_acc_x, obj0_acc_y)`` columns and to a commanded acceleration. Returns
    ``{acc_x, acc_y, acc_mag, n_valid}``; needs >= 3 visible frames to identify the curvature.
    """
    c = ball_centroids(frames, darkness_thresh)  # (T, 2)
    t = np.arange(c.shape[0], dtype=np.float64)
    valid = ~np.isnan(c).any(axis=1)
    if valid.sum() < 3:
        return {"acc_x": float("nan"), "acc_y": float("nan"), "acc_mag": float("nan"),
                "n_valid": int(valid.sum())}
    tt = t[valid]
    A = np.stack([np.ones_like(tt), tt, tt ** 2], axis=1)  # (n, 3)
    cx, _, _, _ = np.linalg.lstsq(A, c[valid, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, c[valid, 1], rcond=None)
    ax, ay = 2.0 * cx[2], 2.0 * cy[2]
    return {"acc_x": float(ax), "acc_y": float(ay),
            "acc_mag": float(np.hypot(ax, ay)), "n_valid": int(valid.sum())}
