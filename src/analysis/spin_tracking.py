"""Measure a 3D ball's SPIN directly from rendered/decoded pixels (the crosstalk experiment).

The translation half of :mod:`~src.data.spin_ball3d` is already covered:
:func:`src.analysis.ball_tracking.measured_velocity` reads the ball's image-plane velocity off a
darkness centroid, and the dataset's ``obj0_vel_x/y`` columns are defined to be exactly that quantity.
This module supplies the rotational half — recovering the constant world spin rate ``omega`` (rad/frame
about the vertical axis) from the bright marker, with no learned tracker and no ground truth.

**Why it is not just ``atan2`` of the marker offset.** The marker sits at a fixed polar angle from the
ball's +z pole, so as the ball spins its centre traces a HORIZONTAL CIRCLE in the world. Through a
camera at 62 deg elevation that circle images as an ELLIPSE whose centre is *offset from the ball's
image centre* (by the projection of the marker's constant height above the ball centre). Taking
``atan2`` of the raw offset therefore gives an angle that is neither the world azimuth nor even a
monotone rescaling of it near the ellipse's minor axis — it produces a spurious 2/rev ripple, and a
linear fit to it returns a biased ``omega`` whose bias depends on ``phi0``. Which would be a
*phase-dependent* error, i.e. exactly the kind of artefact that could masquerade as spin<-velocity
crosstalk.

**What this module does instead.** The ball is known to sit on the tabletop, so its image centre
determines its world position uniquely (ray / plane intersection). At that world position the marker's
locus is, to first order in (marker offset / camera distance),

    d(phi) = A cos(phi) + B sin(phi) + C

with ``A``, ``B``, ``C`` 2-vectors computed exactly by projecting three known world points. Inverting
the 2x2 ``M = [A B]`` recovers ``cos phi, sin phi`` and hence the true world azimuth; unwrapping and
least-squares-fitting ``phi(t) = phi0 + omega*t`` gives ``omega``. Everything is analytic from the
scene's fixed camera — see :func:`ellipse_basis`.

The camera model is re-derived here from the scene constants rather than queried from MuJoCo, so
analysis runs on CPU nodes with no EGL/GL available. ``tests/test_spin_ball3d.py`` pins it against
:meth:`src.data.spin_ball3d.SpinBall3D.project` to machine precision.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.spin_ball3d import (
    BALL_R,
    CAM_DIST,
    CAM_ELEV,
    CAM_FOVY,
    MARKER_OFFSET,
    MARKER_POLAR,
    TABLE_H,
)

BALL_PLANE_Z = TABLE_H + BALL_R      # the height the ball centre lives at, for the whole clip

# Default marker mask threshold on the (R - B) channel difference. The marker is (1.0, 0.75, 0.10)
# -> 0.90; the wood table is the next-warmest thing in frame at 0.18, and the ball/walls/floor are all
# slightly NEGATIVE. 0.30 sits far above the wood and far below the marker, and because the mask is a
# soft weight (clamp(R - B - thresh, 0)) rather than a hard cut, it degrades gracefully when the
# decoder blurs the marker into the dark ball beneath it.
MARKER_RB_THRESH = 0.30


# -- camera ----------------------------------------------------------------------------------------
def _camera() -> tuple[np.ndarray, np.ndarray, float]:
    """``(cam_pos, cam_mat, focal)`` for the fixed scene camera, matching the MJCF exactly.

    MJCF: ``pos="0 -D*cos(th) TABLE_H + D*sin(th)"``, ``xyaxes="1 0 0  0 sin(th) cos(th)"``; the camera
    z axis is ``x_axis x y_axis`` and the camera looks down its own -z.
    """
    th = np.radians(CAM_ELEV)
    cam_pos = np.array([0.0, -CAM_DIST * np.cos(th), TABLE_H + CAM_DIST * np.sin(th)])
    x_ax = np.array([1.0, 0.0, 0.0])
    y_ax = np.array([0.0, np.sin(th), np.cos(th)])
    z_ax = np.cross(x_ax, y_ax)                       # (0, -cos th, sin th)
    cam_mat = np.stack([x_ax, y_ax, z_ax], axis=1)    # columns are the camera axes
    focal = 1.0 / np.tan(np.radians(CAM_FOVY) / 2)
    return cam_pos, cam_mat, focal


def project(points: np.ndarray) -> np.ndarray:
    """World ``(...,3)`` -> normalized image ``(...,2)`` in [0,1] (x right, y DOWN)."""
    cam_pos, cam_mat, f = _camera()
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    pc = (p - cam_pos) @ cam_mat
    xn = f * pc[:, 0] / (-pc[:, 2])
    yn = f * pc[:, 1] / (-pc[:, 2])
    return np.stack([(xn + 1) / 2, (1 - yn) / 2], axis=1)


def unproject_to_ball_plane(img_xy: np.ndarray) -> np.ndarray:
    """Normalized image ``(...,2)`` -> the world point on ``z = BALL_PLANE_Z`` it images from.

    The ball centre is constrained to that plane for the whole clip (it slides, it never leaves the
    tabletop), so a single image point determines it. NaN image points map to NaN.
    """
    cam_pos, cam_mat, f = _camera()
    uv = np.atleast_2d(np.asarray(img_xy, dtype=np.float64))
    xn = uv[:, 0] * 2.0 - 1.0
    yn = 1.0 - uv[:, 1] * 2.0
    # camera-frame ray direction: a point at camera depth -zc has (xn*(-zc)/f, yn*(-zc)/f, zc)
    d_cam = np.stack([xn / f, yn / f, -np.ones_like(xn)], axis=1)
    d_world = d_cam @ cam_mat.T                         # cam_mat columns are axes -> transpose maps back
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (BALL_PLANE_Z - cam_pos[2]) / d_world[:, 2]
    return cam_pos[None, :] + t[:, None] * d_world


def ellipse_basis(world_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(A, B, C)`` of ``d(phi) = A cos phi + B sin phi + C`` for a ball centred at ``world_c``.

    Computed by exact projection of three known world points, so ``A``/``B``/``C`` absorb the local
    perspective scale and the ellipse's tilt without any small-angle assumption about the camera; the
    only approximation left is that a general ``phi`` interpolates sinusoidally between them, which is
    first-order exact in (marker offset / camera distance) ~ 0.03.
    """
    c = np.asarray(world_c, dtype=np.float64).reshape(3)
    s, cz = np.sin(MARKER_POLAR), np.cos(MARKER_POLAR)
    p0 = project(c)[0]
    pC = project(c + np.array([0.0, 0.0, MARKER_OFFSET * cz]))[0]
    pA = project(c + np.array([MARKER_OFFSET * s, 0.0, MARKER_OFFSET * cz]))[0]
    pB = project(c + np.array([0.0, MARKER_OFFSET * s, MARKER_OFFSET * cz]))[0]
    C = pC - p0
    return pA - p0 - C, pB - p0 - C, C


# -- pixel readout ---------------------------------------------------------------------------------
def marker_centroids(frames: torch.Tensor, rb_thresh: float = MARKER_RB_THRESH) -> np.ndarray:
    """Per-frame marker centre ``(T, 2)`` in normalized [0,1] coords, NaN where the marker is absent.

    Soft ``R - B`` weighting: the marker is the only strongly warm object in the scene (see
    :data:`MARKER_RB_THRESH`). Returns NaN for frames whose warm mass is negligible, so a decode that
    fails to render the marker at all is reported as unreadable rather than silently centred.
    """
    x = frames.detach().cpu().float()
    if x.dim() != 4:
        raise ValueError(f"expected (T,C,H,W), got {tuple(x.shape)}")
    if x.shape[1] < 3:
        raise ValueError("marker tracking needs RGB frames")
    t, _c, h, w = x.shape
    warm = (x[:, 0] - x[:, 2] - rb_thresh).clamp(min=0.0)      # (T, H, W)
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    mass = warm.sum(dim=(1, 2))
    cx = (warm * xs).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    cy = (warm * ys).sum(dim=(1, 2)) / mass.clamp(min=1e-6)
    out = torch.stack([cx, cy], dim=1).numpy()
    out[mass.numpy() < 1e-4] = np.nan
    return out


def ball_centroids_unoccluded(frames: torch.Tensor, darkness_thresh: float = 0.5,
                              rb_thresh: float = MARKER_RB_THRESH) -> np.ndarray:
    """Ball centre ``(T, 2)`` from a silhouette with the marker's OCCLUSION BITE repaired.

    The plain darkness centroid is not a clean ball centre in this scene. The marker rides outside the
    ball, but its image disk still overlaps the ball's silhouette over much of a revolution and
    sometimes straddles the limb, so it removes a phase-dependent bite of dark pixels and drags the
    centroid through a 1/rev wobble of ~0.7 px. That wobble enters the azimuth directly, so it biases
    ``omega``; and a spin-dependent *position* readout is exactly the kind of instrument artefact this
    experiment must not confuse with a latent-space coupling.

    The repair is geometric, not statistical: the ball images as a disk of an analytically known radius,
    so warm (marker) pixels lying INSIDE that disk are pixels the marker is standing in front of, and
    are restored to the silhouette at the ball's mean darkness. Marker pixels outside the disk are left
    out, since there the marker occludes the table, not the ball. This cuts the wobble to ~0.18 px.
    """
    from .ball_tracking import ball_centroids
    from src.data.spin_ball3d import BALL_R

    x = frames.detach().cpu().float()
    if x.dim() != 4 or x.shape[1] < 3:
        raise ValueError(f"expected RGB (T,C,H,W), got {tuple(x.shape)}")
    t_len, _c, h, w = x.shape
    gray = x.mean(dim=1)
    dark = (1.0 - gray).clamp(min=0.0)
    dark = torch.where(dark > (1.0 - darkness_thresh), dark, torch.zeros_like(dark))
    warm = ((x[:, 0] - x[:, 2] - rb_thresh) > 0).float()

    c0 = ball_centroids(frames, darkness_thresh)
    good = np.isfinite(c0).all(axis=1)
    if not good.any():
        return c0
    # Analytic image radius of the ball at each tracked position (perspective varies across the table).
    world = unproject_to_ball_plane(np.where(good[:, None], c0, 0.0))
    r_img = np.array([np.linalg.norm(project(p + np.array([BALL_R, 0.0, 0.0]))[0] - project(p)[0])
                      for p in world])
    ys = torch.linspace(0, 1, h).view(1, h, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w)
    cx = torch.from_numpy(np.nan_to_num(c0[:, 0])).float().view(t_len, 1, 1)
    cy = torch.from_numpy(np.nan_to_num(c0[:, 1])).float().view(t_len, 1, 1)
    rr = torch.from_numpy(r_img).float().view(t_len, 1, 1) * 1.02   # 2% slack for the soft limb
    inside = (((xs - cx) ** 2 + (ys - cy) ** 2) < rr ** 2).float()
    n_dark = (dark > 0).float().sum(dim=(1, 2)).clamp(min=1.0)
    fill = (dark.sum(dim=(1, 2)) / n_dark).view(t_len, 1, 1)        # ball's own mean darkness
    silhouette = dark + warm * inside * fill

    mass = silhouette.sum(dim=(1, 2))
    out = torch.stack([(silhouette * xs).sum(dim=(1, 2)) / mass.clamp(min=1e-6),
                       (silhouette * ys).sum(dim=(1, 2)) / mass.clamp(min=1e-6)], dim=1).numpy()
    out[(mass.numpy() < 1e-3) | ~good] = np.nan
    return out


def straightened_ball_track(frames: torch.Tensor, darkness_thresh: float = 0.5,
                            rb_thresh: float = MARKER_RB_THRESH) -> np.ndarray:
    """Ball centre ``(T, 2)``: :func:`ball_centroids_unoccluded`, then fitted to a straight line.

    The ball's image track is a straight, constant-rate line (free rigid-body glide, see
    :func:`~src.data.spin_ball3d._scene_xml`), so fitting ``pos(t) = p0 + v t`` per axis and using the
    FITTED positions removes whatever 1/rev residue the occlusion repair leaves, by projecting it onto a
    model the scene already guarantees. No ground truth is consulted.
    """
    c = ball_centroids_unoccluded(frames, darkness_thresh, rb_thresh)
    t = np.arange(len(c), dtype=np.float64)
    valid = np.isfinite(c).all(axis=1)
    if valid.sum() < 3:
        return c
    mat = np.stack([np.ones(valid.sum()), t[valid]], axis=1)
    out = np.full_like(c, np.nan)
    for ax in (0, 1):
        coef, *_ = np.linalg.lstsq(mat, c[valid, ax], rcond=None)
        out[:, ax] = coef[0] + coef[1] * t
    return out


def marker_azimuth(
    frames: torch.Tensor,
    ball_xy: np.ndarray | None = None,
    darkness_thresh: float = 0.5,
    rb_thresh: float = MARKER_RB_THRESH,
) -> np.ndarray:
    """Per-frame WORLD azimuth ``phi`` of the marker ``(T,)``, radians, NaN where unreadable.

    ``ball_xy`` may be supplied (e.g. the ground-truth track) to isolate the spin readout from any
    error in the position readout; by default it is :func:`straightened_ball_track`, i.e. the same
    darkness pixels the velocity metric uses, with the marker's own 1/rev wobble fitted out.
    """
    ball = np.asarray(ball_xy, dtype=np.float64) if ball_xy is not None \
        else straightened_ball_track(frames, darkness_thresh, rb_thresh)
    mark = marker_centroids(frames, rb_thresh)
    out = np.full(len(ball), np.nan)
    world = unproject_to_ball_plane(ball)
    for t in range(len(ball)):
        if not np.isfinite(ball[t]).all() or not np.isfinite(mark[t]).all():
            continue
        A, B, C = ellipse_basis(world[t])
        M = np.stack([A, B], axis=1)                  # 2x2: columns A, B
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        cs = np.linalg.solve(M, mark[t] - ball[t] - C)
        out[t] = float(np.arctan2(cs[1], cs[0]))
    return out


def measured_spin(
    frames: torch.Tensor,
    ball_xy: np.ndarray | None = None,
    darkness_thresh: float = 0.5,
    rb_thresh: float = MARKER_RB_THRESH,
) -> dict[str, float]:
    """Empirical constant spin rate from decoded frames.

    Returns ``{omega, phi0, residual, n_valid}`` with ``omega`` in rad/FRAME (the dataset's
    ``obj0_omega`` units), from an unwrapped least-squares fit ``phi(t) = phi0 + omega*t``.
    ``residual`` is the max absolute fit residual in radians — a healthy clip sits near zero, and a
    large value means the unwrap was ambiguous or the marker readout is unreliable, so callers can
    reject rather than quote a fitted slope that means nothing.
    """
    phi = marker_azimuth(frames, ball_xy, darkness_thresh, rb_thresh)
    valid = np.isfinite(phi)
    n = int(valid.sum())
    if n < 3:
        return {"omega": float("nan"), "phi0": float("nan"), "residual": float("nan"), "n_valid": n}
    t = np.arange(len(phi), dtype=np.float64)[valid]
    # Unwrap over the VALID samples only; a dropped frame otherwise injects a phantom 2pi jump.
    ph = _unwrap_gappy(phi[valid], t)
    slope, icept = np.polyfit(t, ph, 1)
    resid = float(np.max(np.abs(ph - (icept + slope * t))))
    return {"omega": float(slope), "phi0": float(icept), "residual": resid, "n_valid": n}


def _unwrap_gappy(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unwrap ``phi`` sampled at (possibly non-consecutive) times ``t``.

    ``np.unwrap`` assumes unit spacing; across a gap of ``dt`` frames the true step can exceed pi
    legitimately, so the branch is chosen against the *rate* implied by the previous step rather than
    against a fixed pi. Falls back to the plain nearest-branch rule for the first step.
    """
    out = np.array(phi, dtype=np.float64, copy=True)
    if len(out) < 2:
        return out
    prev_rate = 0.0
    for i in range(1, len(out)):
        dt = max(t[i] - t[i - 1], 1e-9)
        expected = out[i - 1] + prev_rate * dt
        k = np.round((expected - out[i]) / (2 * np.pi))
        out[i] += 2 * np.pi * k
        prev_rate = (out[i] - out[i - 1]) / dt
    return out


__all__ = [
    "marker_centroids", "marker_azimuth", "measured_spin", "ellipse_basis",
    "ball_centroids_unoccluded", "straightened_ball_track",
    "project", "unproject_to_ball_plane", "BALL_PLANE_Z", "MARKER_RB_THRESH",
]
