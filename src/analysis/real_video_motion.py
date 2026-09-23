"""Motion measurement for REAL video -- the Physics-IQ replacement for :mod:`ball_tracking`.

``ball_tracking`` thresholds on darkness to find one dark ball on a white background. That is exact
on the sim and meaningless on real footage, where there is no single dark object, the background is
textured, and lighting changes. This module measures the same quantities from real video and
deliberately mirrors ball_tracking's public API -- ``measured_velocity`` / ``measured_acceleration``
/ ``measured_angvel`` over a ``(T,C,H,W)`` tensor returning a dict -- so downstream steering code
works by swapping one import.

TWO INDEPENDENT CHANNELS, because on real video a single tracker cannot be trusted and there is no
ground truth to check it against:

  channel M (mask)  frame-differencing motion mask -> largest connected component -> centroid track.
                    Same idea as the benchmark's own physiq/binary_mask_generator.py.
  channel F (flow)  RAFT optical flow, aggregated inside the motion mask. Also yields angular
                    velocity from the antisymmetric part of an affine fit to the flow field.

Their agreement is the confidence signal (``pseudo_gt_confidence``): the two fail in different ways
(the mask fires on shadows and multi-object scenes; flow degrades on large displacements and
texture-poor regions), so when they agree the measurement is probably real.

ALL measurements are in the encoder's own 256x256 pixel frame, never metric world units: the
Physics-IQ 3840x2160 source is squashed anisotropically (x by 15x, y by 8.4x) to reach 256x256, so
"down" is not 90 degrees and image-plane speed is not proportional to world speed.
"""

from __future__ import annotations

import numpy as np
import torch

_RAFT = None


def _to_numpy(frames: torch.Tensor) -> np.ndarray:
    """(T,C,H,W) float [0,1] -> (T,H,W) float grayscale."""
    x = frames.detach().cpu().float().numpy()
    if x.ndim == 4:
        x = x.transpose(0, 2, 3, 1)
    return x @ np.array([0.299, 0.587, 0.114], dtype=x.dtype) if x.shape[-1] == 3 else x[..., 0]


def video_background(frames_full: np.ndarray | torch.Tensor) -> np.ndarray:
    """Temporal-median background from the WHOLE video. Compute once per clip and reuse."""
    g = _to_numpy(frames_full) if isinstance(frames_full, torch.Tensor) else frames_full
    if g.ndim == 4:                                   # (T,H,W,3) uint8 cache
        g = (g.astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.median(g, axis=0)


def motion_mask(frames: torch.Tensor, thresh: float = 10 / 255.0,
                dilate: int = 2, bg: np.ndarray | None = None) -> np.ndarray:
    """(T,H,W) bool mask of moving pixels, by |frame - background|.

    Differencing against a MEDIAN background rather than the previous frame is what makes this
    survive real footage: consecutive-frame differencing fires on compression noise everywhere and
    misses slow motion, while the median is a clean background estimate for a fixed camera, which
    every Physics-IQ clip has.

    `bg` MUST come from the whole video (see :func:`video_background`). Taking the median over a
    16-frame window instead silently erases the object whenever it is slow-moving or near-static
    within that window -- the median then *is* the object, the difference is ~0, and the window
    reads as "no motion" even though the clip is full of it.
    """
    import cv2

    g = _to_numpy(frames)
    if bg is None:
        bg = np.median(g, axis=0)
    d = np.abs(g - bg[None])
    m = d > thresh
    if dilate > 0:
        k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        m = np.stack([cv2.morphologyEx(mi.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
                      for mi in m])
    return m


def component_track(frames: torch.Tensor, thresh: float = 10 / 255.0,
                    bg: np.ndarray | None = None) -> dict:
    """Track the largest moving connected component. Returns centroids (T,2) in (x,y) pixels."""
    import cv2

    m = motion_mask(frames, thresh, bg=bg)
    T = m.shape[0]
    cent = np.full((T, 2), np.nan)
    area = np.zeros(T)
    ncomp = np.zeros(T, dtype=int)
    for t in range(T):
        n, _, stats, cxy = cv2.connectedComponentsWithStats(m[t].astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        ncomp[t] = int((areas >= 4).sum())
        j = int(np.argmax(areas)) + 1
        cent[t] = cxy[j]
        area[t] = float(stats[j, cv2.CC_STAT_AREA])
    return {"centroids": cent, "area": area, "n_components": ncomp,
            "mask_frac": m.reshape(T, -1).mean(1)}


def _polyfit_track(cent: np.ndarray, deg: int) -> tuple[np.ndarray | None, int]:
    """Least-squares polynomial fit to a centroid track. Returns (coeffs, n_valid)."""
    ok = np.isfinite(cent).all(1)
    n = int(ok.sum())
    if n < deg + 2:
        return None, n
    t = np.arange(len(cent))[ok]
    t = t - t.mean()
    return np.polyfit(t, cent[ok], deg), n


def measured_velocity(frames: torch.Tensor, thresh: float = 10 / 255.0,
                      bg: np.ndarray | None = None) -> dict[str, float]:
    """Mean image-plane velocity (px/frame) of the dominant moving component."""
    tr = component_track(frames, thresh, bg=bg)
    c, n = _polyfit_track(tr["centroids"], 1)
    if c is None:
        return {"vel_x": float("nan"), "vel_y": float("nan"), "speed": float("nan"),
                "n_valid": n, "ok": False}
    vx, vy = float(c[0, 0]), float(c[0, 1])
    return {"vel_x": vx, "vel_y": vy, "speed": float(np.hypot(vx, vy)),
            "n_valid": n, "ok": True}


def measured_acceleration(frames: torch.Tensor, thresh: float = 10 / 255.0,
                          bg: np.ndarray | None = None) -> dict[str, float]:
    """Image-plane acceleration (px/frame^2) from a quadratic fit: a = 2*c2."""
    tr = component_track(frames, thresh, bg=bg)
    c, n = _polyfit_track(tr["centroids"], 2)
    if c is None:
        return {"acc_x": float("nan"), "acc_y": float("nan"), "accel": float("nan"),
                "n_valid": n, "ok": False}
    ax, ay = 2.0 * float(c[0, 0]), 2.0 * float(c[0, 1])
    return {"acc_x": ax, "acc_y": ay, "accel": float(np.hypot(ax, ay)),
            "n_valid": n, "ok": True}


def _raft(device: str):
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        _RAFT = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).eval().to(device)
    return _RAFT


@torch.no_grad()
def raft_flow(frames: torch.Tensor, device: str = "cuda", batch: int = 8) -> np.ndarray:
    """Dense optical flow between consecutive frames. Returns (T-1, 2, H, W)."""
    model = _raft(device)
    x = frames.to(device).float()
    if x.max() <= 1.0:                       # RAFT expects [-1, 1]
        x = x * 2.0 - 1.0
    a, b = x[:-1], x[1:]
    out = []
    for i in range(0, len(a), batch):
        out.append(model(a[i:i + batch], b[i:i + batch])[-1].cpu())
    return torch.cat(out).numpy()


def flow_affine_fit(flow: np.ndarray, mask: np.ndarray) -> dict:
    """Fit v(x) = v0 + A (x - c) inside `mask`. The antisymmetric part of A gives angular velocity.

    For a rigidly rotating object the flow field about its centroid is exactly antisymmetric, so
    omega = (dvy/dx - dvx/dy) / 2 recovers spin without ever segmenting the object's orientation --
    which is what makes rotation measurable on real video at all.
    """
    H, W = flow.shape[1], flow.shape[2]
    ys, xs = np.nonzero(mask)
    if len(xs) < 24:
        return {"v0": [np.nan, np.nan], "omega": np.nan, "div": np.nan, "n_px": int(len(xs))}
    cx, cy = xs.mean(), ys.mean()
    X = np.stack([np.ones_like(xs, float), xs - cx, ys - cy], 1)
    vx, vy = flow[0][ys, xs], flow[1][ys, xs]
    bx, *_ = np.linalg.lstsq(X, vx, rcond=None)
    by, *_ = np.linalg.lstsq(X, vy, rcond=None)
    return {"v0": [float(bx[0]), float(by[0])],
            "omega": float(0.5 * (by[1] - bx[2])),      # antisymmetric part -> rotation
            "div": float(bx[1] + by[2]),                # divergence -> approach/expansion
            "n_px": int(len(xs))}


def measured_angvel(frames: torch.Tensor, device: str = "cuda",
                    thresh: float = 10 / 255.0, bg: np.ndarray | None = None) -> dict[str, float]:
    """Mean angular velocity (rad/frame) from the antisymmetric part of the flow field."""
    m = motion_mask(frames, thresh, bg=bg)
    fl = raft_flow(frames, device=device)
    om = [flow_affine_fit(fl[t], m[t]) for t in range(len(fl))]
    vals = np.array([o["omega"] for o in om], float)
    good = np.isfinite(vals)
    return {"omega": float(np.nanmean(vals)) if good.any() else float("nan"),
            "n_valid": int(good.sum()), "ok": bool(good.any())}


def flow_velocity(frames: torch.Tensor, device: str = "cuda",
                  thresh: float = 10 / 255.0, bg: np.ndarray | None = None) -> dict[str, float]:
    """Channel-F velocity/acceleration: mean in-mask flow per step, then a fit over time."""
    m = motion_mask(frames, thresh, bg=bg)
    fl = raft_flow(frames, device=device)
    v = np.full((len(fl), 2), np.nan)
    for t in range(len(fl)):
        if m[t].sum() >= 24:
            v[t] = [fl[t][0][m[t]].mean(), fl[t][1][m[t]].mean()]
    ok = np.isfinite(v).all(1)
    if ok.sum() < 3:
        return {"vel_x": np.nan, "vel_y": np.nan, "speed": np.nan,
                "acc_x": np.nan, "acc_y": np.nan, "n_valid": int(ok.sum()), "ok": False}
    t = np.arange(len(v))[ok] - np.arange(len(v))[ok].mean()
    lin = np.polyfit(t, v[ok], 1)            # d(velocity)/dt = acceleration
    return {"vel_x": float(v[ok, 0].mean()), "vel_y": float(v[ok, 1].mean()),
            "speed": float(np.hypot(*v[ok].mean(0))),
            "acc_x": float(lin[0, 0]), "acc_y": float(lin[0, 1]),
            "n_valid": int(ok.sum()), "ok": True}


def pseudo_gt_confidence(mask_meas: dict, flow_meas: dict) -> float:
    """Agreement between the two channels, in [0,1]. Carry as a sample weight, not a hard filter."""
    a = np.array([mask_meas.get("vel_x", np.nan), mask_meas.get("vel_y", np.nan)], float)
    b = np.array([flow_meas.get("vel_x", np.nan), flow_meas.get("vel_y", np.nan)], float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(a @ b / (na * nb))
    scale = float(min(na, nb) / max(na, nb))     # penalise magnitude disagreement too
    return max(0.0, cos) * scale
