"""Loss functions for decoder training.

A weighted mixture of pixel, perceptual, temporal, and physics-aware terms. Each term is its own
function so it can be unit-tested; :class:`DecoderLoss` combines them per the loss config. Terms whose
weight is zero are skipped (no wasted compute). Optional perceptual loss (LPIPS) is imported lazily and
skipped with a warning if the dependency is absent — never silently faked.

Frame tensors are ``(B, T, C, H, W)`` in ``[0, 1]``; state tensors are ``(B, T, state_dim)``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust L1 variant: ``sqrt((x-y)^2 + eps^2)``."""
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def foreground_weighted_charbonnier(
    pred: torch.Tensor, target: torch.Tensor, gamma: float = 10.0, eps: float = 1e-3,
    fg_thresh: float = 0.5,
) -> torch.Tensor:
    """Charbonnier reconstruction that scores the (dark) foreground separately from the background.

    On the clean moving-ball scene the object is a small dark disk (~2% of pixels) on a white
    background, so an *area-averaged* pixel loss is minimized by predicting a blank frame and the
    decoder collapses (ball dropped entirely). We instead compute the charbonnier **mean over the
    foreground** and **mean over the background** independently and return ``bg + gamma * fg`` so the
    tiny ball contributes on equal footing with the whole background regardless of its area.

    Foreground membership uses a **hard darkness threshold** (``1-target > fg_thresh``). An earlier
    *soft* darkness weight (``w = 1-target``) failed: the background's residual darkness (~0.05) leaks
    into the foreground denominator (98% of pixels × 0.05 >> 2% of pixels × 0.8), diluting the ball
    gradient ~4x, so the decoder still collapsed to a uniform frame (observed: fg4 step_300 rel-darkness
    decayed 0.039→0.01). With a hard mask the foreground mean is taken over ball pixels ONLY, so a
    uniform-collapse prediction incurs the full ``gamma * ~0.77`` foreground penalty (vs a diluted
    ~0.19) and the only way down is to actually localize and darken the ball.
    """
    err = torch.sqrt((pred - target) ** 2 + eps**2)                    # (B,T,C,H,W)
    dark = (1.0 - target.mean(dim=2, keepdim=True)).clamp(min=0.0)     # (B,T,1,H,W) target darkness
    fg = (dark > fg_thresh).to(err.dtype)                              # hard ball mask (no bg leak)
    bg = 1.0 - fg
    fg_err = (err * fg).sum() / fg.expand_as(err).sum().clamp(min=1.0)  # mean error on the ball ONLY
    bg_err = (err * bg).sum() / bg.expand_as(err).sum().clamp(min=1.0)  # mean error on the background
    return bg_err + gamma * fg_err


def marker_weighted_charbonnier(
    pred: torch.Tensor, target: torch.Tensor, redness_thresh: float = 0.12, eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier scored ONLY on the pixels where the target's coloured marker actually is.

    Why this term has to exist. ``foreground_weighted_charbonnier`` splits pixels by a hard DARKNESS
    threshold (``1 - gray > 0.5``), which is right for a dark ball on a light background and silently
    wrong for a BRIGHT marker. On ``spin_ball3d`` the marker is amber ``(1.0, 0.75, 0.10)``, luma 0.617,
    so its darkness is 0.383 -- it fails the foreground test and is scored as BACKGROUND, where it is
    ~55 px in 65536 (0.08% of the frame) and its gradient is negligible against the wall and floor. The
    decoder then does the rational thing and erases it: two full runs reached
    ``marker_mass_decoded/rendered = 0.000`` while rendering the ball's motion beautifully (speed corr
    0.942). This is exactly the dilution trap that function's own docstring describes for the ball,
    landing on the marker instead.

    The mask is by COLOUR rather than brightness: ``R - max(G, B)``, which is 0.25 for the marker and
    <= 0 for every other material in the scene (ball 0.10/0.10/0.13, floor 0.70/0.70/0.74, wall
    0.86/0.86/0.89). Weighting target-marker pixels is what makes a BLANK prediction expensive: render
    nothing there and the error is the full marker contrast rather than zero.

    The weight is the sum of TARGET redness and PREDICTED redness, and the second half is not optional.
    A first version masked on the target alone, which is asymmetric: it makes omitting the marker costly
    while making SPURIOUS marker paint free, since amber invented where the target has none simply falls
    outside the mask. Trained for 4000 steps that way, the decoder did exactly what the loss asked -- it
    flooded the frame with marker colour, taking ``marker_mass_decoded/rendered`` from 0.000 to **467**,
    with speed correlation collapsing to nan and heading error 127.5 deg. Including predicted redness
    puts weight wherever EITHER image has marker colour, so a false marker lands on a high-weight pixel
    whose target is background and pays the full contrast. Soft (raw redness, not a threshold) so the
    penalty is differentiable in the prediction rather than a mask that gradients cannot flow through.

    Deliberately NOT an orientation loss. ``frame_orientation_loss`` supervises the marker's image-space
    direction against ``obj0_theta``, which is correct only when the marker rotates in the image plane.
    Here it sits 35 deg off the ball's pole under a 62 deg elevation camera, so it traces a foreshortened
    ELLIPSE and the certified tracker needs ``spin_tracking.ellipse_basis`` plus an unprojection to
    recover theta -- a naive image-space atan2 cannot equal it. Measured on real rendered frames, that
    loss's target and its detector disagree by ~97 deg, i.e. worse than chance, so switching it on here
    would have trained against noise. This term needs no angle convention at all.
    """
    err = torch.sqrt((pred - target) ** 2 + eps**2)                    # (B,T,C,H,W)

    def _redness(x: torch.Tensor) -> torch.Tensor:
        return (x[:, :, 0] - torch.maximum(x[:, :, 1], x[:, :, 2])).clamp(min=0.0)   # (B,T,H,W)

    red_t = _redness(target)
    red_p = _redness(pred)
    # Hard-gate the TARGET side at the threshold (0.12 selects the ~29 px marker core and rejects the
    # warm-lit surfaces that a 0.05 cut would sweep in), but keep the PREDICTED side soft and ungated so
    # spurious marker paint is penalised smoothly from the very first amber pixel rather than only once
    # it crosses a threshold.
    w = (red_t > redness_thresh).to(err.dtype) * red_t + red_p
    w = w.unsqueeze(2)                                                 # (B,T,1,H,W)
    return (err * w).sum() / w.expand_as(err).sum().clamp(min=1e-3)


def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Mean SSIM over frames. Inputs ``(B, T, C, H, W)`` in ``[0, 1]``. Returns SSIM in ``[0, 1]``."""
    b, t, c, h, w = pred.shape
    p = pred.reshape(b * t, c, h, w)
    g = target.reshape(b * t, c, h, w)
    win = _gaussian_window(window_size, 1.5, c, pred.device)
    p, g = p.float(), g.float()
    pad = window_size // 2
    mu_p = F.conv2d(p, win, padding=pad, groups=c)
    mu_g = F.conv2d(g, win, padding=pad, groups=c)
    mu_p2, mu_g2, mu_pg = mu_p**2, mu_g**2, mu_p * mu_g
    sig_p = F.conv2d(p * p, win, padding=pad, groups=c) - mu_p2
    sig_g = F.conv2d(g * g, win, padding=pad, groups=c) - mu_g2
    sig_pg = F.conv2d(p * g, win, padding=pad, groups=c) - mu_pg
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_pg + c1) * (2 * sig_pg + c2)) / ((mu_p2 + mu_g2 + c1) * (sig_p + sig_g + c2))
    return s.mean().clamp(0, 1)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target)


def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor, scales: int = 3) -> torch.Tensor:
    total = 0.0
    p, g = pred, target
    b, t = pred.shape[:2]
    for i in range(scales):
        total = total + (1.0 - ssim(p, g))
        if i < scales - 1:
            p = F.avg_pool2d(p.flatten(0, 1), 2).reshape(b, t, p.shape[2], p.shape[3] // 2, p.shape[4] // 2)
            g = F.avg_pool2d(g.flatten(0, 1), 2).reshape(b, t, g.shape[2], g.shape[3] // 2, g.shape[4] // 2)
    return total / scales


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match frame-to-frame deltas, encouraging consistent motion rather than per-frame independence."""
    dp = pred[:, 1:] - pred[:, :-1]
    dg = target[:, 1:] - target[:, :-1]
    return F.l1_loss(dp, dg)


def _soft_ball_centroid(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable per-frame centroid of the dark ball. Returns (centroid (B,T,2), mass (B,T), dark²-weighted spread (B,T)).

    Weights each pixel by ``darkness²`` (= ``(1-gray)²``) so the dark ball (darkness ~0.8 -> w ~0.66)
    dominates the faint white background (darkness ~0.05 -> w ~0.003) by ~250x per pixel — the centroid
    tracks the ball, not the frame center. Fully differentiable (soft mass-weighted mean), so it can
    supervise WHERE the rendered ball is, frame by frame.
    """
    gray = frames.mean(dim=2)                                   # (B,T,H,W) ~1 bg, ~0 ball
    w = (1.0 - gray).clamp(min=0.0) ** 2                        # dark² weighting
    b, t, h, wd = w.shape
    xs = torch.linspace(0, 1, wd, device=w.device).view(1, 1, 1, wd)
    ys = torch.linspace(0, 1, h, device=w.device).view(1, 1, h, 1)
    mass = w.sum(dim=(2, 3))                                    # (B,T)
    denom = mass.clamp(min=1e-4)
    cx = (w * xs).sum(dim=(2, 3)) / denom
    cy = (w * ys).sum(dim=(2, 3)) / denom
    cen = torch.stack([cx, cy], dim=-1)                        # (B,T,2)
    # second moment about the centroid = how spread-out the dark mass is (a path-covering smear is large)
    var = (w * ((xs - cx[..., None, None]) ** 2 + (ys - cy[..., None, None]) ** 2)).sum(dim=(2, 3)) / denom
    return cen, mass, var


def frame_position_loss(pred: torch.Tensor, target_state: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    """MSE between the rendered ball's per-frame centroid and the GT per-frame position.

    THE fix for temporal-average blur: an L1/SSIM pixel loss is happy to render the ball as a static
    smear covering its whole path (low average pixel error), whose centroid barely moves -> the decoded
    speed collapses ~4x. This loss ties the *rendered* ball's centroid to the exact GT position at EVERY
    frame, so the only way down is to render the ball translating at the true speed. Uses GT position
    (normalized [0,1], same convention as the soft centroid). The moving_ball dataset names the columns
    ``pos_x``/``pos_y``; multi-object synthetic_physics uses ``obj0_pos_x``/``obj0_pos_y`` — accept both.
    """
    def _find(*names: str) -> int | None:
        for n in names:
            if n in state_keys:
                return state_keys.index(n)
        return None

    xi = _find("pos_x", "obj0_pos_x")
    yi = _find("pos_y", "obj0_pos_y")
    if xi is None or yi is None:
        return pred.new_zeros(())
    gt = torch.stack([target_state[..., xi], target_state[..., yi]], dim=-1)  # (B,T,2)
    cen, _, _ = _soft_ball_centroid(pred)
    return F.mse_loss(cen, gt)


def frame_spread_loss(pred: torch.Tensor, max_var: float = 0.01) -> torch.Tensor:
    """Penalize a rendered dark mass that is more SPREAD OUT than a compact disk (anti path-smear).

    Only penalizes spread in EXCESS of ``max_var`` (a disk of radius ~0.11 has 2nd moment ~0.006), so a
    correct compact ball is free while a path-covering smear is pushed down. Complements the centroid
    loss: centroid says *where*, spread says *don't smear across the trajectory*.
    """
    _, _, var = _soft_ball_centroid(pred)
    return (var - max_var).clamp(min=0.0).mean()


def _soft_marker_direction(frames: torch.Tensor, red_thresh: float = 0.2,
                           body_power: float = 1.0, body_thresh: float = 0.0,
                           marker_thresh: float = 0.0) -> torch.Tensor:
    """Differentiable per-frame UNIT direction ``(marker - centre)`` of the spinning bar+marker ``(B,T,2)``.

    Rotational analog of :func:`_soft_ball_centroid`. BODY weight = darkness with red softly suppressed ->
    its centroid is the rotation CENTRE; MARKER weight = redness ``R - max(G,B)`` -> its centroid is the
    leading end. Direction = normalized ``(marker - centre)``, matching the tracker's
    ``atan2(marker_y-centre_y, marker_x-centre_x)`` convention, so it can supervise the rendered ORIENTATION
    frame by frame. Fully differentiable (soft mass-weighted means).

    ``body_thresh`` gates which pixels may vote for the rotation CENTRE at all, and on a
    bright-background scene it is the difference between this function tracking the ball and tracking
    the room. The centre is a weighted mean over the WHOLE frame, so it only lands on the ball when the
    ball dominates the total weight. On ``spin_ball3d`` it does not: floor darkness ~0.29 and wall ~0.14
    against the ball's ~0.89 is only a ~3x per-pixel edge, while the background covers ~25x more pixels.
    The centre lands in the room, so ``marker - centre`` encodes WHERE THE BALL IS rather than the
    marker's phase.

    ``body_power`` alone does NOT fix this, and it was tried first. Squaring reweights the room but does
    not remove it -- floor ``0.082 x ~50%`` of pixels still outstrips ball ``0.79 x ~2%`` by ~2.6x.
    Measured against ground-truth theta on real frames, power 1 gave 97 deg of error and power 2 gave
    121 deg, both at or worse than the 90 deg chance level, and correcting the TARGET for the scene's
    ellipse geometry did not help either (116 deg) precisely because the fault is here, in the detector.

    A hard darkness gate does fix it, because it changes the SUPPORT rather than the weighting: at 0.5
    the ball (0.89) is admitted while floor (0.29), wall (0.14) and the marker itself (0.38) are all
    excluded, which is the same cut ``foreground_weighted_charbonnier`` uses to isolate this ball.
    Default 0.0 disables the gate so every previously-certified ``angvel`` config reproduces exactly.
    """
    b_, t_, c_, h, w = frames.shape
    r, g, b = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]     # (B,T,H,W)
    gray = frames.mean(dim=2)
    dark = (1.0 - gray).clamp(min=0.0) ** body_power
    redness = (r - torch.maximum(g, b)).clamp(min=0.0)
    # Gate the MARKER support too. Ungated, this is the dominant bug on a warm-lit scene: redness>0.05
    # covers 29.8% of a spin_ball3d frame, and summing that faint background over ~19500 px swamps the
    # marker's ~29 px at 0.25 by roughly 190:1, so the "marker centroid" is really the centroid of the
    # room. Measured, that put the direction 97-105 deg from truth -- at or worse than chance -- which no
    # target correction could rescue (the exactly-projected target scored 116 deg).
    if marker_thresh > 0.0:
        # STEEP SIGMOID, not a hard cut. These weights are applied to the DECODER'S OUTPUT during
        # training, and early on it may have no pixel above the threshold at all -- a hard mask would
        # then zero every weight and the centroid's gradient with it, exactly when the signal is most
        # needed. A sigmoid at k=100 is within a fraction of a degree of the hard gate on real frames
        # while still passing gradient to sub-threshold pixels, so the marker can be pushed INTO
        # existence rather than only refined once it already exists.
        marker_w = redness * torch.sigmoid((redness - marker_thresh) * 100.0) + 1e-8
    else:
        marker_w = redness + 1e-8
    body_w = dark * torch.sigmoid((red_thresh - redness) * 40.0) + 1e-8   # dark but not red
    if body_thresh > 0.0:
        # Gate the SUPPORT, not just the weight: only genuinely dark pixels may vote for the centre.
        body_w = body_w * torch.sigmoid(((1.0 - gray) - body_thresh) * 100.0) + 1e-8
    xs = torch.linspace(0, 1, w, device=frames.device).view(1, 1, 1, w)
    ys = torch.linspace(0, 1, h, device=frames.device).view(1, 1, h, 1)

    def _cen(wt):
        m = wt.sum(dim=(2, 3))
        cx = (wt * xs).sum(dim=(2, 3)) / m
        cy = (wt * ys).sum(dim=(2, 3)) / m
        return cx, cy

    bx, by = _cen(body_w)
    mx, my = _cen(marker_w)
    ux, uy = mx - bx, my - by
    n = torch.sqrt(ux * ux + uy * uy) + 1e-6
    return torch.stack([ux / n, uy / n], dim=-1)                    # (B,T,2)


def frame_orientation_loss(pred: torch.Tensor, target_state: torch.Tensor,
                           state_keys: list[str], body_power: float = 1.0) -> torch.Tensor:
    """MSE between the rendered marker's per-frame unit direction and the GT orientation ``(cos,sin)theta``.

    THE angular analog of :func:`frame_position_loss`. An L1/SSIM pixel loss is happy to render the spinning
    marker as a faint temporally-blurred red arc whose per-frame orientation does NOT sweep at the true rate
    -> decoded angular velocity is unfaithful (verified: ``measured_angvel(decode(true H_b))`` correlates ~0
    / negatively with GT omega). This ties the RENDERED marker direction to the exact GT orientation at EVERY
    frame, so the only way down is to render the marker at the true angle each frame -> faithful spin. The GT
    column ``theta``/``obj0_theta`` equals the renderer's ``phi`` (marker at ``centre + off*(cos phi, sin phi)``,
    same convention as the tracker), so there is no offset to fit. Zero (skipped) for datasets without a
    ``theta`` column (translation scenarios), so only the angvel decoder pays it.
    """
    def _find(*names: str) -> int | None:
        for n in names:
            if n in state_keys:
                return state_keys.index(n)
        return None

    ti = _find("obj0_theta", "theta")
    if ti is None:
        return pred.new_zeros(())
    theta = target_state[..., ti]                                  # (B,T)
    tgt = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (B,T,2)
    u = _soft_marker_direction(pred, body_power=body_power)
    return F.mse_loss(u, tgt)


def frame_orientation_projected_loss(
    pred: torch.Tensor, target_state: torch.Tensor, state_keys: list[str],
    body_thresh: float = 0.5, marker_thresh: float = 0.18,
) -> torch.Tensor:
    """Marker PHASE supervision for a 3-D scene, with the target projected through the scene geometry.

    ``frame_orientation_loss`` compares the rendered marker's image direction against
    ``(cos theta, sin theta)``. That is right only when the marker rotates IN the image plane. On
    ``spin_ball3d`` it sits 35 deg off the ball's pole under a 62 deg elevation camera, so its image
    offset traces a foreshortened ELLIPSE: ``d(theta) = A cos theta + B sin theta + C``. Measured on real
    frames, the naive target is 70 deg from truth even with a perfect detector, so it cannot supervise
    phase here.

    This builds the correct target instead, reusing the certified tracker's own geometry: unproject the
    ball's image centre to the plane it slides on, take ``spin_tracking.ellipse_basis`` there, and
    evaluate the ellipse at the GT ``theta``. Validated against the tracker's measured marker offset at
    **0.2 px**, with the offset angle sweeping 344 deg over a clip -- so the target is both exact and
    genuinely informative about phase.

    WHY THE TERM THIS REPLACES WAS NEEDED. A colour-masked reconstruction term (``marker``) makes the
    marker EXIST but says nothing about WHERE IN THE ROTATION it sits, and a phase-averaged smear
    satisfies it: at 12k steps the decoder reached marker mass 2.10 with ``omega_corr = -0.19``, i.e.
    plenty of marker pixels and no spin signal. Presence and phase are different constraints.

    The detector thresholds are not tuning knobs, they are the difference between signal and noise. Both
    centroids are means over the WHOLE frame, so on this warm-lit scene the room wins both: ungated,
    redness>0.05 covers 29.8% of the frame and swamps the marker's ~29 px by ~190:1, and the darkness
    weight lets floor+wall outvote the ball. Ungated the direction lands 105 deg from truth; gated it
    lands 4.7 deg.
    """
    def _find(*names: str) -> int | None:
        for n in names:
            if n in state_keys:
                return state_keys.index(n)
        return None

    ti = _find("obj0_theta", "theta")
    xi = _find("obj0_pos_x", "pos_x")
    yi = _find("obj0_pos_y", "pos_y")
    if ti is None or xi is None or yi is None:
        return pred.new_zeros(())

    try:
        from src.analysis import spin_tracking as _st
    except Exception:                                  # scene module unavailable -> skip, never fake
        return pred.new_zeros(())

    import numpy as _np
    st_np = target_state.detach().float().cpu().numpy()
    theta, px, py = st_np[..., ti], st_np[..., xi], st_np[..., yi]
    shp = theta.shape

    # VECTORIZED over (batch x frames). The scalar version -- a Python double loop calling
    # ``ellipse_basis`` per frame, each of which makes four separate ``project`` calls -- ran 64+
    # projections per step on the CPU while the GPU idled, and measured 43.5 min/1000 steps against the
    # 15 min/1000 of the same run without this term. That is the loss costing 3x the model. ``project``
    # and ``unproject_to_ball_plane`` both accept ``(...,3)`` / ``(...,2)``, so the whole target is four
    # batched calls regardless of batch size, with identical arithmetic.
    wc = _st.unproject_to_ball_plane(_np.stack([px.ravel(), py.ravel()], axis=1))     # (N,3)
    s_, cz_ = _np.sin(_st.MARKER_POLAR), _np.cos(_st.MARKER_POLAR)
    off = _st.MARKER_OFFSET
    p0 = _st.project(wc)
    pC = _st.project(wc + _np.array([0.0, 0.0, off * cz_]))
    pA = _st.project(wc + _np.array([off * s_, 0.0, off * cz_]))
    pB = _st.project(wc + _np.array([0.0, off * s_, off * cz_]))
    C = pC - p0
    A = pA - p0 - C
    B = pB - p0 - C
    th = theta.ravel()[:, None]
    d = A * _np.cos(th) + B * _np.sin(th) + C
    d = d / (_np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    tgt_t = torch.as_tensor(d.reshape(shp + (2,)), dtype=pred.dtype, device=pred.device)

    u = _soft_marker_direction(pred, body_thresh=body_thresh, marker_thresh=marker_thresh)
    return F.mse_loss(u, tgt_t)


def masked_state_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """MSE over per-frame state, honoring a per-column validity ``mask`` ``(B, state_dim)``."""
    err = (pred - target) ** 2
    if mask is not None:
        m = mask.unsqueeze(1)  # (B, 1, state_dim)
        denom = m.sum().clamp(min=1.0)
        return (err * m).sum() / denom / pred.shape[1]
    return err.mean()


def _state_cols(state_keys: list[str], substr: str) -> list[int]:
    return [i for i, k in enumerate(state_keys) if substr in k]


def trajectory_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    """L2 on position columns across time (object trajectories)."""
    cols = _state_cols(state_keys, "pos_")
    if not cols:
        return pred.new_zeros(())
    return F.mse_loss(pred[..., cols], target[..., cols])


def velocity_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    cols = _state_cols(state_keys, "vel_")
    if not cols:
        return pred.new_zeros(())
    return F.mse_loss(pred[..., cols], target[..., cols])


def acceleration_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    cols = _state_cols(state_keys, "acc_")
    if not cols:
        return pred.new_zeros(())
    return F.mse_loss(pred[..., cols], target[..., cols])


def collision_loss(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str]) -> torch.Tensor:
    """Binary cross-entropy on the collision-event column."""
    cols = _state_cols(state_keys, "collision_event")
    if not cols:
        return pred.new_zeros(())
    logits = pred[..., cols]
    tgt = target[..., cols].clamp(0, 1)
    return F.binary_cross_entropy_with_logits(logits, tgt)


class _LPIPS:
    """Lazy LPIPS holder; returns None if the optional dependency is missing."""

    _net: Any = None
    _warned = False

    @classmethod
    def get(cls):
        if cls._net is not None:
            return cls._net
        try:
            import lpips

            cls._net = lpips.LPIPS(net="alex")
            cls._net.eval()
            for p in cls._net.parameters():
                p.requires_grad_(False)
            return cls._net
        except Exception:
            if not cls._warned:
                warnings.warn("LPIPS unavailable (pip install -e .[extras]); perceptual loss skipped.", stacklevel=2)
                cls._warned = True
            return None


def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    net = _LPIPS.get()
    if net is None:
        return pred.new_zeros(())
    net = net.to(pred.device)
    b, t, c, h, w = pred.shape
    p = pred.reshape(b * t, c, h, w).float() * 2 - 1
    g = target.reshape(b * t, c, h, w).float() * 2 - 1
    return net(p, g).mean()


class DecoderLoss(torch.nn.Module):
    """Weighted combination of the above terms, driven by a loss config."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        pred_frames: torch.Tensor | None,
        target_frames: torch.Tensor | None,
        pred_state: torch.Tensor | None = None,
        target_state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        state_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        c = self.cfg
        device = (pred_frames if pred_frames is not None else pred_state).device
        total = torch.zeros((), device=device)
        logs: dict[str, float] = {}

        def add(name: str, weight: float, value: torch.Tensor) -> None:
            nonlocal total
            if weight != 0.0:
                total = total + weight * value
                logs[name] = float(value.detach())

        if pred_frames is not None and target_frames is not None:
            if target_frames.shape != pred_frames.shape:
                target_frames = F.interpolate(
                    target_frames.flatten(0, 1), size=pred_frames.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).reshape(pred_frames.shape)
            lo, hi = getattr(c, "target_lo", 0.0), getattr(c, "target_hi", 1.0)
            if lo != 0.0 or hi != 1.0:
                # compress targets off the sigmoid boundary so the decoder optimum sits at a finite,
                # non-saturated logit (keeps gradients alive -> escapes uniform-collapse on white scenes)
                target_frames = lo + (hi - lo) * target_frames
            add("charbonnier", c.charbonnier, charbonnier_loss(pred_frames, target_frames))
            if getattr(c, "frame_orientation_projected", 0.0) != 0.0:
                add("frame_orientation_projected", c.frame_orientation_projected,
                    frame_orientation_projected_loss(
                        pred_frames, target_state, state_keys,
                        getattr(c, "frame_orientation_body_thresh", 0.5),
                        getattr(c, "frame_orientation_marker_thresh", 0.18)))
            if getattr(c, "marker", 0.0) != 0.0:
                add("marker", c.marker,
                    marker_weighted_charbonnier(pred_frames, target_frames,
                                                getattr(c, "marker_redness_thresh", 0.12)))
            if getattr(c, "foreground", 0.0) != 0.0:
                add("foreground", c.foreground,
                    foreground_weighted_charbonnier(pred_frames, target_frames,
                                                    getattr(c, "foreground_gamma", 50.0)))
            add("ssim", c.ssim, ssim_loss(pred_frames, target_frames))
            add("ms_ssim", c.ms_ssim, ms_ssim_loss(pred_frames, target_frames))
            if c.lpips != 0.0:
                add("lpips", c.lpips, lpips_loss(pred_frames, target_frames))
            add("temporal", c.temporal_consistency, temporal_consistency_loss(pred_frames, target_frames))
            # Per-frame rendered-ball position supervision (kills temporal-average blur). Uses the raw
            # rendered frames + GT position; independent of the target compression above.
            if target_state is not None and state_keys is not None:
                add("frame_position", getattr(c, "frame_position", 0.0),
                    frame_position_loss(pred_frames, target_state, state_keys))
                # Per-frame rendered-ORIENTATION supervision (kills the rotation smear that makes decoded
                # angular velocity unfaithful). Only nonzero-weighted for the angvel decoder.
                add("frame_orientation", getattr(c, "frame_orientation", 0.0),
                    frame_orientation_loss(pred_frames, target_state, state_keys,
                                           getattr(c, "frame_orientation_body_power", 1.0)))
            add("frame_spread", getattr(c, "frame_spread", 0.0),
                frame_spread_loss(pred_frames, getattr(c, "frame_spread_max_var", 0.01)))

        if pred_state is not None and target_state is not None and state_keys is not None:
            add("state", c.state, masked_state_loss(pred_state, target_state, state_mask))
            add("trajectory", c.trajectory, trajectory_loss(pred_state, target_state, state_keys))
            add("velocity", c.velocity, velocity_loss(pred_state, target_state, state_keys))
            add("acceleration", c.acceleration, acceleration_loss(pred_state, target_state, state_keys))
            add("collision", c.collision, collision_loss(pred_state, target_state, state_keys))

        logs["total"] = float(total.detach())
        return total, logs
