"""Reconstruction metrics: PSNR, SSIM/MS-SSIM, L1/L2, temporal consistency.

Frame tensors are ``(B, T, C, H, W)`` in ``[0, 1]``. LPIPS and FVD are optional/deferred and handled
explicitly (skipped with a note rather than faked) so the metrics JSON never contains fabricated values.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from ..decoders.loss_functions import ssim as _ssim_fn


def _match(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.shape == pred.shape:
        return target
    return F.interpolate(
        target.flatten(0, 1), size=pred.shape[-2:], mode="bilinear", align_corners=False
    ).reshape(pred.shape[0], pred.shape[1], pred.shape[2], *pred.shape[-2:])


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    target = _match(pred, target)
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(_ssim_fn(pred, _match(pred, target)))


def ms_ssim(pred: torch.Tensor, target: torch.Tensor, scales: int = 3) -> float:
    target = _match(pred, target)
    vals = []
    p, g = pred, target
    b, t = pred.shape[:2]
    for i in range(scales):
        vals.append(float(_ssim_fn(p, g)))
        if i < scales - 1 and p.shape[-1] >= 8:
            p = F.avg_pool2d(p.flatten(0, 1), 2).reshape(b, t, p.shape[2], p.shape[3] // 2, p.shape[4] // 2)
            g = F.avg_pool2d(g.flatten(0, 1), 2).reshape(b, t, g.shape[2], g.shape[3] // 2, g.shape[4] // 2)
    return float(sum(vals) / len(vals))


def l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.l1_loss(pred, _match(pred, target)))


def l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.mse_loss(pred, _match(pred, target)).sqrt())


def temporal_consistency(pred: torch.Tensor, target: torch.Tensor) -> float:
    """L1 between predicted and target frame-to-frame deltas (lower = more consistent motion)."""
    target = _match(pred, target)
    dp = pred[:, 1:] - pred[:, :-1]
    dg = target[:, 1:] - target[:, :-1]
    return float(F.l1_loss(dp, dg))


def fvd(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    """Fréchet Video Distance (deferred). Requires an I3D feature extractor; returns ``None`` for now."""
    warnings.warn("FVD is not implemented (needs I3D features); reported as null.", stacklevel=2)
    return None


def lpips(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    from ..decoders.loss_functions import _LPIPS

    net = _LPIPS.get()
    if net is None:
        return None
    target = _match(pred, target)
    b, t, c, h, w = pred.shape
    net = net.to(pred.device)
    with torch.no_grad():
        v = net(pred.reshape(b * t, c, h, w) * 2 - 1, target.reshape(b * t, c, h, w) * 2 - 1)
    return float(v.mean())


def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    """Compute the full reconstruction-metric suite for one batch."""
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "ms_ssim": ms_ssim(pred, target),
        "l1": l1(pred, target),
        "l2": l2(pred, target),
        "temporal_consistency": temporal_consistency(pred, target),
        "lpips": lpips(pred, target),
        "fvd": fvd(pred, target),
    }
