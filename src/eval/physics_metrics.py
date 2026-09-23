"""Physics metrics for synthetic data (exact ground truth available).

Operate on predicted vs. ground-truth state ``(B, T, state_dim)`` with a column ``state_keys`` schema
and a per-column validity ``mask``. Metrics:

* position / velocity / acceleration RMSE,
* direction angular error (from the velocity vector),
* gravity parameter error,
* collision-event F1,
* object-permanence accuracy (the ``visible`` flag),
* trajectory rollout error (cumulative position drift).

Columns that are masked-out (e.g. for real benchmark data with no numeric labels) are skipped, so this
returns only the metrics the dataset can actually support — nothing is fabricated.
"""

from __future__ import annotations

import numpy as np
import torch


def _cols(state_keys: list[str], substr: str, mask: torch.Tensor) -> list[int]:
    return [i for i, k in enumerate(state_keys) if substr in k and mask[i] > 0]


def _rmse(pred: torch.Tensor, target: torch.Tensor, cols: list[int]) -> float | None:
    if not cols:
        return None
    return float(torch.sqrt(((pred[..., cols] - target[..., cols]) ** 2).mean()))


def direction_angular_error(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str],
                            mask: torch.Tensor) -> float | None:
    vx = _cols(state_keys, "vel_x", mask)
    vy = _cols(state_keys, "vel_y", mask)
    if not vx or not vy:
        return None
    pa = torch.atan2(pred[..., vy[0]], pred[..., vx[0]])
    ta = torch.atan2(target[..., vy[0]], target[..., vx[0]])
    diff = torch.atan2(torch.sin(pa - ta), torch.cos(pa - ta)).abs()
    return float(diff.mean() * 180.0 / np.pi)


def collision_f1(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str],
                 mask: torch.Tensor, threshold: float = 0.5) -> float | None:
    cols = _cols(state_keys, "collision_event", mask)
    if not cols:
        return None
    p = (torch.sigmoid(pred[..., cols[0]]) > threshold).float()
    t = (target[..., cols[0]] > threshold).float()
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def object_permanence_accuracy(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str],
                               mask: torch.Tensor) -> float | None:
    cols = _cols(state_keys, "visible", mask)
    if not cols:
        return None
    p = (pred[..., cols[0]] > 0.5).float()
    t = (target[..., cols[0]] > 0.5).float()
    return float((p == t).float().mean())


def trajectory_rollout_error(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str],
                             mask: torch.Tensor) -> float | None:
    """Cumulative position drift over time (mean over batch of the final-frame position error)."""
    cols = _cols(state_keys, "pos_", mask)
    if not cols:
        return None
    err = torch.sqrt(((pred[..., cols] - target[..., cols]) ** 2).sum(-1))  # (B, T)
    return float(err[:, -1].mean())


def physics_metrics(pred: torch.Tensor, target: torch.Tensor, state_keys: list[str],
                    mask: torch.Tensor) -> dict[str, float | None]:
    """Full physics-metric suite; ``None`` for metrics the dataset doesn't support."""
    return {
        "position_rmse": _rmse(pred, target, _cols(state_keys, "pos_", mask)),
        "velocity_rmse": _rmse(pred, target, _cols(state_keys, "vel_", mask)),
        "acceleration_rmse": _rmse(pred, target, _cols(state_keys, "acc_", mask)),
        "gravity_error": _rmse(pred, target, _cols(state_keys, "gravity", mask)),
        "direction_angular_error_deg": direction_angular_error(pred, target, state_keys, mask),
        "collision_f1": collision_f1(pred, target, state_keys, mask),
        "object_permanence_acc": object_permanence_accuracy(pred, target, state_keys, mask),
        "trajectory_rollout_error": trajectory_rollout_error(pred, target, state_keys, mask),
    }
