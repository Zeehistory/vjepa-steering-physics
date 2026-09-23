"""Baselines and controls — required so reconstruction / physics claims are credible.

* **copy-last-frame** — predict every output frame as the (downsampled) first/last input frame. A
  reconstruction model must beat this to be meaningful.
* **mean-frame** — predict the temporal mean frame everywhere.
* **oracle-state** — uses ground-truth state directly; upper bound for physics metrics (must be ~perfect).
* **random-frame** — uniform-noise frames; lower-bound sanity check.
* **shuffled-latent** control is implemented in the probe module (``train_probe``).

Each returns a frame or state tensor matching the target, so the same metric functions apply.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def copy_first_frame(target: torch.Tensor) -> torch.Tensor:
    """``(B, T, C, H, W)`` -> repeat frame 0 across time."""
    return target[:, :1].expand_as(target).clone()


def mean_frame(target: torch.Tensor) -> torch.Tensor:
    return target.mean(dim=1, keepdim=True).expand_as(target).clone()


def random_frame(target: torch.Tensor, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=target.device).manual_seed(seed)
    return torch.rand(target.shape, generator=g, device=target.device)


def oracle_state(target_state: torch.Tensor, noise: float = 0.0, seed: int = 0) -> torch.Tensor:
    """Return GT state (optionally with small noise) — physics-metric upper bound."""
    if noise <= 0:
        return target_state.clone()
    g = torch.Generator(device=target_state.device).manual_seed(seed)
    return target_state + noise * torch.randn(target_state.shape, generator=g, device=target_state.device)


def downsample_to(frames: torch.Tensor, size: int) -> torch.Tensor:
    b, t, c, h, w = frames.shape
    return F.interpolate(frames.flatten(0, 1), size=(size, size), mode="bilinear",
                         align_corners=False).reshape(b, t, c, size, size)


def frame_baselines(target: torch.Tensor) -> dict[str, torch.Tensor]:
    """All frame-level baselines for a target batch."""
    return {
        "copy_first_frame": copy_first_frame(target),
        "mean_frame": mean_frame(target),
        "random_frame": random_frame(target),
    }
