"""Checkpointing: EMA weights and robust resume."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, m in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers(), strict=True):
            s.copy_(m)

    def state_dict(self) -> dict[str, Any]:
        return self.shadow.state_dict()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMA | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: EMA | None = None,
    map_location: str = "cpu",
) -> int:
    """Load a checkpoint in place; returns the stored step."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and "ema" in ckpt:
        ema.shadow.load_state_dict(ckpt["ema"])
    return int(ckpt.get("step", 0))
