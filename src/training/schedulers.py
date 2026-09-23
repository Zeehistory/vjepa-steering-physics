"""Learning-rate schedulers with linear warmup."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    warmup_steps: int = 0,
    max_steps: int = 1000,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Return a step-wise ``LambdaLR`` (call ``.step()`` every optimizer step)."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if name == "constant":
            return 1.0
        if name == "linear":
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
        # cosine (default)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
