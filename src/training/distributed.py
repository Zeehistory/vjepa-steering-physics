"""Distributed-training helpers built around HuggingFace Accelerate.

These thin wrappers keep ``train_decoder`` readable and centralize rank/seed/printing logic so the same
code runs on CPU, single-GPU, multi-GPU, and multi-node (via ``accelerate launch`` / SLURM).
"""

from __future__ import annotations

from typing import Any


def build_accelerator(mixed_precision: str = "no", grad_accum: int = 1, log_with: Any = None):
    """Create an ``accelerate.Accelerator`` (imported lazily)."""
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    return Accelerator(
        mixed_precision=mixed_precision if mixed_precision in ("fp16", "bf16") else "no",
        gradient_accumulation_steps=grad_accum,
        log_with=log_with,
        project_config=ProjectConfiguration(automatic_checkpoint_naming=False),
    )


def is_main_process(accelerator) -> bool:
    return accelerator.is_main_process


def rank_zero_print(accelerator, *args: Any) -> None:
    if accelerator.is_main_process:
        print(*args)
