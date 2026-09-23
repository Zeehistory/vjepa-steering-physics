"""Device and hardware utilities."""

from __future__ import annotations

import platform
from typing import Any

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def hardware_summary() -> dict[str, Any]:
    """Return a JSON-serializable description of the hardware (logged with every run)."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["cuda_version"] = torch.version.cuda
    return info
