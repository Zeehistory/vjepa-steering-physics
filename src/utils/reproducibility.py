"""Reproducibility: seeding, determinism, and run provenance snapshots."""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gpu import hardware_summary


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit() -> str | None:
    """Best-effort current git commit hash; ``None`` if not a repo / git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:
        return None


def git_is_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance dict logged with every run."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "hardware": hardware_summary(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(output_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_meta.json"
    path.write_text(json.dumps(run_metadata(extra), indent=2))
    return path
