"""SLURM helpers for multi-node launches.

Minimal, dependency-free helpers to detect a SLURM allocation and emit a submission script. Full
auto-requeue / signal handling is deferred (see docs/reproducibility.md).
"""

from __future__ import annotations

import os
from pathlib import Path


def in_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def slurm_env() -> dict[str, str | None]:
    keys = ["SLURM_JOB_ID", "SLURM_NODEID", "SLURM_PROCID", "SLURM_NTASKS", "SLURM_GPUS_ON_NODE"]
    return {k: os.environ.get(k) for k in keys}


def write_sbatch_script(
    path: str | Path,
    command: str,
    nodes: int = 1,
    gpus_per_node: int = 8,
    partition: str = "gpu",
    time: str = "24:00:00",
    job_name: str = "vjepa-decoder",
) -> Path:
    """Emit a multi-node ``accelerate``-friendly sbatch script."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time}
#SBATCH --output=%x-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun {command}
"""
    path.write_text(script)
    return path
