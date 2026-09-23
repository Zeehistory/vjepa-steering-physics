"""Evaluation suite: reconstruction, physics, latent metrics, baselines, generalization."""

from __future__ import annotations

from .baselines import frame_baselines, oracle_state
from .generalization import dataset_shift_report, frechet_distance, rbf_mmd2
from .latent_metrics import linear_cka, participation_ratio, rbf_cka, twonn_intrinsic_dim
from .physics_metrics import physics_metrics
from .reconstruction_metrics import reconstruction_metrics

__all__ = [
    "reconstruction_metrics",
    "physics_metrics",
    "frame_baselines",
    "oracle_state",
    "linear_cka",
    "rbf_cka",
    "participation_ratio",
    "twonn_intrinsic_dim",
    "dataset_shift_report",
    "rbf_mmd2",
    "frechet_distance",
]
