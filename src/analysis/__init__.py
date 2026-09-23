"""Latent-space analysis: geometry, manifolds, direction codes, intervention/steering, visualization."""

from __future__ import annotations

from . import robotics_steering, steering, subspace, visualization
from .direction_codes import circular_code_score
from .intervention import (
    apply_intervention,
    apply_intervention_multi,
    discover_direction,
    intervention_sweep,
)
from .latent_geometry import layerwise_cka_matrix, pca, pooled_features
from .manifold_analysis import embed_2d, variable_axis_alignment
from .steering import (
    category_mean_direction,
    category_readout,
    category_steering_direction,
    decode_intervention,
    discover_quantity_direction,
    interpolate_trajectories,
    readout_along_direction,
    steer_to_target,
)
from .subspace import category_directions, cosine_matrix, principal_angles, separability

__all__ = [
    "visualization",
    "steering",
    "subspace",
    "robotics_steering",
    "circular_code_score",
    "discover_direction",
    "apply_intervention",
    "apply_intervention_multi",
    "intervention_sweep",
    "pooled_features",
    "pca",
    "layerwise_cka_matrix",
    "embed_2d",
    "variable_axis_alignment",
    "decode_intervention",
    "readout_along_direction",
    "steer_to_target",
    "interpolate_trajectories",
    "discover_quantity_direction",
    "category_steering_direction",
    "category_mean_direction",
    "category_readout",
    "category_directions",
    "cosine_matrix",
    "principal_angles",
    "separability",
]
