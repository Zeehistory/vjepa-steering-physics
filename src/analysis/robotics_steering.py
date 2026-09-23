"""Robotics: detect the achievable-vs-non-achievable latent difference, then steer (Step 4).

The robotics question ("we can detect and steer — so what?") becomes concrete here:

1. **Detect** — is success vs. failure linearly separable in the latent space? A linear probe accuracy
   well above chance (with a shuffled-label control) says the encoder represents *task achievability*.
2. **Direction** — the success direction is the difference of class means (success − failure) of pooled
   latents; this is the axis we steer along.
3. **Steer** — add ``alpha * success_direction`` to a *failed* clip's latents and decode, asking whether
   the decoded behaviour moves toward success; alternatively interpolate a failed latent trajectory
   toward a matched successful one (:func:`analysis.steering.interpolate_trajectories`).

Reuses the same intervention/steering primitives as the physics steps. Success/failure labels arrive via
``category`` ("success"/"failure"), so :func:`analysis.latent_geometry.pooled_features` yields features +
labels directly (works for both DROID and the ``robot_toy`` fallback).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .latent_geometry import pooled_features


def detect_achievability(features: np.ndarray, labels: list[str], seed: int = 0) -> dict[str, float]:
    """Linear-probe accuracy for success-vs-failure, with a shuffled-label control."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {"accuracy": float("nan"), "ctrl_shuffled_accuracy": float("nan")}
    splits = int(max(2, min(5, counts.min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    acc = float(accuracy_score(y, cross_val_predict(model, features, y, cv=cv)))
    rng = np.random.default_rng(seed)
    ys = y[rng.permutation(len(y))]
    ctrl = float(accuracy_score(ys, cross_val_predict(model, features, ys, cv=cv)))
    return {"accuracy": acc, "ctrl_shuffled_accuracy": ctrl, "majority_rate": float(counts.max() / len(y))}


def success_direction(features: np.ndarray, labels: list[str]) -> np.ndarray:
    """Unit latent direction pointing failure -> success (difference of standardised class means)."""
    f = (features - features.mean(0, keepdims=True)) / (features.std(0, keepdims=True) + 1e-8)
    y = np.asarray(labels)
    d = f[y == "success"].mean(0) - f[y == "failure"].mean(0)
    return (d / (np.linalg.norm(d) + 1e-12)).astype(np.float32)


def achievability_report(latent_dir: str | Path, layer: int, seed: int = 0) -> dict:
    """Convenience: detection accuracy + success direction at one layer."""
    feats, cats = pooled_features(latent_dir, layer)
    return {
        "layer": int(layer),
        "detection": detect_achievability(feats, cats, seed=seed),
        "direction": success_direction(feats, cats),
        "n_success": int(sum(c == "success" for c in cats)),
        "n_failure": int(sum(c == "failure" for c in cats)),
    }
