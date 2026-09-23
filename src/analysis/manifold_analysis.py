"""Manifold structure analysis (Experiment 5).

Tools to test whether physical variables occupy smooth, low-dimensional subspaces:

* nonlinear embeddings (UMAP if installed, else PCA fallback — never silently fails),
* intrinsic-dimensionality estimates (re-exported from ``eval.latent_metrics``),
* a generic helper to correlate a latent direction with a scalar physical variable.

t-SNE/UMAP are visualization aids only; quantitative claims should lean on probes + intrinsic-dim, per
the project's research-discipline guidance.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..eval.latent_metrics import participation_ratio, twonn_intrinsic_dim

__all__ = ["embed_2d", "participation_ratio", "twonn_intrinsic_dim", "variable_axis_alignment"]


def embed_2d(x: np.ndarray, method: str = "umap", seed: int = 0) -> np.ndarray:
    """2D embedding of ``(N, D)`` features. Falls back to PCA if UMAP/t-SNE are unavailable."""
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
        except Exception:
            warnings.warn("UMAP unavailable (pip install -e .[extras]); falling back to PCA.", stacklevel=2)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            return TSNE(n_components=2, random_state=seed, init="pca").fit_transform(x)
        except Exception:
            warnings.warn("t-SNE failed; falling back to PCA.", stacklevel=2)
    from .latent_geometry import pca

    return pca(x, 2)[0]


def variable_axis_alignment(features: np.ndarray, variable: np.ndarray) -> dict[str, float]:
    """Fit a 1D linear axis predicting a scalar physical variable from features.

    Returns the fraction of variance explained (R²) and the smoothness of the projection — a high R²
    with low residual indicates the variable lies along a near-linear latent axis (e.g. a gravity axis).
    """
    from sklearn.linear_model import LinearRegression

    f = features - features.mean(0, keepdims=True)
    reg = LinearRegression().fit(f, variable)
    pred = reg.predict(f)
    ss_res = float(((variable - pred) ** 2).sum())
    ss_tot = float(((variable - variable.mean()) ** 2).sum() + 1e-12)
    return {"r2": 1.0 - ss_res / ss_tot, "axis_norm": float(np.linalg.norm(reg.coef_))}
