"""Latent geometry: PCA, layerwise pooled features, and CKA matrices.

Helpers to turn a latent cache into feature matrices and to compute the geometric summaries used by the
manifold / dataset-shift experiments. Heavy nonlinear embeddings (UMAP/t-SNE) live in
``manifold_analysis`` so this module stays dependency-light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..encoders.feature_extractor import LatentDataset
from ..eval.latent_metrics import linear_cka


def pooled_features(latent_dir: str | Path, layer: int, pool: str = "mean") -> tuple[np.ndarray, list[str]]:
    """Return ``(N, D)`` clip-level pooled features for one layer plus the per-clip category labels."""
    ds = LatentDataset(latent_dir, layers=[layer])
    feats, cats = [], []
    for i in range(len(ds)):
        s = ds[i]
        tok = s["layers"][layer].numpy()
        feats.append(tok.mean(0) if pool == "mean" else tok.max(0))
        cats.append(s["category"])
    return np.stack(feats, 0), cats


def pca(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return projected coordinates ``(N, k)`` and the explained-variance ratio."""
    xc = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    coords = xc @ vt[:n_components].T
    var = (s**2) / (s**2).sum()
    return coords, var[:n_components]


def layerwise_cka_matrix(latent_dir: str | Path, layers: list[int]) -> np.ndarray:
    """CKA similarity matrix between the pooled features of each layer."""
    feats = {li: pooled_features(latent_dir, li)[0] for li in layers}
    n = len(layers)
    mat = np.zeros((n, n))
    for i, a in enumerate(layers):
        for j, b in enumerate(layers):
            mat[i, j] = linear_cka(feats[a], feats[b])
    return mat
