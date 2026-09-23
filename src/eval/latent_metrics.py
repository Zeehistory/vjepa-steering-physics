"""Latent-space analysis metrics.

* **CKA** (linear & RBF) — representational similarity between two sets of features (e.g. across
  datasets or layers).
* **Linear-vs-nonlinear decodability gap** — convenience wrapper interpreting probe results.
* **Intrinsic dimensionality** — participation ratio (PCA-eigenvalue based) and a TwoNN estimate.
* **SVCCA** — deferred stub (documented).

These operate on numpy feature matrices ``(N, D)``.
"""

from __future__ import annotations

import warnings

import numpy as np


def _center(gram: np.ndarray) -> np.ndarray:
    n = gram.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ gram @ h


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment in ``[0, 1]`` (1 = identical representations)."""
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    gx, gy = x @ x.T, y @ y.T
    cx, cy = _center(gx), _center(gy)
    hsic = (cx * cy).sum()
    denom = np.sqrt((cx * cx).sum() * (cy * cy).sum()) + 1e-12
    return float(hsic / denom)


def rbf_cka(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    def rbf_gram(a: np.ndarray) -> np.ndarray:
        sq = np.sum(a**2, 1)
        d = sq[:, None] + sq[None, :] - 2 * a @ a.T
        s = sigma or np.sqrt(np.median(d[d > 0]) / 2 + 1e-12)
        return np.exp(-d / (2 * s**2))

    cx, cy = _center(rbf_gram(x)), _center(rbf_gram(y))
    hsic = (cx * cy).sum()
    return float(hsic / (np.sqrt((cx * cx).sum() * (cy * cy).sum()) + 1e-12))


def participation_ratio(x: np.ndarray) -> float:
    """Intrinsic-dimensionality proxy: ``(Σλ)² / Σλ²`` of the feature covariance eigenvalues."""
    x = x - x.mean(0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    if eig.sum() == 0:
        return 0.0
    return float((eig.sum() ** 2) / (np.sum(eig**2) + 1e-12))


def twonn_intrinsic_dim(x: np.ndarray, frac: float = 0.9) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al., 2017)."""
    from scipy.spatial import cKDTree

    n = x.shape[0]
    if n < 10:
        return float("nan")
    tree = cKDTree(x)
    dist, _ = tree.query(x, k=3)
    r1, r2 = dist[:, 1], dist[:, 2]
    valid = (r1 > 0) & (r2 > 0)
    mu = np.sort(r2[valid] / r1[valid])
    f = np.arange(1, len(mu) + 1) / len(mu)
    cut = int(frac * len(mu))
    x_ = np.log(mu[:cut])
    y_ = -np.log(1 - f[:cut] + 1e-12)
    d = float(np.sum(x_ * y_) / (np.sum(x_ * x_) + 1e-12))
    return d


def svcca(x: np.ndarray, y: np.ndarray) -> float:
    """SVCCA similarity (deferred stub)."""
    warnings.warn("SVCCA not implemented; use linear_cka/rbf_cka. Returning nan.", stacklevel=2)
    return float("nan")
