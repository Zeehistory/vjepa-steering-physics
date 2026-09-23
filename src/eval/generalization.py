"""Dataset-shift / generalization metrics in latent space (Experiment 4).

Given pooled latent feature matrices from two sources (e.g. synthetic physics vs. Physics-IQ), quantify
how far apart their representations are with distribution distances. Used to measure — not guess — how
far physics-benchmark data sits from other video distributions in V-JEPA latent space.

* **linear MMD²** (RBF kernel) — maximum mean discrepancy.
* **Fréchet distance** — Gaussian approximation (mean/cov), the FID-style latent distance.
* **CKA** is available in ``latent_metrics`` for similarity.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Squared MMD with an RBF kernel between feature sets ``(Nx, D)`` and ``(Ny, D)``."""
    def k(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
        sa = np.sum(a**2, 1)[:, None]
        sb = np.sum(b**2, 1)[None, :]
        d = sa + sb - 2 * a @ b.T
        return np.exp(-d / (2 * s**2))

    if sigma is None:
        cat = np.concatenate([x, y], 0)
        d = np.sum(cat**2, 1)[:, None] + np.sum(cat**2, 1)[None, :] - 2 * cat @ cat.T
        sigma = float(np.sqrt(np.median(d[d > 0]) / 2 + 1e-12))
    kxx, kyy, kxy = k(x, x, sigma), k(y, y, sigma), k(x, y, sigma)
    m, n = len(x), len(y)
    return float(kxx.sum() / (m * m) + kyy.sum() / (n * n) - 2 * kxy.sum() / (m * n))


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """FID-style Fréchet distance between Gaussian approximations of two feature sets."""
    mu1, mu2 = x.mean(0), y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


def dataset_shift_report(features_by_source: dict[str, np.ndarray]) -> list[dict]:
    """Pairwise shift metrics between every pair of sources."""
    sources = sorted(features_by_source)
    records = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a, b = sources[i], sources[j]
            records.append({
                "source_a": a, "source_b": b,
                "mmd2": round(rbf_mmd2(features_by_source[a], features_by_source[b]), 6),
                "frechet": round(frechet_distance(features_by_source[a], features_by_source[b]), 4),
            })
    return records
