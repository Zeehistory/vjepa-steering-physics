"""Circular population-code detection for direction variables (Experiment 5).

Motion *direction* is a circular variable (θ ∈ [-π, π)). If the latent space represents it with a
population code, we expect a 2D subspace where the representation traces a closed loop as θ sweeps the
circle. This module tests for that structure by fitting the latent projection onto ``[cos θ, sin θ]``
and measuring how well a circle is recovered.

A high circular-fit R² (with a much lower R² for a shuffled control) is evidence of a circular code —
distinct from merely linearly decoding a scalar.
"""

from __future__ import annotations

import numpy as np


def circular_code_score(features: np.ndarray, theta: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Test whether ``features`` ``(N, D)`` encode angle ``theta`` ``(N,)`` as a circular code.

    Method: regress features onto ``[cos θ, sin θ]`` (2D target embedded on the unit circle), then
    measure canonical correlation between the predicted 2D coordinates and the true circle points.
    Returns the mean canonical correlation and a shuffled-control score.
    """
    from sklearn.cross_decomposition import CCA

    circle = np.stack([np.cos(theta), np.sin(theta)], 1)  # (N, 2)

    def cca_corr(f: np.ndarray, c: np.ndarray) -> float:
        cca = CCA(n_components=2)
        fc, cc = cca.fit_transform(f, c)
        corrs = [np.corrcoef(fc[:, k], cc[:, k])[0, 1] for k in range(2)]
        return float(np.nanmean(np.abs(corrs)))

    real = cca_corr(features, circle)
    rng = np.random.default_rng(seed)
    shuffled = cca_corr(features, circle[rng.permutation(len(circle))])
    return {"circular_cca": real, "ctrl_shuffled_cca": shuffled, "gap": real - shuffled}
