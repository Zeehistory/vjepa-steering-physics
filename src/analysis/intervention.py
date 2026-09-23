"""Latent intervention: discover a physical direction and measure decoded effect (Experiment 6).

Pipeline:

1. **Discover** a candidate latent direction for a scalar physical variable (e.g. gravity, speed) by
   fitting a linear axis on pooled latents (difference-of-means or regression coefficient).
2. **Intervene** by adding ``alpha * direction`` to the latents.
3. **Decode** the perturbed latents and measure whether the decoded physical state changes in the
   predicted way (monotonic in ``alpha``).

This is deliberately careful: a *real* steering direction should produce a monotonic, controlled change
in the decoded variable; an ineffective one will not. The caller supplies a trained decoder.
"""

from __future__ import annotations

import numpy as np
import torch


def discover_direction(features: np.ndarray, variable: np.ndarray, method: str = "regression") -> np.ndarray:
    """Return a unit latent direction ``(D,)`` associated with increasing ``variable``."""
    f = features - features.mean(0, keepdims=True)
    if method == "diff_means":
        hi = f[variable >= np.median(variable)].mean(0)
        lo = f[variable < np.median(variable)].mean(0)
        d = hi - lo
    else:  # regression
        from sklearn.linear_model import Ridge

        d = Ridge(alpha=1.0).fit(f, variable).coef_
    norm = np.linalg.norm(d) + 1e-12
    return (d / norm).astype(np.float32)


@torch.no_grad()
def apply_intervention(
    latents: dict[int, torch.Tensor],
    direction: torch.Tensor,
    alpha: float,
    layer: int,
) -> dict[int, torch.Tensor]:
    """Return a copy of ``latents`` with ``alpha * direction`` added to every token of ``layer``."""
    out = {k: v.clone() for k, v in latents.items()}
    out[layer] = out[layer] + alpha * direction.view(1, 1, -1).to(out[layer].device)
    return out


@torch.no_grad()
def apply_intervention_multi(
    latents: dict[int, torch.Tensor],
    directions: dict[int, torch.Tensor],
    alpha: float,
) -> dict[int, torch.Tensor]:
    """Add ``alpha * directions[L]`` to every token of each layer ``L`` in ``directions``.

    Generalises :func:`apply_intervention` to steer several layers at once (the per-layer vectors are
    expected to already carry their intended magnitude — e.g. scaled by each layer's token norm).
    Layers absent from ``directions`` are passed through unchanged so the decoder still sees them.
    """
    out = {k: v.clone() for k, v in latents.items()}
    for layer, d in directions.items():
        out[layer] = out[layer] + alpha * d.view(1, 1, -1).to(out[layer].device)
    return out


@torch.no_grad()
def intervention_sweep(
    decoder,
    latents: dict[int, torch.Tensor],
    grid: tuple[int, int, int],
    direction: torch.Tensor,
    layer: int,
    alphas: list[float],
    readout: str,
    state_keys: list[str],
) -> list[dict[str, float]]:
    """Sweep ``alpha`` and record the decoded readout variable, to test monotonic controllability."""
    cols = [i for i, k in enumerate(state_keys) if readout in k]
    results = []
    for a in alphas:
        perturbed = apply_intervention(latents, direction, a, layer)
        out = decoder(perturbed, grid)
        if out.state is None:
            raise ValueError("intervention_sweep needs a decoder that outputs state (mode C / co-trained).")
        val = float(out.state[..., cols].mean()) if cols else float("nan")
        results.append({"alpha": a, "readout": val})
    return results
