"""Latent steering: edit a latent along a discovered physical direction and decode the result (Step 3).

Builds on :mod:`analysis.intervention` (which discovers a direction and tests decoded-state
controllability) to support the *real-video* steering loop: learn a direction for a physical quantity
on a labelled (synthetic) cache, then add ``alpha * direction`` to the latents of a real Physics-IQ
clip and **decode the steered latent to pixels** with the trained transformer decoder.

Because real videos have no ground-truth state, controllability is verified two ways:

* **independent readout** — a regressor fit on the *labelled* cache predicts the quantity from pooled
  latents; applied to the steered real latents it should move monotonically with ``alpha`` (this is
  also the Step-3 *transfer* test: does a synthetic direction carry to real data?);
* **decoded frames** — eyeballed (and, where the decoder has a state head, read out directly).

The discipline from ``intervention`` carries over: a real direction produces a smooth, monotonic change;
an ineffective one does not.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..encoders.feature_extractor import LatentDataset
from .intervention import apply_intervention, discover_direction


def clip_features_and_scalar(
    latent_dir: str | Path, layer: int, group: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clip-level pooled features ``(N, D)`` and a scalar magnitude of ``group`` per clip ``(N,)``.

    The scalar is the per-frame L2 magnitude over the state columns whose name contains ``group``
    (masked to valid columns), averaged over frames — a sign-free target suitable for a 1-D direction
    (e.g. speed for ``vel``, gravity for ``gravity``). Only clips with valid state for the group count.
    """
    ds = LatentDataset(latent_dir, layers=[layer])
    feats, scal, ids = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        keys = s["state_keys"]
        mask = s["state_mask"].numpy()
        cols = [j for j, k in enumerate(keys) if group in k and (j >= len(mask) or mask[j] > 0)]
        if not cols:
            continue
        state = s["state"].numpy()[:, cols]               # (T, |cols|)
        mag = float(np.sqrt((state ** 2).sum(1)).mean())  # mean per-frame magnitude
        feats.append(s["layers"][layer].numpy().mean(0))  # (D,)
        scal.append(mag)
        ids.append(s["id"])
    if not feats:
        raise ValueError(f"No clips with valid state for group '{group}' in {latent_dir}.")
    return np.stack(feats, 0), np.asarray(scal, dtype=np.float32), ids


def fit_readout(features: np.ndarray, scalar: np.ndarray, alpha: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a standardised Ridge readout ``features -> scalar``; returns a predict function ``(M,D)->(M,)``.

    Used as an *independent* verification of steering on data without ground-truth labels.
    """
    from sklearn.linear_model import Ridge

    mu, sd = features.mean(0, keepdims=True), features.std(0, keepdims=True) + 1e-8
    model = Ridge(alpha=alpha).fit((features - mu) / sd, scalar)

    def predict(x: np.ndarray) -> np.ndarray:
        return model.predict((x - mu) / sd)

    return predict


@torch.no_grad()
def decode_intervention(
    decoder: Any,
    latents: dict[int, torch.Tensor],
    grid: tuple[int, int, int],
    direction: torch.Tensor,
    layer: int,
    alphas: list[float],
) -> dict[float, Any]:
    """Decode the latents at each intervention strength. Returns ``{alpha: DecoderOutput}``.

    The decoder is run on ``latents + alpha * direction`` (applied to ``layer``); the caller pulls
    ``.frames`` and/or ``.state`` from each output.
    """
    out: dict[float, Any] = {}
    for a in alphas:
        perturbed = apply_intervention(latents, direction, a, layer)
        out[a] = decoder(perturbed, grid)
    return out


@torch.no_grad()
def readout_along_direction(
    latents: dict[int, torch.Tensor],
    direction: torch.Tensor,
    layer: int,
    alphas: list[float],
    readout: Callable[[np.ndarray], np.ndarray],
) -> list[dict[str, float]]:
    """Independent-readout controllability curve: predicted quantity vs ``alpha`` (latent-space only).

    Pools the steered latents and applies the fitted ``readout`` — no decoder needed, so it works on
    real data with no ground-truth state.
    """
    rows = []
    base = latents[layer]  # (B, L, D)
    d = direction.view(1, 1, -1).to(base.device)
    for a in alphas:
        pooled = (base + a * d).mean(dim=(0, 1)).cpu().numpy()[None, :]  # (1, D)
        rows.append({"alpha": float(a), "readout": float(readout(pooled)[0])})
    return rows


def steer_to_target(
    latents: dict[int, torch.Tensor],
    direction: torch.Tensor,
    layer: int,
    target: float,
    readout: Callable[[np.ndarray], np.ndarray],
    search: tuple[float, float] = (-6.0, 6.0),
    steps: int = 49,
) -> tuple[float, dict[int, torch.Tensor]]:
    """Find ``alpha`` along ``direction`` whose independent readout best matches ``target``.

    Grid-searches ``alpha`` over ``search`` and returns ``(best_alpha, steered_latents)``. Concrete,
    decoder-free counterpart to the gradient-based ideal — sufficient for 1-D physical directions.
    """
    alphas = list(np.linspace(search[0], search[1], steps))
    rows = readout_along_direction(latents, direction, layer, alphas, readout)
    best = min(rows, key=lambda r: abs(r["readout"] - target))
    return best["alpha"], apply_intervention(latents, direction, best["alpha"], layer)


def interpolate_trajectories(
    latents_fail: dict[int, torch.Tensor],
    latents_success: dict[int, torch.Tensor],
    steps: int = 8,
) -> list[dict[int, torch.Tensor]]:
    """Linearly interpolate between a failed and a successful latent trajectory (robotics use, Step 4).

    Returns ``steps`` latent dicts from ``latents_fail`` (t=0) to ``latents_success`` (t=1). Decoding
    these shows whether the latent path from failure to success passes through plausible intermediate
    states — the bridge from "detect" to "steer" for the robotics question.
    """
    ts = np.linspace(0.0, 1.0, steps)
    keys = set(latents_fail) & set(latents_success)
    return [{k: (1 - t) * latents_fail[k] + t * latents_success[k] for k in keys} for t in ts]


def category_steering_direction(
    directions_npz: str | Path, layer: int, from_category: str, to_category: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the raw-latent steering direction ``w_to - w_from`` from a category-probe ``.npz``.

    The ``.npz`` is produced by :mod:`scripts.probe_categories` and holds, per layer, the linear probe's
    per-class weight vectors (``layer{L}_coef``), the class order (``layer{L}_classes``), and the feature
    mean/std the probe saw (``layer{L}_std``). If the probe was standardized, the weight vectors live in
    z-scored space and the equivalent *raw-latent* direction is ``coef / std`` (because a standardized
    score ``w·(x-mu)/sigma`` equals ``(w/sigma)·x + const``). We therefore return

        d_raw = (w_to - w_from) / std        (std = 1 for a --no_standardize probe),

    normalized to unit L2 — the direction to add to raw latents so a clip reads more like ``to_category``
    and less like ``from_category`` (e.g. fluid -> solid). Returns ``(unit_direction (D,), info)``.
    """
    data = np.load(Path(directions_npz), allow_pickle=False)
    classes = [str(c) for c in data[f"layer{layer}_classes"]]
    coef = data[f"layer{layer}_coef"]  # (C, D)
    std = data[f"layer{layer}_std"] if f"layer{layer}_std" in data else np.ones(coef.shape[1], np.float32)
    for c in (from_category, to_category):
        if c not in classes:
            raise ValueError(f"Category '{c}' not among probe classes {classes} at layer {layer}.")
    w_to = coef[classes.index(to_category)]
    w_from = coef[classes.index(from_category)]
    d_raw = (w_to - w_from) / (std + 1e-8)
    norm = float(np.linalg.norm(d_raw) + 1e-12)
    info = {"classes": classes, "from_category": from_category, "to_category": to_category,
            "standardized": bool(data["standardized"]) if "standardized" in data else None,
            "raw_norm": norm}
    return (d_raw / norm).astype(np.float32), info


def category_mean_direction(
    latent_dir: str | Path, layer: int, from_category: str, to_category: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Difference-of-class-means steering direction at ``layer``: ``mean(to) - mean(from)``.

    Unlike the probe's *discriminative* weight vector (max-margin, often along low-variance nuisance
    dims), this is the actual **translation** in latent space from the ``from_category`` cloud's centroid
    to the ``to_category`` cloud's — the vector a generative decoder is far more likely to render. Both
    centroids are clip-level mean-pooled latents from ``latent_dir``. Returns ``(unit_direction, info)``.
    """
    ds = LatentDataset(latent_dir, layers=[layer])
    pos, neg = [], []
    for i in range(len(ds)):
        s = ds[i]
        f = s["layers"][layer].numpy().mean(0)
        if s["category"] == to_category:
            pos.append(f)
        elif s["category"] == from_category:
            neg.append(f)
    if not pos or not neg:
        raise ValueError(f"Need both '{from_category}' ({len(neg)}) and '{to_category}' ({len(pos)}) "
                         f"clips in {latent_dir} at layer {layer}.")
    d = np.mean(pos, 0) - np.mean(neg, 0)
    norm = float(np.linalg.norm(d) + 1e-12)
    info = {"from_category": from_category, "to_category": to_category, "n_from": len(neg),
            "n_to": len(pos), "raw_norm": norm, "method": "diff_means"}
    return (d / norm).astype(np.float32), info


def category_readout(
    latent_dir: str | Path, layer: int, positive_category: str, exclude_ids: set[str] | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Independent probe-free check: fit a standardized one-vs-rest logistic for ``positive_category``.

    Returns ``predict(x (M,D)) -> P(positive_category) (M,)``. Fit on the *target* cache's pooled latents
    (optionally excluding the clips being steered, to keep the readout independent of them), so tracking
    this probability along the ``alpha`` sweep is a genuine — not circular — controllability signal.
    """
    from sklearn.linear_model import LogisticRegression

    ds = LatentDataset(latent_dir, layers=[layer])
    feats, y = [], []
    for i in range(len(ds)):
        s = ds[i]
        if exclude_ids and s["id"] in exclude_ids:
            continue
        feats.append(s["layers"][layer].numpy().mean(0))
        y.append(1 if s["category"] == positive_category else 0)
    X = np.stack(feats, 0)
    yv = np.asarray(y)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    model = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit((X - mu) / sd, yv)
    pos_col = list(model.classes_).index(1)

    def predict(x: np.ndarray) -> np.ndarray:
        return model.predict_proba((x - mu) / sd)[:, pos_col]

    return predict


def velocity_components(
    latent_dir: str | Path, layer: int
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Clip-pooled latents ``(N, D)`` and the *signed* velocity targets per clip, for the moving-ball cache.

    Returns ``(features, {"vel_x", "vel_y", "speed": (N,)}, ids)``. Unlike :func:`clip_features_and_scalar`
    (which returns only the sign-free magnitude), this exposes the signed components so a *directional*
    velocity axis (e.g. "more rightward motion") can be learned, not just "more speed".
    """
    ds = LatentDataset(latent_dir, layers=[layer])
    feats, vx, vy, sp, ids = [], [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        keys = s["state_keys"]
        st = s["state"].numpy()
        def col(name: str) -> float:
            return float(st[:, keys.index(name)].mean())
        feats.append(s["layers"][layer].numpy().mean(0))
        vx.append(col("obj0_vel_x")); vy.append(col("obj0_vel_y")); sp.append(col("obj0_speed"))
        ids.append(s["id"])
    return (np.stack(feats, 0),
            {"vel_x": np.asarray(vx, np.float32), "vel_y": np.asarray(vy, np.float32),
             "speed": np.asarray(sp, np.float32)},
            ids)


def discover_velocity_direction(
    latent_dir: str | Path, layer: int, target: str = "speed", method: str = "regression",
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Learn a unit latent direction + independent readout for a signed velocity ``target``.

    ``target`` is one of ``speed`` (sign-free magnitude), ``vel_x`` or ``vel_y`` (signed components).
    Returns ``(direction (D,), readout_fn, info)``. The readout is fit on the same labelled cache and is
    used as the *non-circular* latent-space check that the decoded edit moved velocity (the pixel-level
    centroid tracker is the independent visual check).
    """
    feats, comps, ids = velocity_components(latent_dir, layer)
    if target not in comps:
        raise ValueError(f"target must be one of {list(comps)}, got '{target}'.")
    y = comps[target]
    direction = discover_direction(feats, y, method=method)
    readout = fit_readout(feats, y)
    info = {"target": target, "method": method, "n": len(y),
            "y_min": float(y.min()), "y_max": float(y.max()), "raw_norm": 1.0}
    return direction, readout, info


def discover_quantity_direction(
    latent_dir: str | Path, layer: int, group: str, method: str = "regression"
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Convenience: learn a unit direction *and* an independent readout for ``group`` from a cache.

    Returns ``(direction (D,), readout_fn)`` — the two objects Step 3 steering needs.
    """
    feats, scal, _ = clip_features_and_scalar(latent_dir, layer, group)
    direction = discover_direction(feats, scal, method=method)
    readout = fit_readout(feats, scal)
    return direction, readout
