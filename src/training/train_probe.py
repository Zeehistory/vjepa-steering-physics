"""Layerwise probing: how decodable is each physical variable from each encoder layer?

This implements **Experiment 1**. For every requested encoder layer we build a regression dataset that
maps the (spatially-pooled) latent at each temporal position to the ground-truth physical state at the
corresponding frame, then fit:

* a **linear** probe (Ridge) — measures *linearly accessible* information,
* an **MLP** probe (small MLPRegressor) — measures *nonlinearly accessible* information,

and report R² / RMSE per physical-variable group (position, velocity, acceleration, gravity,
collision). Two **controls** are always computed so claims stay honest:

* **shuffled-latent** — rows permuted vs. labels; decodability must collapse to ~0.
* **randomized-label** — labels permuted; must collapse to ~0.

Results are returned as a list of records and written to ``layerwise_decodability.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..encoders.feature_extractor import LatentDataset

# Variable groups reported in the decodability table (substring match on state key names).
VARIABLE_GROUPS = ["pos", "vel", "acc", "radius", "gravity", "collision_event", "visible"]


def _pool_layer_to_frames(
    tokens: np.ndarray, grid: tuple[int, int, int], num_frames: int
) -> np.ndarray:
    """``(L, D)`` tokens -> ``(num_frames, D)`` by spatial-pooling per temporal position then aligning."""
    tp, hp, wp = grid
    n = tp * hp * wp
    tokens = tokens[:n].reshape(tp, hp * wp, -1).mean(axis=1)  # (T', D)
    idx = np.round(np.linspace(0, tp - 1, num_frames)).astype(int)
    return tokens[idx]  # (num_frames, D)


def _build_xy(dataset: LatentDataset, layer: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Stack (frame-aligned latent, frame state) pairs across the dataset for one layer."""
    xs, ys = [], []
    state_keys: list[str] = []
    mask: np.ndarray | None = None
    for i in range(len(dataset)):
        s = dataset[i]
        state_keys = s["state_keys"]
        frames = s["state"].shape[0]
        feat = _pool_layer_to_frames(s["layers"][layer].numpy(), tuple(s["grid"]), frames)
        xs.append(feat)
        ys.append(s["state"].numpy())
        if mask is None:
            mask = s["state_mask"].numpy()
    X = np.concatenate(xs, 0)
    Y = np.concatenate(ys, 0)
    return X, Y, state_keys, mask if mask is not None else np.ones(Y.shape[-1])


def _group_cols(state_keys: list[str], group: str, mask: np.ndarray) -> list[int]:
    return [i for i, k in enumerate(state_keys) if group in k and mask[i] > 0]


def _fit_eval(
    X: np.ndarray, Y: np.ndarray, cols: list[int], kind: str, seed: int, shuffle_latent: bool,
    shuffle_labels: bool,
) -> dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    y = Y[:, cols]
    x = X.copy()
    rng = np.random.default_rng(seed)
    if shuffle_latent:
        x = x[rng.permutation(len(x))]
    if shuffle_labels:
        y = y[rng.permutation(len(y))]
    if np.allclose(y.std(0).sum(), 0):  # constant target -> undefined R²
        return {"r2": 0.0, "rmse": float(np.sqrt(((y - y.mean(0)) ** 2).mean()))}
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=seed)
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    # Standardize targets too: physical quantities have tiny, heterogeneous scales (gravity ~1e-3),
    # which otherwise make the MLP diverge to absurd negative R². Metrics are reported on the original
    # scale via inverse_transform.
    yscaler = StandardScaler().fit(ytr)
    ytr_s = yscaler.transform(ytr)
    if kind == "linear":
        model: Any = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(256,), alpha=1e-3, max_iter=1000, early_stopping=True,
            n_iter_no_change=25, random_state=seed,
        )
    model.fit(xtr, ytr_s)
    pred_s = model.predict(xte)
    pred = yscaler.inverse_transform(pred_s if pred_s.ndim == 2 else pred_s[:, None])
    yte = yte if yte.ndim == 2 else yte[:, None]
    return {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(((pred - yte) ** 2).mean())),
    }


def probe_layers(
    latent_dir: str | Path,
    layers: list[int] | str = "all",
    seed: int = 0,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run linear + MLP probes (with controls) for each layer × variable group."""
    dataset = LatentDataset(latent_dir, layers="all")
    available = dataset.available_layers()
    layer_list = available if layers == "all" else [int(x) for x in layers]

    records: list[dict[str, Any]] = []
    for layer in layer_list:
        X, Y, state_keys, mask = _build_xy(dataset, layer)
        for group in VARIABLE_GROUPS:
            cols = _group_cols(state_keys, group, mask)
            if not cols:
                continue
            for kind in ("linear", "mlp"):
                real = _fit_eval(X, Y, cols, kind, seed, False, False)
                shuf_lat = _fit_eval(X, Y, cols, kind, seed, True, False)
                shuf_lab = _fit_eval(X, Y, cols, kind, seed, False, True)
                records.append({
                    "layer": layer, "variable": group, "probe": kind,
                    "r2": round(real["r2"], 4), "rmse": round(real["rmse"], 5),
                    "ctrl_shuffled_latent_r2": round(shuf_lat["r2"], 4),
                    "ctrl_randomized_label_r2": round(shuf_lab["r2"], 4),
                })

    if output_csv is not None and records:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    return records
