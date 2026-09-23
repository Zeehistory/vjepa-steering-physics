"""Temporal velocity probe (Step 2, velocity-first).

The supervisor's key methodological point: dynamics live in the *temporal* axis, so a probe should
**not** collapse the ``(T'*Hp*Wp, D)`` token grid to a single ``(D,)`` clip vector. Instead we keep the
temporal structure as a ``(T', D)`` sequence (``T'`` ≈ 8 temporal patch positions for V-JEPA2, each a
spatial-pool over that timestep) and probe **velocity** from it. Three representations are compared so we
can see exactly how much the temporal axis buys:

* ``clip_pool``   — ``(D,)``     : mean over *all* tokens (the old 1x1024 baseline).
* ``temporal``    — ``(T'*D,)``  : the full ``8x1024`` sequence flattened (keeps per-timestep info).
* ``temporal_diff`` — ``((T'-1)*D,)`` : consecutive temporal differences ``z_{t+1}-z_t`` flattened.
  Velocity is a *rate of change*, so first differences are the natural linear feature for it; this tests
  whether the model represents motion as a displacement between adjacent temporal tokens.

Targets are the **clip-constant** velocity labels (our dataset has exactly constant velocity), probed as:

* ``vel``   — the signed velocity vector ``(vel_x, vel_y)``  (direction-aware),
* ``speed`` — the scalar speed ``|v|``,
* ``angle`` — direction, probed as ``(cos θ, sin θ)`` to avoid the 2π wrap discontinuity.

Each is fit with a **linear** (Ridge) and an **MLP** probe, with the two honest controls always on:
shuffled-latent and randomized-label (both must collapse to R²≈0). Reports per-layer R² so we can read
off *which layer* best encodes velocity and *which representation* (pooled vs temporal) is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..encoders.feature_extractor import LatentDataset


def _temporal_tokens(tokens: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """``(L, D)`` flat tokens -> ``(T', D)`` by spatial-pooling within each temporal patch position."""
    tp, hp, wp = grid
    n = tp * hp * wp
    return tokens[:n].reshape(tp, hp * wp, -1).mean(axis=1)  # (T', D)


def _clip_targets(state: np.ndarray, state_keys: list[str]) -> dict[str, np.ndarray]:
    """Clip-constant velocity targets from the per-frame state (constant-velocity dataset).

    Uses the temporal mean (labels are constant, so the mean is exact and robust to any rendering
    rounding). ``angle`` is returned as ``(cos, sin)`` to keep it continuous.
    """
    def col(name: str) -> float:
        return float(state[:, state_keys.index(name)].mean())

    vx, vy = col("obj0_vel_x"), col("obj0_vel_y")
    speed = col("obj0_speed")
    ang = col("obj0_angle")
    return {
        "vel": np.array([vx, vy], dtype=np.float32),
        "speed": np.array([speed], dtype=np.float32),
        "angle": np.array([np.cos(ang), np.sin(ang)], dtype=np.float32),
    }


def _build_reps(dataset: LatentDataset, layer: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-clip representations {name: (N, *)} and targets {name: (N, *)} for one layer."""
    reps: dict[str, list[np.ndarray]] = {"clip_pool": [], "temporal": [], "temporal_diff": []}
    tgts: dict[str, list[np.ndarray]] = {"vel": [], "speed": [], "angle": []}
    for i in range(len(dataset)):
        s = dataset[i]
        seq = _temporal_tokens(s["layers"][layer].numpy(), tuple(s["grid"]))  # (T', D)
        reps["clip_pool"].append(seq.mean(0))                                  # (D,)
        reps["temporal"].append(seq.reshape(-1))                               # (T'*D,)
        reps["temporal_diff"].append(np.diff(seq, axis=0).reshape(-1))         # ((T'-1)*D,)
        t = _clip_targets(s["state"].numpy(), s["state_keys"])
        for k, v in t.items():
            tgts[k].append(v)
    reps_arr = {k: np.stack(v, 0) for k, v in reps.items()}
    tgts_arr = {k: np.stack(v, 0) for k, v in tgts.items()}
    return reps_arr, tgts_arr


def _fit_eval(
    X: np.ndarray, Y: np.ndarray, kind: str, seed: int, shuffle_latent: bool, shuffle_labels: bool,
) -> dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    x, y = X.copy(), Y.copy()
    rng = np.random.default_rng(seed)
    if shuffle_latent:
        x = x[rng.permutation(len(x))]
    if shuffle_labels:
        y = y[rng.permutation(len(y))]
    if np.allclose(y.std(0).sum(), 0):
        return {"r2": 0.0, "rmse": 0.0}
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=seed)
    xs = StandardScaler().fit(xtr)
    xtr, xte = xs.transform(xtr), xs.transform(xte)
    ys = StandardScaler().fit(ytr)
    ytr_s = ys.transform(ytr)
    if kind == "linear":
        model: Any = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=1000,
                             early_stopping=True, n_iter_no_change=25, random_state=seed)
    model.fit(xtr, ytr_s)
    pred = ys.inverse_transform(np.atleast_2d(model.predict(xte)).reshape(len(xte), -1))
    return {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(((pred - yte) ** 2).mean())),
    }


def probe_velocity(
    latent_dir: str | Path,
    layers: list[int] | str = "all",
    seed: int = 0,
    output_csv: str | Path | None = None,
    representations: tuple[str, ...] = ("clip_pool", "temporal", "temporal_diff"),
    targets: tuple[str, ...] = ("vel", "speed", "angle"),
) -> list[dict[str, Any]]:
    """Temporal velocity probes (linear + MLP, with controls) per layer x representation x target."""
    dataset = LatentDataset(latent_dir, layers="all")
    available = dataset.available_layers()
    layer_list = available if layers == "all" else [int(x) for x in layers]

    records: list[dict[str, Any]] = []
    for layer in layer_list:
        reps, tgts = _build_reps(dataset, layer)
        # Evict the per-shard latent cache between layers: it holds every shard's full multi-layer
        # token tensors, so without clearing it the 100GB+ cache accumulates across layers and OOMs.
        dataset._shard_cache.clear()
        for rep in representations:
            X = reps[rep]
            for tgt in targets:
                Y = tgts[tgt]
                for kind in ("linear", "mlp"):
                    real = _fit_eval(X, Y, kind, seed, False, False)
                    sl = _fit_eval(X, Y, kind, seed, True, False)
                    sy = _fit_eval(X, Y, kind, seed, False, True)
                    records.append({
                        "layer": layer, "representation": rep, "target": tgt, "probe": kind,
                        "feat_dim": int(X.shape[1]),
                        "r2": round(real["r2"], 4), "rmse": round(real["rmse"], 6),
                        "ctrl_shuffled_latent_r2": round(sl["r2"], 4),
                        "ctrl_randomized_label_r2": round(sy["r2"], 4),
                    })

    if output_csv is not None and records:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
    return records
