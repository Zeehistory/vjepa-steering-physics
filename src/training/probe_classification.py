"""Category-structure probing: are physics *categories* linearly separable in the latent space?

This implements **Step 1** of the physics-steering roadmap. Physics-IQ ships only coarse category
labels (solid mechanics, fluids, optics, ...) and *no* per-frame numeric physics state, so we cannot
regress quantities here (that is :mod:`train_probe`, Step 2). Instead we ask a structural question:

* Can a **linear** classifier separate the categories from clip-level pooled latents? If yes, each
  category occupies a (linearly) distinct region — and the classifier's per-class weight vector is a
  *candidate steering direction* we reuse downstream (Step 3).
* Does an **MLP** classifier do better? The linear-vs-MLP gap measures how much category information is
  present but *non-linearly* entangled.

Two methodological points that matter for Physics-IQ specifically:

* **Scenario-grouped CV.** Each scenario appears as ~6 near-duplicate clips (3 perspectives x 2 takes).
  Random CV would leak a scenario across train/test, so the probe memorises scenario *appearance* and
  accuracy is inflated. We use scenario-grouped folds (StratifiedGroupKFold) so we measure
  generalisation to *unseen scenarios*.
* **Class imbalance / tiny classes.** Categories with fewer than ``min_scenarios_per_class`` distinct
  scenarios cannot support valid grouped CV (Physics-IQ: magnetism=2, thermodynamics=3 scenarios) and
  are dropped — but reported, never silently. ``majority_rate`` is the honest baseline to beat.

Controls: a **scenario-level shuffled-label** control (permute the category assigned to each whole
scenario) must collapse to ~majority rate; and a pixel/appearance baseline (mean colour) plus the
layerwise sweep test whether separability is physics (rises with depth) or appearance (flat near pixel).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..data.physics_iq_categories import scenario_for_id
from ..encoders.feature_extractor import LatentDataset


def _load_pooled(latent_dir: str | Path, layer: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clip-level mean-pooled latents ``(N, D)``, category labels, and sample ids for one layer."""
    ds = LatentDataset(latent_dir, layers=[layer])
    X, cats, ids = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        X.append(s["layers"][layer].numpy().mean(0))
        cats.append(s["category"])
        ids.append(s["id"])
    return np.stack(X, 0), np.asarray(cats), ids


def _load_pixels(latent_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Clip-level mean-pooled *pixel* features (appearance baseline), labels, and ids."""
    ds = LatentDataset(latent_dir, layers="all")
    X, cats, ids = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        X.append(s["frames"].numpy().mean(axis=(0, 2, 3)))  # mean colour per channel
        cats.append(s["category"])
        ids.append(s["id"])
    return np.stack(X, 0), np.asarray(cats), ids


def _filter_classes(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, min_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Drop classes with fewer than ``min_groups`` distinct scenarios (can't support grouped CV)."""
    keep, dropped = [], {}
    for c in sorted(set(y.tolist())):
        ng = len(set(groups[y == c].tolist()))
        (keep.append(c) if ng >= min_groups else dropped.update({c: ng}))
    mask = np.isin(y, keep)
    return X[mask], y[mask], groups[mask], keep, dropped


def _grouped_n_splits(y: np.ndarray, groups: np.ndarray, cap: int = 5) -> int:
    """Largest valid fold count: bounded by the class with the fewest distinct scenarios."""
    per_class = [len(set(groups[y == c].tolist())) for c in set(y.tolist())]
    return int(max(2, min(cap, min(per_class))))


def _group_shuffle(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Scenario-level label shuffle: permute the category assigned to each whole scenario."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    glabel = np.array([y[groups == g][0] for g in uniq])
    mapping = dict(zip(uniq, glabel[rng.permutation(len(uniq))], strict=False))
    return np.array([mapping[g] for g in groups])


def _predict_oof(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: int):
    """Out-of-fold predictions with scenario-grouped CV (falls back to GroupKFold, then plain)."""
    from sklearn.model_selection import (
        GroupKFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        cross_val_predict,
    )

    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return cross_val_predict(model, X, y, cv=cv)
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return cross_val_predict(model, X, y, cv=cv, groups=groups)
    except Exception:
        cv = GroupKFold(n_splits=n_splits)
        return cross_val_predict(model, X, y, cv=cv, groups=groups)


def _make_estimator(kind: str, seed: int):
    """A single linear/MLP estimator (no scaler — scaling is composed in by ``_make_model``)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if kind == "linear":
        # Raw (unstandardized) latents can be ill-conditioned, so allow more iterations regardless.
        return LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    return MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=600, random_state=seed,
                         early_stopping=True)


def _make_model(kind: str, seed: int, standardize: bool):
    """Estimator, optionally preceded by a z-score :class:`StandardScaler`.

    ``standardize=False`` feeds *raw* latent values to the classifier. This is the un-normalized probe:
    it answers "which raw subspace encodes the category" without the per-dimension rescaling that
    StandardScaler applies (which can distort the geometry, inflating low-variance dims). The standardized
    and raw weight vectors are related by ``w_raw = w_std / sigma`` (see ``feat_std`` saved with the
    directions), so a standardized run can also be inverse-mapped into raw space after the fact.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    est = _make_estimator(kind, seed)
    return make_pipeline(StandardScaler(), est) if standardize else make_pipeline(est)


def _fit_eval_clf(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, kind: str, seed: int, n_splits: int,
    shuffle_labels: bool, standardize: bool = True,
) -> dict[str, Any]:
    """Scenario-grouped CV accuracy + macro-F1, out-of-fold confusion, and (linear) weight directions."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    if shuffle_labels:
        y = _group_shuffle(y, groups, seed) if groups is not None else \
            y[np.random.default_rng(seed).permutation(len(y))]

    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "classes": classes, "confusion": None, "coef": None}

    model = _make_model(kind, seed, standardize)
    pred = _predict_oof(model, X, y, groups, n_splits, seed)
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average="macro"))
    cm = confusion_matrix(y, pred, labels=classes)

    coef = None
    if kind == "linear" and not shuffle_labels:
        full = _make_model("linear", seed, standardize).fit(X, y)
        coef = full.named_steps["logisticregression"].coef_.astype(np.float32)
    return {"accuracy": acc, "macro_f1": f1, "classes": classes, "confusion": cm, "coef": coef}


def _filter_to_categories(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, categories: list[str] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only rows whose label is in ``categories`` (explicit allow-list; ``None`` keeps all)."""
    if not categories:
        return X, y, groups
    mask = np.isin(y, list(categories))
    return X[mask], y[mask], groups[mask]


def classify_categories(
    latent_dir: str | Path,
    layers: list[int] | str = "all",
    seed: int = 0,
    output_csv: str | Path | None = None,
    pixel_baseline: bool = True,
    min_scenarios_per_class: int = 4,
    group_by_scenario: bool = True,
    standardize: bool = True,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run linear + MLP category classifiers per layer with scenario-grouped CV + controls.

    ``standardize`` toggles the per-dimension z-scoring (StandardScaler) in front of every classifier;
    ``False`` runs the probe on raw latent values. ``categories`` is an explicit allow-list of category
    slugs to investigate (e.g. drop thermodynamics/magnetism, which have too few scenarios for valid
    grouped CV anyway). ``None`` keeps every category that survives ``min_scenarios_per_class``.

    Returns ``records`` (rows per layer x probe), ``confusions`` (per-layer linear out-of-fold matrix),
    ``directions`` (per-layer linear per-class weight vectors + the feature mean/std used, so the raw
    steering direction is recoverable as ``coef / std`` — Step-3 candidates), and ``meta`` (kept/dropped
    categories, fold count, majority baseline, standardize/category settings).
    """
    dataset = LatentDataset(latent_dir, layers="all")
    available = dataset.available_layers()
    layer_list = available if layers == "all" else [int(x) for x in layers]

    records: list[dict[str, Any]] = []
    confusions: dict[int, dict[str, Any]] = {}
    directions: dict[int, dict[str, Any]] = {}
    meta: dict[str, Any] = {}

    def _run(X, y, groups, layer_tag, probe_prefix=""):
        X, y, groups = _filter_to_categories(X, y, groups, categories)
        if group_by_scenario:
            X, y, groups, keep, dropped = _filter_classes(X, y, groups, min_scenarios_per_class)
            n_splits = _grouped_n_splits(y, groups) if len(set(y.tolist())) >= 2 else 2
        else:
            groups, keep, dropped, n_splits = None, sorted(set(y.tolist())), {}, 5
        if not meta:
            _, counts = np.unique(y, return_counts=True)
            meta.update({"kept_categories": keep, "dropped_categories": dropped,
                         "n_splits": int(n_splits), "n_samples": int(len(y)),
                         "majority_rate": round(float(counts.max() / counts.sum()), 4),
                         "group_by_scenario": group_by_scenario, "standardize": standardize,
                         "categories_filter": list(categories) if categories else None})
        for kind in ("linear", "mlp"):
            real = _fit_eval_clf(X, y, groups, kind, seed, n_splits, shuffle_labels=False,
                                 standardize=standardize)
            ctrl = _fit_eval_clf(X, y, groups, kind, seed, n_splits, shuffle_labels=True,
                                 standardize=standardize)
            records.append({
                "layer": layer_tag, "probe": f"{probe_prefix}{kind}",
                "accuracy": round(real["accuracy"], 4), "macro_f1": round(real["macro_f1"], 4),
                "ctrl_shuffled_label_accuracy": round(ctrl["accuracy"], 4),
                "n_classes": len(real["classes"]), "n_samples": int(len(y)),
                "standardize": standardize,
            })
            if probe_prefix == "" and kind == "linear":
                confusions[layer_tag] = {"matrix": real["confusion"], "classes": real["classes"]}
                if real["coef"] is not None:
                    # Save the feature mean/std so the raw-space direction (coef/std) is recoverable
                    # even from a standardized run; for a raw run std=1 and coef is already raw-space.
                    std = X.std(0).astype(np.float32) if standardize else np.ones(X.shape[1], np.float32)
                    directions[layer_tag] = {"classes": real["classes"], "coef": real["coef"],
                                             "mean": X.mean(0).astype(np.float32), "std": std,
                                             "standardized": bool(standardize)}

    for layer in layer_list:
        X, y, ids = _load_pooled(latent_dir, layer)
        groups = np.array([scenario_for_id(i) for i in ids])
        _run(X, y, groups, layer)

    if pixel_baseline:
        Xp, yp, ids = _load_pixels(latent_dir)
        gp = np.array([scenario_for_id(i) for i in ids])
        _run(Xp, yp, gp, -1, probe_prefix="pixel_")

    if output_csv is not None and records:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return {"records": records, "confusions": confusions, "directions": directions, "meta": meta}
