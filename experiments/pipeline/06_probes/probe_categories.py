

#!/usr/bin/env python
"""Step 1: probe whether physics *categories* form distinct subspaces in the VJEPA latent space.

Runs linear + MLP category classifiers per encoder layer (with shuffled-label and pixel/appearance
controls), then characterises the geometry: per-category direction cosines, principal angles between
category subspaces, and classifier-free separability. The linear probe's per-class weight vectors are
saved as ``category_directions.npz`` — these are the candidate steering directions reused in Step 3.

Example
-------
    python experiments/pipeline/06_probes/probe_categories.py \
        --latent_dir outputs/latents/physics_iq/vjepa2_large \
        --output_dir outputs/analysis/physics_iq/category_probe
"""

from __future__ import annotations

# --- repo-root shim: make ``src`` importable however this script is invoked ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis import subspace
from src.analysis import visualization as viz
from src.analysis.latent_geometry import pooled_features
from src.training.probe_classification import classify_categories


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="all", help='"all" or comma-separated layer indices')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_pixel_baseline", action="store_true")
    p.add_argument("--min_scenarios_per_class", type=int, default=4,
                   help="drop categories with fewer distinct scenarios (need >= for valid grouped CV)")
    p.add_argument("--no_group", action="store_true",
                   help="disable scenario-grouped CV (NOT recommended — leaks scenarios across folds)")
    p.add_argument("--no_standardize", action="store_true",
                   help="probe RAW latent values (skip StandardScaler z-scoring) to study which raw "
                        "subspace encodes the category without per-dim rescaling distortion")
    p.add_argument("--categories", default="solid_mechanics,fluid_dynamics,optics",
                   help='explicit category allow-list (comma-separated); "" keeps all. Default drops '
                        "thermodynamics + magnetism (too few scenarios for valid grouped CV)")
    p.add_argument("--subspace_layer", type=int, default=-1,
                   help="layer for subspace geometry; -1 = deepest available")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = "all" if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    categories = [c for c in args.categories.split(",") if c] or None
    standardize = not args.no_standardize

    result = classify_categories(
        args.latent_dir, layers=layers, seed=args.seed,
        output_csv=out / "category_decodability.csv",
        pixel_baseline=not args.no_pixel_baseline,
        min_scenarios_per_class=args.min_scenarios_per_class,
        group_by_scenario=not args.no_group,
        standardize=standardize,
        categories=categories,
    )
    records = result["records"]
    if records:
        viz.classification_by_layer_plot(records, out / "category_accuracy_by_layer.png")

    # Per-layer confusion matrices (linear probe, out-of-fold).
    for layer, cm in result["confusions"].items():
        if cm["matrix"] is not None:
            viz.confusion_matrix_plot(
                cm["matrix"], cm["classes"], out / f"confusion_layer{layer:02d}.png",
                title=f"Category confusion (layer {layer}, linear)")

    # Save the per-class linear directions (steering-direction bridge to Step 3). We also persist the
    # feature mean/std the probe saw, so the *raw-space* steering direction is recoverable as coef/std
    # (for a --no_standardize run std=1 and coef is already raw-space). ``standardized`` records which.
    if result["directions"]:
        np.savez(
            out / "category_directions.npz",
            standardized=np.asarray(standardize),
            **{f"layer{li}_classes": np.asarray(d["classes"]) for li, d in result["directions"].items()},
            **{f"layer{li}_coef": d["coef"] for li, d in result["directions"].items()},
            **{f"layer{li}_mean": d["mean"] for li, d in result["directions"].items()},
            **{f"layer{li}_std": d["std"] for li, d in result["directions"].items()},
        )

    # Subspace geometry at the chosen layer.
    avail = list({r["layer"] for r in records if r["layer"] >= 0})
    sub_layer = max(avail) if args.subspace_layer < 0 else args.subspace_layer
    feats, cats = pooled_features(args.latent_dir, sub_layer)
    if categories:  # mirror the classifier's allow-list so the geometry matches the probe
        cats_arr = np.asarray(cats)
        mask = np.isin(cats_arr, categories)
        feats, cats = feats[mask], cats_arr[mask].tolist()
    classes, means = subspace.category_directions(feats, cats, standardize=standardize)
    cos = subspace.cosine_matrix(means)
    pa_classes, angles = subspace.principal_angles(feats, cats, standardize=standardize)
    sep = subspace.separability(feats, cats, standardize=standardize)

    viz.similarity_heatmap(cos, classes, out / f"category_cosine_layer{sub_layer:02d}.png",
                           title=f"Category direction cosine (layer {sub_layer})")
    viz.similarity_heatmap(angles, pa_classes, out / f"category_angles_layer{sub_layer:02d}.png",
                           title=f"Principal angles (rad, layer {sub_layer})", vmin=0.0,
                           vmax=float(np.pi / 2), cmap="viridis")

    summary = {
        "latent_dir": str(args.latent_dir),
        "subspace_layer": int(sub_layer),
        "cv": result["meta"],
        "subspace_categories": classes,
        "separability": {k: round(float(v), 4) for k, v in sep.items()},
        "best_linear": max((r for r in records if r["probe"] == "linear" and r["layer"] >= 0),
                           key=lambda r: r["accuracy"], default=None),
        "best_mlp": max((r for r in records if r["probe"] == "mlp" and r["layer"] >= 0),
                        key=lambda r: r["accuracy"], default=None),
        "pixel_baseline": [r for r in records if str(r["probe"]).startswith("pixel_")],
    }
    (out / "category_probe_summary.json").write_text(json.dumps(summary, indent=2))
    m = result["meta"]
    print(f"[probe_categories] {len(records)} probe rows; kept={m.get('kept_categories')} "
          f"dropped={m.get('dropped_categories')} folds={m.get('n_splits')} "
          f"majority={m.get('majority_rate')}")
    if summary["best_linear"]:
        b = summary["best_linear"]
        print(f"[probe_categories] best linear acc={b['accuracy']} (macroF1={b['macro_f1']}) "
              f"@layer{b['layer']} vs shuffled ctrl={b['ctrl_shuffled_label_accuracy']} / "
              f"majority={m.get('majority_rate')}")


if __name__ == "__main__":
    main()
