

#!/usr/bin/env python
"""Step 2 (velocity-first): equivariance probe — does rotating velocity rotate the latent subspace?

The supervisor's equivariance test: if we have the ball moving at the SAME SPEED but in different
DIRECTIONS (the ``rotated`` dataset), a true "velocity subspace" should be equivariant to direction
rotation. Concretely, if ``phi(v)`` is the projection of the latent onto the velocity subspace, then
    phi(R_theta * v) ≈ R_theta * phi(v)
i.e. rotating the velocity direction by angle ``theta`` should rotate the subspace representation by
the same angle (up to some fixed subspace rotation).

We test this in two complementary ways:

1. **Subspace dimensionality and structure**: Fit a 2D PCA subspace on all ``(latent, velocity_direction)``
   pairs. If the subspace is equivariant, the projection of each clip onto that 2D plane should trace a
   *circle* as the velocity direction sweeps 0..2pi. We measure the circularity (ratio of PCA eigenvalues:
   a true circle has ratio≈1, a flat line has ratio≈0). We also directly regress the angle from the
   2D projection coordinates (R² near 1 means the direction is recoverable from the 2D subspace).

2. **Equivariance error**: For each pair of clips at velocities ``v1`` and ``v2`` (same speed, different
   directions, angle difference ``theta``), compute the actual rotation ``R_obs`` of their subspace
   projections and compare to the expected ``R_theta``. The equivariance error is
   ``||R_obs - R_theta||_F / 2`` (0=perfect equivariance, 1=completely wrong). Plot vs. angle difference.

3. **Camera-rotation equivariance** (if run with ``moving_ball_equivariance.yaml camera_rotation: true``):
   same speed + direction, but the camera is rolled by varying angles. The velocity subspace should be
   stable under camera roll (invariant, a special case of equivariance with R=I).

Outputs: ``equivariance_report.json``, ``subspace_circle.png``, ``equivariance_error.png``.

Example
-------
    python experiments/pipeline/06_probes/probe_equivariance.py \
        --latent_dir outputs/latents/moving_ball_equivariance/vjepa2_large \
        --output_dir outputs/analysis/moving_ball_equivariance/equivariance \
        --layer 18
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.encoders.feature_extractor import LatentDataset


def _pool(tokens: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Clip-mean-pool: ``(L, D)`` -> ``(D,)``."""
    tp, hp, wp = grid
    n = tp * hp * wp
    return tokens[:n].mean(0)


def _load_layer(dataset: LatentDataset, layer: int):
    """Load clip-pooled features, ground-truth velocity angle and speed for each clip."""
    feats, angles, speeds, vel_vecs = [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        keys = s["state_keys"]
        st = s["state"].numpy()
        feats.append(_pool(s["layers"][layer].numpy(), tuple(s["grid"])))
        vx = float(st[:, keys.index("obj0_vel_x")].mean())
        vy = float(st[:, keys.index("obj0_vel_y")].mean())
        angles.append(float(np.arctan2(vy, vx)))
        speeds.append(float(np.sqrt(vx**2 + vy**2)))
        vel_vecs.append([vx, vy])
    return (np.stack(feats, 0), np.array(angles), np.array(speeds), np.array(vel_vecs))


def _fit_velocity_subspace(features: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """PCA subspace of the clip-mean-centered features. Returns (projections (N,k), components (k,D))."""
    mu = features.mean(0, keepdims=True)
    centered = features - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]                         # (k, D)
    projections = centered @ components.T        # (N, k)
    return projections, components


def _circularity(proj2d: np.ndarray) -> float:
    """Ratio of smaller to larger PCA eigenvalue of a 2D point cloud (1=circle, 0=line)."""
    _, s, _ = np.linalg.svd(proj2d - proj2d.mean(0), full_matrices=False)
    return float(s[1] / (s[0] + 1e-12))


def _angle_from_projection(proj2d: np.ndarray, gt_angles: np.ndarray) -> float:
    """R² of recovering the velocity angle from the 2D subspace projection."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X = proj2d
    # target: (cos theta, sin theta) — continuous, no 2pi wrap
    Y = np.stack([np.cos(gt_angles), np.sin(gt_angles)], axis=1)
    n = len(X)
    split = max(1, int(0.8 * n))
    model = Ridge(alpha=1.0).fit(X[:split], Y[:split])
    pred = model.predict(X[split:])
    return float(r2_score(Y[split:], pred))


def _equivariance_errors(proj2d: np.ndarray, gt_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For randomly sampled pairs, compare predicted vs observed subspace rotation.

    Returns (angle_diffs, equivariance_errors) both of shape (M,).
    """
    n = len(proj2d)
    rng = np.random.default_rng(0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 500:  # subsample for speed
        idx = rng.choice(len(pairs), 500, replace=False)
        pairs = [pairs[k] for k in idx]

    dthetas, errors = [], []
    for i, j in pairs:
        dtheta = float(gt_angles[j] - gt_angles[i])
        # expected rotation matrix R(dtheta)
        ca, sa = np.cos(dtheta), np.sin(dtheta)
        R_expected = np.array([[ca, -sa], [sa, ca]])
        # observed: solve p_j ≈ R_obs @ p_i
        p_i = proj2d[i]
        p_j = proj2d[j]
        if np.linalg.norm(p_i) < 1e-8 or np.linalg.norm(p_j) < 1e-8:
            continue
        # best-fit rotation (Kabsch on a single point pair)
        M = np.outer(p_j, p_i)  # (2,2)
        U, _, Vt = np.linalg.svd(M)
        R_obs = U @ Vt
        err = float(np.linalg.norm(R_obs - R_expected) / 2.0)
        dthetas.append(abs(dtheta % np.pi))  # fold into [0, pi]
        errors.append(err)
    return np.array(dthetas), np.array(errors)


def _plot_circle(proj2d: np.ndarray, angles: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sc = ax.scatter(proj2d[:, 0], proj2d[:, 1], c=angles % (2 * np.pi),
                    cmap="hsv", s=20, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="velocity direction (rad)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Velocity subspace 2D projection\n(circular = equivariant)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_errors(dthetas: np.ndarray, errors: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(np.degrees(dthetas), errors, s=8, alpha=0.4, color="steelblue")
    # running median
    bins = np.linspace(0, 180, 13)
    binc = (bins[:-1] + bins[1:]) / 2
    meds = [np.median(errors[(np.degrees(dthetas) >= bins[k]) & (np.degrees(dthetas) < bins[k+1])])
            if ((np.degrees(dthetas) >= bins[k]) & (np.degrees(dthetas) < bins[k+1])).any()
            else np.nan for k in range(len(binc))]
    ax.plot(binc, meds, "o-", color="tomato", label="median error")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axhline(1.0, color="grey", ls="--", lw=0.7, label="max possible error")
    ax.set_xlabel("angle difference |Δθ| (deg)"); ax.set_ylabel("equivariance error")
    ax.set_title("Equivariance error vs velocity direction change\n(0=perfect, 1=none)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True,
                   help="latent cache for the moving_ball_equivariance (rotated) dataset")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layer", type=int, default=-1,
                   help="which layer to analyse (-1 = deepest available)")
    p.add_argument("--layers", default=None,
                   help="run multiple layers (overrides --layer): 'all' or comma-sep")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = LatentDataset(args.latent_dir, layers="all")
    available = dataset.available_layers()
    if args.layers is not None:
        layer_list = available if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    else:
        layer_list = [max(available) if args.layer < 0 else args.layer]

    report = {}
    for layer in layer_list:
        feats, angles, speeds, _ = _load_layer(dataset, layer)
        # Evict the per-shard latent cache between layers to avoid accumulating every shard's full
        # multi-layer token tensors in RAM (OOMs the job when run with --layers all).
        dataset._shard_cache.clear()
        proj2d, _ = _fit_velocity_subspace(feats, k=2)
        circ = _circularity(proj2d)
        r2_ang = _angle_from_projection(proj2d, angles)
        dthetas, errors = _equivariance_errors(proj2d, angles)
        mean_err = float(np.mean(errors)) if len(errors) else float("nan")
        report[str(layer)] = {
            "circularity": round(circ, 4),
            "angle_r2_from_subspace": round(r2_ang, 4),
            "equivariance_error_mean": round(mean_err, 4),
            "n_clips": int(len(feats)),
            "n_pair_samples": int(len(errors)),
        }
        print(f"  layer {layer:2d}: circularity={circ:.3f}  angle_R²={r2_ang:.3f}  "
              f"equivariance_err={mean_err:.3f}  (n={len(feats)})")
        _plot_circle(proj2d, angles, out / f"subspace_circle_L{layer}.png")
        if len(dthetas):
            _plot_errors(dthetas, errors, out / f"equivariance_error_L{layer}.png")

    (out / "equivariance_report.json").write_text(json.dumps(report, indent=2))
    print(f"[probe_equivariance] report -> {out / 'equivariance_report.json'}")


if __name__ == "__main__":
    main()
