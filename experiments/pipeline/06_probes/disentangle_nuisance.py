

#!/usr/bin/env python
"""Velocity vs nuisance disentanglement (size / colour / background).

The PI's "nothing wasted" plan: build datasets that each vary ONE factor (velocity, size, colour,
background) with everything else held fixed, take the same-scene difference H_b - H_a to isolate that
factor, PCA each into a subspace, and ask whether the VELOCITY subspace is geometrically DISTINCT from
the appearance-nuisance subspaces. If velocity is near-orthogonal to size/colour/background, then the
velocity edit is disentangled from appearance (steering velocity shouldn't change how the ball looks).

Calibration: in a ~2M-dim latent, two RANDOM k-dim subspaces are ~90 deg apart, so ~90 deg = "share
nothing" and angles well below 90 deg = "share directions". We report the random baseline alongside.

Velocity subspace is loaded from the v2d global PCA basis (global_basis_L*.npy from velocity_subspace.py);
size/colour/background subspaces are computed here from their caches (rank0 vs rank K-1 anchor pairs).

    python experiments/pipeline/06_probes/disentangle_nuisance.py \
        --velocity_basis_dir outputs/analysis/moving_ball_v2d/subspace \
        --size_dir .../moving_ball_scene_size/train/vjepa2_large \
        --color_dir .../moving_ball_scene_color/train/vjepa2_large \
        --bg_dir .../moving_ball_scene_background/train/vjepa2_large \
        --layers 6,12,18,23 --k 8 --output_dir outputs/analysis/moving_ball_v2d/disentangle
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
import gc
import json
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def factor_basis(latent_dir, layers, k, max_pairs):
    """Top-k PCA basis of the same-scene difference H_b - H_a (rank0 vs rank K-1) for one factor cache.

    Returns {layer: (k, D) orthonormal basis}. Streams pairs into a per-layer buffer (one layer kept at a
    time would need re-reads; here we buffer all layers but cap the pair count to bound RAM).
    """
    ds = LatentDataset(latent_dir, layers=layers)
    scenes = vo.group_scenes(ds)
    buf = {L: [] for L in layers}
    for s in sorted(scenes):
        ranks = sorted(scenes[s])
        if len(ranks) < 2:
            continue
        sa = ds[scenes[s][ranks[0]]]; sb = ds[scenes[s][ranks[-1]]]
        for L in layers:
            buf[L].append((vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L])).astype(np.float32))
        if max_pairs and len(buf[layers[0]]) >= max_pairs:
            break
    out = {}
    for L in layers:
        X = np.stack(buf[L], 0)
        basis, _ = vo.pca_gram(X, k=k)
        out[L] = basis
        del X, buf[L]; gc.collect()
    return out, len(scenes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--velocity_basis_dir", required=True, help="dir with global_basis_L*.npy (v2d)")
    p.add_argument("--size_dir", required=True)
    p.add_argument("--color_dir", required=True)
    p.add_argument("--bg_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=500)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    k = args.k
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.velocity_basis_dir)

    # velocity subspace from the saved v2d global PCA basis (top-k)
    bases = {"velocity": {L: np.load(vdir / f"global_basis_L{L}.npy")[:k].astype(np.float64) for L in layers}}
    for name, d in [("size", args.size_dir), ("color", args.color_dir), ("background", args.bg_dir)]:
        print(f"[disent] computing {name} subspace from {d}", flush=True)
        b, nsc = factor_basis(d, layers, k, args.max_pairs)
        bases[name] = {L: b[L].astype(np.float64) for L in layers}
        np.save(out / f"basis_{name}_k{k}.npy", np.stack([b[L] for L in layers], 0).astype(np.float32))
        print(f"[disent]   {name}: {nsc} scenes", flush=True)

    factors = ["velocity", "size", "color", "background"]
    rng = np.random.default_rng(0)
    summary = {"layers": layers, "k": k, "per_layer": {}}
    for L in layers:
        D = bases["velocity"][L].shape[1]
        randb = vo.random_basis(D, k, rng)
        ang = {}
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                fi, fj = factors[i], factors[j]
                ang[f"{fi}__{fj}"] = vo.principal_angles_bases(bases[fi][L], bases[fj][L])
        # random baseline vs velocity (calibrates "~90 deg = orthogonal")
        ang["velocity__random"] = vo.principal_angles_bases(bases["velocity"][L], randb)
        summary["per_layer"][str(L)] = ang
        v = ang
        print(f"[disent] L{L} mean principal angle (deg): "
              f"vel-size={v['velocity__size']['mean_deg']:.1f} "
              f"vel-color={v['velocity__color']['mean_deg']:.1f} "
              f"vel-bg={v['velocity__background']['mean_deg']:.1f} | "
              f"vel-random={v['velocity__random']['mean_deg']:.1f} (ref) | "
              f"size-color={v['size__color']['mean_deg']:.1f} "
              f"size-bg={v['size__background']['mean_deg']:.1f} "
              f"color-bg={v['color__background']['mean_deg']:.1f}", flush=True)

    (out / "disentangle_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[disent] wrote {out}/disentangle_summary.json", flush=True)


if __name__ == "__main__":
    main()
