

#!/usr/bin/env python
"""Fit the DIRECTION-AWARE velocity operator (the open lever from canon_ridge).

canon_ridge (start-position roll + global ridge) gave NO lift over the raw ridge because, within a scene,
the start is already shared — the spread of Delta H is DIRECTION-induced (different headings traverse
different tokens), which a position-only canonicalization cannot align. V-JEPA is only approximately
TRANSLATION-equivariant (verified), not rotation-equivariant, so we cannot cleanly rotate the latent.

The equivariance-free fix implemented here: a DIRECTION-CONDITIONED canonicalized ridge. Bin the target
heading into ``n_bins`` wedges; within each wedge fit a position-canonicalized ridge B_theta on
(Delta v -> roll(Delta H, start->center)). At steer time, bin by the target heading, roll H_a's start to
center, apply B_theta, roll back. n_bins=1 reproduces the plain canon_ridge baseline; sweeping n_bins
tests whether direction-conditioning closes the gap toward the per-pair edit (1.7 deg).

Streams the train cache once; saves per (n_bins, bin, layer) ridge artifacts. Reuses fits from
experiments/threads/velocity/04_operators/velocity_subspace.py (which already saved the raw/canon ridge + PCA basis).

    python experiments/pipeline/04_operators/fit_dir_operator.py \
        --train_dir .../moving_ball_scene_v2d/train/vjepa2_large \
        --layers 6,12,18,23 --bins 4,8,16 --ridge 1.0 \
        --output_dir outputs/analysis/moving_ball_v2d/subspace
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--bins", default="4,8,16", help="comma list of n_bins to fit")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    bins_list = [int(x) for x in args.bins.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    scenes = vo.group_scenes(tr)
    print(f"[dir_op] train {len(scenes)} scenes; layers={layers}; bins={bins_list}", flush=True)

    # ops[N][b][L] = RidgeOperator over canonicalized (rolled) Delta H, for heading-bin b of n_bins=N
    ops: dict[int, dict[int, dict[int, vo.RidgeOperator]]] = {
        N: {b: {} for b in range(N)} for N in bins_list}
    counts: dict[int, dict[int, int]] = {N: {b: 0 for b in range(N)} for N in bins_list}
    dim_by_layer: dict[int, int] = {}

    n = 0
    for s in sorted(scenes):
        ranks = sorted(scenes[s]); a = ranks[0]
        sa = tr[scenes[s][a]]; va = vo.clip_velocity(sa)
        grid = tuple(int(x) for x in sa["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[scenes[s][b]]; vb = vo.clip_velocity(sb); dv = vb - va
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                dim_by_layer[L] = dH.size
                dG = vo.roll_layer(dH, grid, sh)          # canonicalize start -> center
                for N in bins_list:
                    bin_idx = vo.direction_bin(vb, N)      # condition on target heading
                    op = ops[N][bin_idx].setdefault(L, vo.RidgeOperator(dH.size, args.ridge))
                    op.add(dv, dG)
                    if L == layers[0]:
                        counts[N][bin_idx] += 1
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[dir_op]   {n}/{len(scenes)} scenes", flush=True)

    # solve + save
    for N in bins_list:
        for b in range(N):
            for L in layers:
                op = ops[N][b].get(L)
                if op is None:                            # empty bin -> zero operator
                    Bt = np.zeros((2, dim_by_layer[L]), dtype=np.float32)
                else:
                    Bt = op.solve().astype(np.float32)
                np.save(out / f"ridge_dirbin{N}_b{b}_Bt_L{L}.npy", Bt)
        print(f"[dir_op] n_bins={N}: bin counts={[counts[N][b] for b in range(N)]}", flush=True)

    meta = {"train_dir": args.train_dir, "layers": layers, "bins": bins_list,
            "ridge_lambda": args.ridge,
            "bin_counts": {str(N): [counts[N][b] for b in range(N)] for N in bins_list},
            "artifacts": "ridge_dirbin{N}_b{b}_Bt_L{L}.npy (canonicalized: roll start->center, apply, roll back)"}
    (out / "dir_operator_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dir_op] wrote artifacts + dir_operator_meta.json to {out}", flush=True)


if __name__ == "__main__":
    main()
