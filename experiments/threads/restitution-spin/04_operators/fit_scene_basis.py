

#!/usr/bin/env python
"""A PCA scene basis, to replace the random projection the conditioned operators condition on.

**The gap this targets.** ``resolve_ridge_sweep.py`` measured, on held-out scenes:

    operator     achieved alignment     reachable-subspace ceiling
    canon             0.387                    0.441
    cond128           0.666                    0.826
    cond512           0.669                    0.868

Two readings follow. The command-only operator sits at 88% of its own span and is genuinely
span-limited -- no estimator recovers what the features cannot express. The conditioned operators are
not: their span already contains 0.87 of the true velocity direction, comfortably past the 0.80 bar,
and they extract 0.67 of it. That residual is ESTIMATION error. Two attacks on it have already failed
(a dense ridge sweep buys +0.004 at k=512; reduced-rank truncation buys nothing at any k), and raising
capacity does not help either -- k=128 to k=512 moves achieved alignment 0.666 to 0.669 while the
ceiling climbs 0.826 to 0.868. More capacity buys reachability the fit cannot harvest.

**Why the scene code is the suspect.** The conditioning vector ``z`` is a random ``k``-dim projection
of a 2,097,152-dim latent. Johnson-Lindenstrauss guarantees such a projection preserves relative
geometry, which is why it was a defensible default -- but "preserves geometry up to a distortion" is a
statement about pairwise distances, not about packing scene-discriminative variance into few
coordinates. Every one of those ``k`` coordinates is paid for in variance: the velocity operator's
feature dimension is ``27 + 3k``, so k=512 spends 1563 parameters estimated from 12,288 pairs. A basis
that captures the scene distribution in its FIRST few directions buys the same conditioning power for
far fewer parameters, which is exactly the trade the measurements say is binding.

**The basis.** Latents are stacked in the CANONICAL (ball-centred) frame -- the same frame the fit and
the code-building both use, since the roll changes the latent and a basis built in the wrong frame
would describe a different quantity -- centred, and reduced by SVD. Because the number of sampled
clips is a few thousand against a 2.1M-dim space, the economical route is the Gram matrix: the top
right-singular vectors come from ``X^T U diag(1/s)`` with ``U, s`` from the (n x n) eigendecomposition
of ``X X^T``. That is exact, not an approximation, and ``n`` is small by construction.

**The comparison this is built to support is controlled**: refit at the SAME k=128 the random-basis
operator used, changing only the basis. Any difference is then attributable to the basis and not to
capacity, ridge, layer, or frame -- all of which are held fixed.

    PYTHONPATH=. python experiments/threads/restitution-spin/04_operators/fit_scene_basis.py \
        --train_dir .../latents/spin_ball3d/train/vjepa2_large \
        --layers 18 --k 128 --out .../analysis/spin_ball3d/scene_basis_L18_k128.npz
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
from pathlib import Path

import sys
from pathlib import Path as _P

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="18")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--clips_per_scene", type=int, default=4,
                    help="clips sampled per scene for the basis. All 16 would be 4096 x 2.1M = 34 GB; "
                         "4 keeps the stacked matrix at ~8.6 GB while still spanning the within-scene "
                         "variation the code sees at test time, since the code is built from whichever "
                         "clip is the BASE, not from a scene average.")
    ap.add_argument("--num_scenes", type=int, default=0)
    ap.add_argument("--max_cached_shards", type=int, default=1)
    ap.add_argument("--canon", type=int, default=1)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    ds = LatentDataset(args.train_dir, layers=layers, max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)
    if args.num_scenes:
        sids = sids[: args.num_scenes]

    rows, kept = [], 0
    for n, s in enumerate(sids):
        idxs = sorted(scenes[s].values())[: args.clips_per_scene]
        ref = ds[idxs[0]]
        grid = tuple(int(x) for x in ref["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(ref), grid)
        for i in idxs:
            f = vo.layer_flat(ds[i]["layers"][layers[0]])
            rows.append((vo.roll_layer(f, grid, sh) if args.canon else f).astype(np.float32))
        kept += 1
        if (n + 1) % 32 == 0:
            print(f"[basis] scene {n + 1}/{len(sids)}, {len(rows)} clips", flush=True)

    X = np.stack(rows); del rows
    mu = X.mean(axis=0)
    X -= mu
    print(f"[basis] stacked {X.shape} ({X.nbytes / 1e9:.1f} GB), computing Gram", flush=True)

    # Exact top-k right singular vectors via the (n x n) Gram. n is a few thousand; D is 2.1M.
    G = (X @ X.T).astype(np.float64)
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1][: args.k]
    evals, U = evals[order], evecs[:, order]
    s = np.sqrt(np.clip(evals, 1e-12, None))
    P = (X.T @ (U / s)).astype(np.float32)          # (D, k), orthonormal columns

    total = float(np.trace(G))
    evr = float(evals.sum() / total) if total > 0 else float("nan")
    # Reported because it is the whole premise: if the top k directions of the scene distribution hold
    # little more variance than k random ones would, this basis cannot buy anything and the result
    # should be read as a negative before the refit is even run.
    print(f"[basis] top-{args.k} explains {100 * evr:.1f}% of centred variance "
          f"across {kept} scenes x {args.clips_per_scene} clips")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{f"P_{layers[0]}": P, f"mu_{layers[0]}": mu.astype(np.float32),
                          "explained_variance_ratio": np.array([evr]),
                          "singular_values": s.astype(np.float32),
                          "k": np.array([args.k]), "canon": np.array([int(args.canon)])})
    print(f"[basis] wrote {args.out}  P={P.shape}", flush=True)


if __name__ == "__main__":
    main()
