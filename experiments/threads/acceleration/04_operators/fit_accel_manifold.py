

#!/usr/bin/env python
"""Fit the concept-space ACTIVATION MANIFOLD for acceleration (port of arXiv:2605.05115).

The paper's recipe, applied to 2-D acceleration:

  1. reduce activations to ``k_pca`` dimensions (paper uses 64),
  2. fit a smooth surface through the per-concept centroids -- thin-plate splines, because the
     concept domain here is 2-D exactly as in the paper's in-context-learning tasks,
  3. steer by moving in the manifold's INTRINSIC coordinates: ``edit = s(a_b) - s(a_a)``.

The representation is the spatially pooled temporal profile ``r(H) in R^{T x D}`` -- the same object
the existing spline operator edits, so ``man`` and ``spline_K*`` live in the *same function class*
(spatially uniform edits) and the comparison is about the concept-space geometry alone.

Targets are SCENE-CENTRED at fit time (``R - mean over the scene's 8 clips``): the ``_mixed`` dataset
randomizes colour and background per scene, and centring is what makes the fitted surface a map of
acceleration rather than a map of appearance. Nothing at steer time needs the scene, because the edit
is a difference and any constant cancels -- it reads the two acceleration labels and never ``H_b``.

Three estimators are fitted, differing ONLY in the concept-space geometry:

  man     thin-plate spline surface through the centroids       (the paper's method)
  manlin  affine block only -- the zero-curvature limit          (the control that isolates curvature)
  mancr   cubic surface via a denser knot set (n_centers x 2)    (curvature-capacity ablation)

Hyperparameters (ridge, n_centers, k_pca) are selected on a held-out slice of TRAIN, never on test.

    python experiments/threads/acceleration/04_operators/fit_accel_manifold.py \
        --train_dir .../moving_ball_scene_accel2d_mixed/train/vjepa2_large \
        --test_dir  .../moving_ball_scene_accel2d_mixed/test/vjepa2_large \
        --layers 6,12,18,23 \
        --output_dir outputs/analysis/moving_ball_accel2d_mixed/manifold
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

from src.analysis import manifold_ops as mo
from src.analysis import spline_ops as sp
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def collect(ds, scenes, scene_ids, layers, grid, qfn, bigU=None):
    """Stream a split into ``Z (N,2)`` accelerations and ``Y{L}`` SCENE-CENTRED representations.

    ``bigU=None`` gives the pooled temporal profile ``(N, T*D)`` -- the spatially uniform
    representation every command-only arm in this thread lives in. ``bigU={L: (k, T*H*W*D)}`` instead
    gives coordinates in the saved global delta subspace, which RETAINS SPATIAL PLACEMENT: the whole
    reason the profile-only oracle decodes at 64deg while the true delta decodes at 11.24 is that
    placement carries most of the signal, and no command-only arm has ever been able to express it.
    """
    Z, Y = [], {L: [] for L in layers}
    for i, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        prof = {L: [] for L in layers}
        accs = []
        for r in ranks:
            smp = ds[scenes[s][r]]
            accs.append(qfn(smp))
            for L in layers:
                flat = vo.layer_flat(smp["layers"][L])
                prof[L].append(flat.astype(np.float32) if bigU is not None
                               else sp.temporal_profile(flat, grid).reshape(-1))
            del smp
        Z.append(np.stack(accs))
        for L in layers:
            if bigU is not None:
                P = mo.project_block(bigU[L], np.stack(prof[L], axis=1))   # (n_ranks, k)
            else:
                P = np.stack(prof[L])
            Y[L].append((P - P.mean(axis=0, keepdims=True)).astype(np.float32))
        del prof
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"    collected {i+1}/{len(scene_ids)} scenes", flush=True)
    return np.concatenate(Z), {L: np.concatenate(Y[L]) for L in layers}


def pair_index(scenes, scene_ids, n_ranks_hint=8):
    """``[(scene_slot, rank_a, rank_b), ...]`` -- anchor rank 0 to every other rank, per scene."""
    out = []
    for slot, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        for b in range(1, len(ranks)):
            out.append((slot, 0, b))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--k_pca", default="64", help="comma list of PCA dims to select over (paper: 64)")
    p.add_argument("--n_centers", default="32,64,128", help="comma list of thin-plate knot counts")
    p.add_argument("--ridge", default="1e-2,1e-3,1e-4,1e-5")
    p.add_argument("--val_scene_frac", type=float, default=0.2,
                   help="slice of TRAIN held out for hyperparameter selection (never test)")
    p.add_argument("--max_scenes", type=int, default=0, help="0 = all; smoke-test knob")
    p.add_argument("--quantity", choices=["accel", "angvel"], default="accel")
    p.add_argument("--bigU_dir", default="",
                   help="fit the manifold in the FULL TOKEN space instead of the pooled profile, using "
                        "global_basis_L*.npy as the reduction (the paper reduces activations by PCA "
                        "before fitting the surface; this is the same move at the token level). "
                        "Artifacts are written with a 'U' suffix -- manU_L*.npz etc -- and the steer "
                        "script lifts them back with the same basis")
    p.add_argument("--bigU_k", type=int, default=128, help="retained dims of global_basis for --bigU_dir")
    args = p.parse_args()

    qfn = vo.clip_angvel if args.quantity == "angvel" else vo.clip_acceleration
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # cap the shard LRU: this pass streams whole splits, and the unbounded default accumulates every
    # decoded shard in RAM (~136 GB on a 4000-clip 4-layer split -- it OOM-killed the first smoke run)
    tr = LatentDataset(args.train_dir, layers=layers, max_cached_shards=2)
    te = LatentDataset(args.test_dir, layers=layers, max_cached_shards=2)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    tr_ids, te_ids = sorted(tr_scenes), sorted(te_scenes)
    if args.max_scenes:
        tr_ids, te_ids = tr_ids[: args.max_scenes], te_ids[: args.max_scenes]
    grid = tuple(int(x) for x in tr[tr_scenes[tr_ids[0]][0]]["grid"])
    T, D = grid[0], int(tr.records[0]["hidden_dim"])
    print(f"[man-fit] train {len(tr_ids)} scenes / test {len(te_ids)}; layers={layers}; "
          f"grid={grid}; profile dim={T*D}", flush=True)

    bigU, sfx = None, ""
    if args.bigU_dir:
        sfx = "U"
        bigU = {}
        for L in layers:
            bigU[L] = np.load(Path(args.bigU_dir) / f"global_basis_L{L}.npy")[: args.bigU_k]
        print(f"[man-fit] token-space mode: basis {bigU[layers[0]].shape} per layer", flush=True)

    n_val = int(round(args.val_scene_frac * len(tr_ids)))
    fit_ids, val_ids = tr_ids[n_val:], tr_ids[:n_val]
    print(f"[man-fit] collecting TRAIN-fit ({len(fit_ids)} scenes)", flush=True)
    Zf, Yf = collect(tr, tr_scenes, fit_ids, layers, grid, qfn, bigU)
    print(f"[man-fit] collecting TRAIN-val ({len(val_ids)} scenes)", flush=True)
    Zv, Yv = collect(tr, tr_scenes, val_ids, layers, grid, qfn, bigU)
    print(f"[man-fit] Z fit {Zf.shape}, |a| range "
          f"[{np.linalg.norm(Zf,axis=1).min():.5f}, {np.linalg.norm(Zf,axis=1).max():.5f}]", flush=True)

    K_PCA = [int(x) for x in args.k_pca.split(",")]
    NC = [int(x) for x in args.n_centers.split(",")]
    RD = [float(x) for x in args.ridge.split(",")]

    val_pairs = pair_index(tr_scenes, val_ids)
    n_ranks = len(sorted(tr_scenes[val_ids[0]]))

    def pair_cos(mf, Zsplit, Ysplit, pairs, ranks_per_scene):
        """Mean cos(predicted edit, true within-scene profile difference) on held-out pairs."""
        cs = []
        for slot, ra, rb in pairs:
            ia, ib = slot * ranks_per_scene + ra, slot * ranks_per_scene + rb
            true = Ysplit[ib].astype(np.float64) - Ysplit[ia].astype(np.float64)
            pred = mf.edit(Zsplit[ia], Zsplit[ib])
            cs.append(vo.cosine(pred, true))
        return float(np.mean(cs))

    # --------------------------------------------------------------- hyperparameter selection (TRAIN)
    chosen, sel_table = {}, {}
    for L in layers:
        best = None
        rows = []
        for k in K_PCA:
            for nc in NC:
                for rd in RD:
                    mf = mo.fit_manifold(Zf, Yf[L], n_centers=nc,
                                         k_pca=min(k, Yf[L].shape[1]), ridge=rd)
                    c = pair_cos(mf, Zv, Yv[L], val_pairs, n_ranks)
                    rows.append({"k_pca": k, "n_centers": nc, "ridge": rd, "val_pair_cos": round(c, 4)})
                    if best is None or c > best[0]:
                        best = (c, k, nc, rd)
        chosen[L] = {"k_pca": best[1], "n_centers": best[2], "ridge": best[3],
                     "val_pair_cos": round(best[0], 4)}
        sel_table[L] = sorted(rows, key=lambda r: -r["val_pair_cos"])[:6]
        print(f"[man-fit] L{L} chose {chosen[L]}", flush=True)

    # --------------------------------------------------------------- final fits on ALL of train
    Zall = np.concatenate([Zf, Zv])
    manifolds = {}
    for L in layers:
        Yall = np.concatenate([Yf[L], Yv[L]])
        c = chosen[L]
        kp = min(c["k_pca"], Yall.shape[1])
        mf = mo.fit_manifold(Zall, Yall, n_centers=c["n_centers"], k_pca=kp, ridge=c["ridge"])
        mo.save_manifold(mf, out / f"man{sfx}_L{L}")
        # zero-curvature control: identical estimator with the RBF block removed
        mfl = mo.fit_manifold(Zall, Yall, n_centers=c["n_centers"], k_pca=kp,
                              ridge=c["ridge"], linear_only=True)
        mo.save_manifold(mfl, out / f"man{sfx}lin_L{L}")
        # curvature-capacity ablation: twice the knots, same everything else
        mfd = mo.fit_manifold(Zall, Yall, n_centers=2 * c["n_centers"], k_pca=kp,
                              ridge=c["ridge"])
        mo.save_manifold(mfd, out / f"man{sfx}dense_L{L}")
        manifolds[L] = {"man": mf, "manlin": mfl, "mandense": mfd}
        del Yall
        gc.collect()
    del Yf, Yv
    # drop the train shard LRU before touching test: otherwise both splits' decoded shards are live at
    # once, which is what OOM-killed the smoke run even with the cap in place
    tr._shard_cache.clear()
    gc.collect()

    # --------------------------------------------------------------- held-out TEST gate (diagnostic)
    # The repo's own history is unambiguous that this gate is DIAGNOSTIC ONLY: four separate arms have
    # moved it without the decode following. It is recorded to catch a broken fit, not to claim a win.
    print("[man-fit] collecting TEST for the held-out gate", flush=True)
    Zt, Yt = collect(te, te_scenes, te_ids, layers, grid, qfn, bigU)
    te_pairs = pair_index(te_scenes, te_ids)
    n_ranks_te = len(sorted(te_scenes[te_ids[0]]))
    gate = {}
    for L in layers:
        gate[str(L)] = {name: round(pair_cos(m, Zt, Yt[L], te_pairs, n_ranks_te), 4)
                        for name, m in manifolds[L].items()}
        # does s^{-1} recover the acceleration it should? (the paper's inverse map, sanity-checked)
        errs = []
        for slot, ra, _ in te_pairs[:: n_ranks_te - 1][:60]:
            i = slot * n_ranks_te + ra
            a_hat = manifolds[L]["man"].invert(Yt[L][i].astype(np.float64))  # noqa: E501
            errs.append(float(np.linalg.norm(a_hat - Zt[i]) / (np.linalg.norm(Zt[i]) + 1e-12)))
        gate[str(L)]["inverse_rel_err"] = round(float(np.median(errs)), 4)
        print(f"[man-fit] L{L} gate {gate[str(L)]}", flush=True)

    summary = {
        "method": "concept-space manifold steering (arXiv:2605.05115) on the pooled temporal profile",
        "train_dir": args.train_dir, "test_dir": args.test_dir, "quantity": args.quantity,
        "layers": layers, "grid": list(grid), "T": T, "D": D,
        "n_train_scenes": len(tr_ids), "n_fit_scenes": len(fit_ids), "n_trainval_scenes": len(val_ids),
        "n_test_scenes": len(te_ids),
        "chosen": {str(L): chosen[L] for L in layers},
        "selection_top6": {str(L): sel_table[L] for L in layers},
        "heldout_gate_pair_cos": gate,
        "representation": "bigU token coords" if bigU is not None else "pooled temporal profile",
        "bigU_dir": args.bigU_dir, "bigU_k": args.bigU_k if bigU is not None else 0,
        "artifacts": f"man{sfx}_L{{L}}.npz / man{sfx}lin_L{{L}}.npz / man{sfx}dense_L{{L}}.npz",
        "note": ("the gate is a LATENT cosine and is diagnostic only -- this repo has four instances of "
                 "it moving without the decode following. experiments/threads/acceleration/05_steering/steer_accel_spline.py decides."),
    }
    (out / f"manifold{sfx}_meta.json").write_text(json.dumps(summary, indent=2))
    print(f"[man-fit] -> {out}/manifold{sfx}_meta.json")


if __name__ == "__main__":
    main()
