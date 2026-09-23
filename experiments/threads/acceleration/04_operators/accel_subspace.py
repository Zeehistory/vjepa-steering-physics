

#!/usr/bin/env python
"""Acceleration subspace + operator fitting (Step 2 "Bravo"): PCA of Delta H, principal angles, ridge.

The acceleration analog of ``experiments/threads/velocity/04_operators/velocity_subspace.py`` — identical math, but the per-clip command is
the constant 2D ACCELERATION (``velocity_ops.clip_acceleration``) instead of the velocity, and the
difference vector is ``Delta a = a_b - a_a``. Latent-only (no decoder). Fits on the TRAIN scene cache,
previews held-out TEST generalization, and saves the SAME artifact filenames the velocity pipeline uses
(``global_basis_L*.npy`` + raw/canonicalized ridge ``ridge_Bt_L*`` / ``ridge_canon_Bt_L*``) so
``fit_command_operators_accel.py`` and ``steer_accel2d.py`` consume them unchanged. Output goes to a
SEPARATE accel artifacts dir, so nothing collides with the velocity artifacts.

Answers, in latent space:
  * within-scene PCA  -> is the local acceleration subspace low-rank (~2D = a's 2 DOF)?
  * global PCA        -> how much higher is the cross-scene rank (the token-misalignment cost)?
  * principal angles  -> do per-scene acceleration subspaces point the same way or scatter?
  * ridge F_U: da->dH -> does a single global linear operator predict the held-out difference vector?

    python experiments/threads/acceleration/04_operators/accel_subspace.py \
        --train_dir .../moving_ball_scene_accel2d/train/vjepa2_large \
        --test_dir  .../moving_ball_scene_accel2d/test/vjepa2_large \
        --layers 6,12,18,23 --output_dir outputs/analysis/moving_ball_accel2d/subspace --save_k 16
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


QUANTITY_FNS = {          # one harness for every physics quantity in the sweep
    "accel": vo.clip_acceleration,   # also used for `gravity` (a 1-DOF acceleration scenario)
    "velocity": vo.clip_velocity,
    "angvel": vo.clip_angvel,
    "angaccel": vo.clip_angaccel,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--quantity", choices=sorted(QUANTITY_FNS), default="accel",
                   help="accel -> clip_acceleration (2D a); angvel -> clip_angvel ([omega,0], rotational).")
    p.add_argument("--save_k", type=int, default=8, help="top-k global PCA components to save as basis")
    p.add_argument("--max_global_pairs", type=int, default=800)
    p.add_argument("--max_angle_scenes", type=int, default=60)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    qfn = QUANTITY_FNS[args.quantity]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    print(f"[a-subspace] train {len(tr_scenes)} scenes, test {len(te_scenes)} scenes; layers={layers}")

    summary = {"train_dir": args.train_dir, "test_dir": args.test_dir, "layers": layers, "quantity": args.quantity,
               "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes), "per_layer": {}}

    dim_by_layer = {}
    ridge = {}
    ridge_canon = {}
    within_curves = {L: [] for L in layers}
    within_pr = {L: [] for L in layers}
    global_buf = {L: [] for L in layers}
    global_dv = []
    scene_bases = {L: [] for L in layers}
    rng = np.random.default_rng(0)

    n_scene = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s])
        a = ranks[0]
        sa = tr[tr_scenes[s][a]]
        va = qfn(sa)
        pos = vo.clip_start_pos(sa)
        grid = tuple(int(x) for x in sa["grid"])
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        sh = vo.canon_shift(pos, grid)
        per_layer_deltas = {L: [] for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            vb = qfn(sb)
            dv = vb - va
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                dim_by_layer[L] = dH.size
                ridge.setdefault(L, vo.RidgeOperator(dH.size, args.ridge)).add(dv, dH)
                ridge_canon.setdefault(L, vo.RidgeOperator(dH.size, args.ridge)).add(
                    dv, vo.roll_layer(dH, grid, sh))
                per_layer_deltas[L].append(dH.astype(np.float32))
                if len(global_buf[L]) < args.max_global_pairs:
                    global_buf[L].append(dH.astype(np.float32))
            if len(global_dv) < args.max_global_pairs:
                global_dv.append(dv)
        for L in layers:
            M = np.stack(per_layer_deltas[L], 0).astype(np.float64)
            basis, ev = vo.pca_gram(M)
            within_curves[L].append(vo.explained_curve(ev, 6))
            within_pr[L].append(vo.participation_ratio(ev))
            if n_scene < args.max_angle_scenes and basis.shape[0] >= 2:
                scene_bases[L].append(basis[:2])
        n_scene += 1
        if n_scene % 100 == 0:
            print(f"[a-subspace]   processed {n_scene}/{len(tr_scenes)} train scenes")

    for L in layers:
        D = dim_by_layer[L]
        Bt = ridge[L].solve()
        Bt_canon = ridge_canon[L].solve()
        np.save(out / f"ridge_Bt_L{L}.npy", Bt.astype(np.float32))
        np.save(out / f"ridge_canon_Bt_L{L}.npy", Bt_canon.astype(np.float32))

        Xg = np.stack(global_buf[L], 0).astype(np.float32)
        gbasis, gev = vo.pca_gram(Xg, k=args.save_k)
        np.save(out / f"global_basis_L{L}.npy", gbasis.astype(np.float32))
        del Xg, global_buf[L]; gc.collect()

        wc = np.asarray(within_curves[L]).mean(0).tolist()
        wpr = float(np.mean(within_pr[L]))

        angs = []
        sb = scene_bases[L]
        for i in range(len(sb)):
            for j in range(i + 1, len(sb)):
                sv = np.linalg.svd(sb[i] @ sb[j].T, compute_uv=False)
                angs.append(float(np.degrees(np.arccos(np.clip(sv, -1, 1)).mean())))
        mean_angle = float(np.mean(angs)) if angs else float("nan")

        summary["per_layer"][str(L)] = {
            "dim": int(D),
            "within_scene_explained_top1to6": [round(x, 4) for x in wc],
            "within_scene_participation_ratio": round(wpr, 3),
            "global_explained_top1to8": [round(x, 4) for x in vo.explained_curve(gev, 8)],
            "global_participation_ratio": round(vo.participation_ratio(gev), 3),
            "mean_principal_angle_deg_between_scene_subspaces": round(mean_angle, 2),
        }
        print(f"[a-subspace] L{L}: within PR={wpr:.2f} (top2={wc[1]:.3f}), "
              f"global PR={summary['per_layer'][str(L)]['global_participation_ratio']:.1f} "
              f"(top2={summary['per_layer'][str(L)]['global_explained_top1to8'][1]:.3f}), "
              f"scene-subspace angle={mean_angle:.1f}deg")

    Bt_mem = {L: ridge[L].Bt for L in layers}
    top2 = {L: np.load(out / f"global_basis_L{L}.npy")[:2].astype(np.float64) for L in layers}
    randb = {L: vo.random_basis(dim_by_layer[L], 2, rng) for L in layers}
    acc = {L: {"rcos": [], "rrel": [], "ucos": [], "xcos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = qfn(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; dv = qfn(sb) - va
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                pred = dv @ Bt_mem[L]
                acc[L]["rcos"].append(vo.cosine(pred, dH))
                acc[L]["rrel"].append(vo.rel_error(pred, dH))
                acc[L]["ucos"].append(vo.cosine(vo.project(dH, top2[L]), dH))
                acc[L]["xcos"].append(vo.cosine(vo.project(dH, randb[L]), dH))
        del Ha; gc.collect()
    gen = {}
    for L in layers:
        gen[str(L)] = {
            "ridge_predict_dH_cosine": round(float(np.mean(acc[L]["rcos"])), 4),
            "ridge_predict_dH_rel_error": round(float(np.mean(acc[L]["rrel"])), 4),
            "U2_projection_retains_cosine": round(float(np.mean(acc[L]["ucos"])), 4),
            "random2_projection_retains_cosine": round(float(np.mean(acc[L]["xcos"])), 4),
        }
        print(f"[a-subspace] L{L} TEST: ridge cos(dH)={gen[str(L)]['ridge_predict_dH_cosine']:.3f} "
              f"relerr={gen[str(L)]['ridge_predict_dH_rel_error']:.3f} | "
              f"U2 retain={gen[str(L)]['U2_projection_retains_cosine']:.3f} vs "
              f"rand2={gen[str(L)]['random2_projection_retains_cosine']:.3f}", flush=True)
    summary["test_latent_generalization"] = gen
    summary["artifacts"] = {"ridge": "ridge_Bt_L*.npy", "ridge_canon": "ridge_canon_Bt_L*.npy",
                            "global_basis": "global_basis_L*.npy", "save_k": args.save_k,
                            "ridge_lambda": args.ridge}
    (out / "subspace_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[a-subspace] wrote {out}/subspace_summary.json + artifacts")


if __name__ == "__main__":
    main()
