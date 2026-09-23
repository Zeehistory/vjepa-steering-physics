

#!/usr/bin/env python
"""Latent GEOMETRY behind spline acceleration steering — CPU only, no decoder.

Answers, in latent space, the two questions the decode run then tests in pixels:

1. IS THE EDIT'S TIME PROFILE CURVED? For the anchor->target pair, report the per-temporal-token norm of
   ``Delta H`` and the fraction of the edit that a K-knot spline can represent at all
   (``cos(K-knot fit, true)``). K=1 is a constant-in-t edit, i.e. the classical global steering vector;
   if its cosine is well below 1 then no amount of operator-fitting can recover the rest, because the
   parametrization cannot express it.

2. IS THE LATENT ACCELERATION FAMILY CURVED? Each scene's 8 clips form a path through latent space
   indexed by acceleration. Report the turning angle between successive chords (0deg = perfectly
   straight, so linear interpolation would be exact), and run the leave-one-rank-out reconstruction --
   Catmull-Rom vs straight-line -- purely in latent space. This is the cheap preview of Protocol B: if
   the spline does not beat the line here, it will not beat it after decoding either.

Also dumps 2D PCA coordinates of the temporal latent trajectories for the trajectory figure.

    python experiments/threads/acceleration/11_run/accel_spline_geometry.py \
        --test_dir .../moving_ball_scene_accel2d_mixed/test/vjepa2_large \
        --layers 12 --num_scenes 20 --output_dir outputs/analysis/moving_ball_accel2d_mixed/spline_geom
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

from src.analysis import spline_ops as sp
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="12")
    p.add_argument("--knots", default="1,2,3,4,6,8")
    p.add_argument("--num_scenes", type=int, default=20)
    p.add_argument("--pca_scenes", type=int, default=6)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    Ks = [int(x) for x in args.knots.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    ds = LatentDataset(args.test_dir, layers=layers)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]
    grid = tuple(int(x) for x in ds[scenes[sids[0]][0]]["grid"])
    T = grid[0]
    B = {K: sp.spline_basis(T, K) for K in Ks}
    print(f"[geom] {len(sids)} scenes, layers={layers}, grid={grid}", flush=True)

    edit_norm_per_t = {L: [] for L in layers}
    knot_cos = {L: {K: [] for K in Ks} for L in layers}
    turn_angles = {L: [] for L in layers}
    loo = {L: {"spline": [], "lerp": []} for L in layers}
    pca_dump = {}

    for n, s in enumerate(sids):
        ranks = sorted(scenes[s])
        samples = {r: ds[scenes[s][r]] for r in ranks}
        accs = {r: vo.clip_acceleration(samples[r]) for r in ranks}
        for L in layers:
            profs = {r: sp.temporal_profile(vo.layer_flat(samples[r]["layers"][L]), grid) for r in ranks}

            # --- 1. the anchor->target edit and its temporal shape
            dR = profs[ranks[-1]] - profs[ranks[0]]
            edit_norm_per_t[L].append(np.linalg.norm(dR, axis=1))
            for K in Ks:
                knot_cos[L][K].append(vo.cosine(sp.smooth_profile(dR, B[K]).reshape(-1), dR.reshape(-1)))

            # --- 2. curvature of the acceleration family. Within a scene the clips sweep the
            # acceleration DIRECTION through a full turn at roughly fixed magnitude, so the family is a
            # closed LOOP parametrized by angle -- not a magnitude ramp. Order by angle and close it.
            angs = np.array([float(np.arctan2(accs[r][1], accs[r][0])) for r in ranks])
            order = np.argsort(np.mod(angs, 2 * np.pi))
            seq = np.stack([profs[ranks[i]].reshape(-1) for i in order], 0)   # (M, T*D)
            angs_sorted = np.mod(angs[order], 2 * np.pi)
            M = len(order)
            chords = np.stack([seq[(i + 1) % M] - seq[i] for i in range(M)], 0)  # closed
            for i in range(M):
                turn_angles[L].append(np.degrees(np.arccos(np.clip(
                    vo.cosine(chords[i], chords[(i + 1) % M]), -1, 1))))

            # --- leave-one-rank-out reconstruction, spline vs line (latent space, no decoder).
            # On a loop every rank is an interior point, so all M are legitimate interpolation targets.
            for held_pos in range(M):
                keep = [i for i in range(M) if i != held_pos]
                ctrl = seq[keep]
                q = sp.angular_index(angs_sorted[keep], float(angs_sorted[held_pos]))
                truth = seq[held_pos]
                den = np.linalg.norm(truth - ctrl.mean(0)) + 1e-12
                loo[L]["spline"].append(
                    float(np.linalg.norm(sp.catmull_rom_closed(ctrl, q) - truth) / den))
                loo[L]["lerp"].append(
                    float(np.linalg.norm(sp.linear_eval_closed(ctrl, q) - truth) / den))

            # --- 2D PCA of the TEMPORAL trajectory, for the figure
            if n < args.pca_scenes and L == layers[0]:
                stack = np.concatenate([profs[r] for r in ranks], axis=0)     # (M*T, D)
                c = stack - stack.mean(0, keepdims=True)
                _, _, vt = np.linalg.svd(c, full_matrices=False)
                coords = (c @ vt[:2].T).reshape(len(ranks), T, 2)
                pca_dump[f"scene{s:05d}"] = {
                    "ranks": [int(r) for r in ranks],
                    "accel_mag": [float(np.linalg.norm(accs[r])) for r in ranks],
                    "accel": [accs[r].tolist() for r in ranks],
                    "accel_angle_deg": [float(np.degrees(np.arctan2(accs[r][1], accs[r][0]))) for r in ranks],
                    "traj_pca": np.round(coords, 4).tolist(),
                }
        del samples
        gc.collect()
        if (n + 1) % 5 == 0:
            print(f"[geom]   {n+1}/{len(sids)} scenes", flush=True)

    summary = {"test_dir": args.test_dir, "layers": layers, "knots": Ks, "T": T,
               "n_scenes": len(sids), "per_layer": {}}
    for L in layers:
        en = np.stack(edit_norm_per_t[L], 0)
        summary["per_layer"][str(L)] = {
            "edit_norm_per_t_mean": np.round(en.mean(0), 2).tolist(),
            "edit_norm_growth_last_over_first": round(float(en.mean(0)[-1] / (en.mean(0)[0] + 1e-9)), 3),
            "knot_cos_ceiling": {str(K): round(float(np.mean(knot_cos[L][K])), 4) for K in Ks},
            "family_turn_angle_deg_mean": round(float(np.mean(turn_angles[L])), 2),
            "family_turn_angle_deg_median": round(float(np.median(turn_angles[L])), 2),
            "loo_rel_err_spline": round(float(np.mean(loo[L]["spline"])), 4),
            "loo_rel_err_lerp": round(float(np.mean(loo[L]["lerp"])), 4),
            "loo_spline_wins_frac": round(float(np.mean(
                np.asarray(loo[L]["spline"]) < np.asarray(loo[L]["lerp"]))), 3),
            "n_loo": len(loo[L]["spline"]),
        }
    (out / "spline_geometry.json").write_text(json.dumps(summary, indent=2))
    (out / "traj_pca.json").write_text(json.dumps(pca_dump, indent=1))

    for L in layers:
        r = summary["per_layer"][str(L)]
        print(f"\n[geom] L{L}")
        print(f"  |edit| per temporal token : {r['edit_norm_per_t_mean']}")
        print(f"  growth (last/first)       : {r['edit_norm_growth_last_over_first']}x")
        print(f"  K-knot ceiling cos        : {r['knot_cos_ceiling']}")
        print(f"  family turning angle      : mean {r['family_turn_angle_deg_mean']}deg "
              f"median {r['family_turn_angle_deg_median']}deg")
        print(f"  leave-one-out rel err     : spline {r['loo_rel_err_spline']} vs "
              f"lerp {r['loo_rel_err_lerp']}  (spline wins {r['loo_spline_wins_frac']*100:.0f}% of {r['n_loo']})")
    print(f"\n[geom] -> {out}/spline_geometry.json")


if __name__ == "__main__":
    main()
