

#!/usr/bin/env python
"""Crosstalk measured in the FULL latent space, basis-free. The PCA figure cannot do this job.

**Why this script exists.** ``pca_viz.py`` projects steering displacements onto probe axes inside a
top-3 PCA and reports how far each edit travels along the "velocity" and "spin" directions. That is a
good picture and a bad measurement, for a reason its own control exposes: ``gt_vel`` is a REAL clip
differing from the base only in its velocity cell, so its true spin change is exactly zero -- and the
projection attributes 0.366 of a full spin displacement to it. The top-3 components hold only ~47% of
the variance while most of every displacement lies off that plane, so the projection mixes the axes.
Any crosstalk number read off it is measuring the truncation.

**What is measured here instead.** No PCA, no probe axes, no truncation. Per scene, the two REAL
displacements are the definition of the two directions:

    D_vel  = H(vi_b, si_a) - H(vi_a, si_a)     what changing ONLY velocity does to the latent
    D_spin = H(vi_a, si_b) - H(vi_a, si_a)     what changing ONLY spin does

Gram-Schmidt gives an orthonormal frame ``(u_v, u_s)`` with ``u_v`` along ``D_vel`` and ``u_s`` the
part of ``D_spin`` orthogonal to it. Then for an edit ``d``:

    gain = (d . u_v) / |D_vel|          fraction of the real velocity displacement achieved
    leak = (d . u_s) / |D_spin_perp|    fraction of the (orthogonalised) real spin displacement caused

Both are dimensionless, both are 1.0 when the edit exactly reproduces the corresponding real change,
and the roles swap for the spin operator. Because the frame is built per scene from that scene's own
ground truth, no global basis and no dimensionality reduction enters.

``norm_rel_vel`` is reported alongside because gain alone cannot distinguish an operator that points
the wrong way from one that is merely too small, and the two call for opposite fixes. Dividing gain by
it gives the ALIGNMENT (cosine), and that comparison settled the question here:

    velocity operator:  gain 0.156, norm 0.398  ->  alignment 0.406
    the two agree to a ratio of 1.011

An unbiased least-squares predictor with correlation ``r`` shrinks its output by ``r`` AND aligns at
``r``, so alignment == norm-ratio is the textbook signature of a correctly-fit regression whose model
class explains ``r^2 ~ 16%`` of the displacement variance. Nothing is broken in the fit. The practical
consequence is sharp: rescaling changes gain and leak together, so the **maximum gain any calibration
of a given operator can reach is its alignment** -- 0.41 here, which is below an 80-90% steering bar by
construction. Gain calibration (the fix that took the robotics loop from 54.8% to 94.7%) therefore
cannot work on this operator, and the only lever left is a model class with higher alignment.

CAVEAT, and it matters before concluding anything about steering: latent cosine is a CONSERVATIVE
proxy. Much of the unexplained displacement may be nuisance that the decoder ignores, in which case an
edit with modest alignment still renders the intended physics -- this repo's own 2-D velocity pipeline
reached 5.73 deg held-out heading error, which no latent cosine would have predicted. The pixel
measurement, not this one, is what decides whether steering works.

**The controls that make the numbers interpretable**, reported alongside:

* ``gt_vel`` and ``gt_spin`` themselves. By construction ``gt_vel`` must score gain 1.0 and leak 0.0,
  and ``gt_spin`` the reverse. These are not decoration -- they are the proof that the frame is sound,
  and they are exactly what the PCA basis failed.
* ``cos(D_vel, D_spin)``: how non-orthogonal the two real directions are in the first place. If the
  latent space itself entangles the quantities, perfect selectivity is not available to ANY operator,
  and that is a property of the representation rather than a failure of the edit.
* a norm-matched random edit, which here (unlike in the PCA view) is a real control: in the full space
  a random direction's projection onto a fixed 2-D frame is the honest noise floor.

    PYTHONPATH=. python experiments/threads/restitution-spin/06_probes/latent_crosstalk.py \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --operators_dir .../analysis/spin_ball3d/operators \
        --out .../analysis/spin_ball3d/latent_crosstalk.json --layers 6,12,18,23
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

import sys
from pathlib import Path as _P

import numpy as np

from src.analysis import spin_ops as so
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--operators_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--num_scenes", type=int, default=64)
    p.add_argument("--n_vel", type=int, default=4)
    p.add_argument("--n_spin", type=int, default=4)
    p.add_argument("--max_cached_shards", type=int, default=2)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    ds = LatentDataset(args.test_dir, layers=layers, max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]

    ops = {k: {L: np.load(Path(args.operators_dir) / f"operator_{k}.npz")[f"B_{L}"].astype(np.float64)
               for L in layers} for k in ("vel", "spin")}
    # A canonicalized operator predicts the edit in the BALL-CENTRED frame. Adding that to a real latent
    # without rolling it back lands the edit wherever the grid centre happens to be, which for most scenes
    # is nowhere near the ball -- it would score near zero and read as "the operator failed". The flag is
    # read from the fit's own metadata rather than passed in, so an operator cannot be misapplied by
    # forgetting a command-line switch.
    meta_p = Path(args.operators_dir) / "operators_meta.json"
    fit_meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    canon = bool(fit_meta.get("canon", False))
    cond_dim = int(fit_meta.get("cond_dim", 0))
    # The conditioning projection must be the one the operator was FIT under, so it is read from the
    # operator file rather than regenerated from a seed.
    projs = {}
    if cond_dim:
        z_npz = np.load(Path(args.operators_dir) / "operator_vel.npz")
        projs = {L: z_npz[f"P_{L}"].astype(np.float64) for L in layers}
        # PCA-basis operators additionally carry the centring vector the basis was defined on.
        mus = {L: z_npz[f"mu_{L}"].astype(np.float64) for L in layers if f"mu_{L}" in z_npz.files}
    print(f"[xtalk-latent] {len(sids)} scenes, layers={layers}, canon={canon}, "
          f"cond_dim={cond_dim}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for n, s in enumerate(sids):
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        vi_a, si_a, vi_b, si_b = 0, 0, args.n_vel - 1, args.n_spin - 1
        sq = so.commutation_square(cells, vi_a, si_a, vi_b, si_b)
        sam = {k: ds[i] for k, i in sq.items()}

        def flat(key):
            return np.concatenate([vo.layer_flat(sam[key]["layers"][L]) for L in layers])

        base = flat("base")
        D_vel = flat("vel_only") - base
        D_spin = flat("spin_only") - base

        va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
        wa, wb = so.clip_spin(sam["base"]), so.clip_spin(sam["spin_only"])
        fv = vo.command_features_pos(va, vb, vo.clip_start_pos(sam["base"]))
        fs = so.spin_command_features(wa, wb, so.clip_phi0(sam["base"]))
        grid = tuple(int(x) for x in sam["base"]["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sam["base"]), grid)
        inv = (-sh[0], -sh[1])

        def _apply(op_by_layer, feats):
            """Predict the edit per layer, undoing canonicalization when the fit used it."""
            parts = []
            for L in layers:
                e = feats @ op_by_layer[L]
                parts.append(vo.roll_layer(e, grid, inv) if canon else e)
            return np.concatenate(parts)

        if cond_dim:
            # Same scene code the fit used: projection of the BASE clip's latent, per layer, averaged
            # and unit-normalized. Built in the SAME frame the operator was fit in -- canonical if the
            # fit was canonical -- since the code is a function of the latent and the roll changes it.
            base_by_layer = {}
            for L in layers:
                f = vo.layer_flat(sam["base"]["layers"][L])
                base_by_layer[L] = vo.roll_layer(f, grid, sh) if canon else f
            z = np.mean([(base_by_layer[L] - mus[L] if L in mus else base_by_layer[L]) @ projs[L]
                         for L in sorted(layers)], axis=0)
            z = z / (np.linalg.norm(z) + 1e-12)
            fv = np.concatenate([fv, z, np.outer(z, np.asarray(vb) - np.asarray(va)).ravel()])
            fs = np.concatenate([fs, z, z * (float(wb) - float(wa))])

        eV = _apply(ops["vel"], fv)
        eS = _apply(ops["spin"], fs)
        r = rng.standard_normal(eV.shape)
        eR = r * (np.linalg.norm(eV) / (np.linalg.norm(r) + 1e-12))

        # Orthonormal frame from the scene's own ground truth.
        nv = np.linalg.norm(D_vel)
        u_v = D_vel / (nv + 1e-12)
        perp = D_spin - (D_spin @ u_v) * u_v
        n_perp = np.linalg.norm(perp)
        u_s = perp / (n_perp + 1e-12)
        cos_vs = float(D_vel @ D_spin / (nv * np.linalg.norm(D_spin) + 1e-12))

        def score(d):
            return {"gain": float(d @ u_v / (nv + 1e-12)),
                    "leak": float(d @ u_s / (n_perp + 1e-12)),
                    "norm_rel_vel": float(np.linalg.norm(d) / (nv + 1e-12))}

        rows.append({"scene": int(s), "cos_Dvel_Dspin": cos_vs,
                     "V": score(eV), "S": score(eS), "rand": score(eR),
                     "gt_vel": score(D_vel), "gt_spin": score(D_spin),
                     "V_plus_S": score(eV + eS)})
        if (n + 1) % 10 == 0:
            print(f"[xtalk-latent] scene {n + 1}/{len(sids)}", flush=True)

    def med(name, key):
        return float(np.median([r[name][key] for r in rows]))

    summary = {"n_scenes": len(rows), "layers": layers,
               "cos_Dvel_Dspin_median": float(np.median([r["cos_Dvel_Dspin"] for r in rows]))}
    for name in ("V", "S", "V_plus_S", "rand", "gt_vel", "gt_spin"):
        summary[name] = {"gain_vs_Dvel": round(med(name, "gain"), 4),
                         "leak_vs_Dspin_perp": round(med(name, "leak"), 4),
                         "norm_rel_Dvel": round(med(name, "norm_rel_vel"), 4)}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print(f"\n# Latent crosstalk, full space ({len(rows)} scenes)\n")
    print(f"cos(D_vel, D_spin) = {summary['cos_Dvel_Dspin_median']:+.3f} "
          f"(how entangled the two REAL directions already are)\n")
    print("| edit | along velocity dir | along spin dir (orth) | ")
    print("|---|---|---|")
    for name in ("gt_vel", "gt_spin", "V", "S", "V_plus_S", "rand"):
        d = summary[name]
        print(f"| `{name}` | {d['gain_vs_Dvel']:+.3f} | {d['leak_vs_Dspin_perp']:+.3f} |")
    print("\nCONTROLS: gt_vel must read (1.000, 0.000) and gt_spin (0.000, 1.000) by construction.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
