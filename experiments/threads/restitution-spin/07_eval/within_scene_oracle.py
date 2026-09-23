

#!/usr/bin/env python
"""The ceiling for ANY scene-conditioned velocity operator, measured by fitting one scene at a time.

**The question this exists to settle.** ``operator_ceiling.py`` bounds COMMAND-ONLY maps (0.590), and
conditioning legitimately beats that bound -- 0.477 at k=8 rising to 0.660 at k=512, still climbing at
roughly +0.025 per doubling of ``k``. Two very different worlds produce that curve:

  * the amortisation is the bottleneck -- a random ``k``-dim projection is a weak way to tell one scene
    from another, and a scene-conditioned map that actually knew the scene would do far better. Then
    the fix is a better scene code, not more capacity.
  * the ``dH`` for a velocity command is simply not a smooth function of the command even WITHIN a
    scene. Then no conditioning scheme helps, 0.80 is not available, and the honest answer is that the
    steering bar cannot be met by an operator of this kind at all.

This measures the second directly, which upper-bounds the first. Each test scene is a complete 4x4
velocity x spin factorial, so a scene can be fitted on its OWN pairs -- the strongest possible form of
scene conditioning, an operator handed that scene's data outright.

**Held out by velocity CELL, not by pair.** Leaving out a single pair leaves both of its velocity cells
present in the training pairs, so the fit has already seen every command it is asked about and the
number would be an interpolation score dressed up as generalisation. Holding out a whole velocity index
means the evaluated command is one no training pair contained. That is the same thing the deployed
operator faces on an unseen scene, minus the transfer.

**The first version of this script fitted each scene FROM SCRATCH, and its numbers must not be reused.**
It scored 0.281 against the deployed operator's 0.669 -- and a ceiling below an achieved value
falsifies the ceiling, not the operator, exactly as ``operator_ceiling.py`` records happening to its own
first version. The error: a single scene supplies 24 training pairs against 27 features, extrapolating
to a velocity cell outside their span, whereas the deployed operator is estimated from 256 scenes and
borrows strength across all of them. That is not an oracle, it is a data-starved fit, and it bounds
nothing.

**What replaces it: shrinkage toward the global operator.** Each scene starts from the global fit and
is allowed to move away from it only as far as its own pairs justify,

    B_s = (Xs'Xs + lam I)^-1 (Xs'Ys + lam B_global)

which is the closed form for ``argmin ||Xs B - Ys||^2 + lam ||B - B_global||^2``. The two ends of the
sweep are both known quantities, which is what makes the middle interpretable: ``lam -> inf`` returns
the global operator exactly (so the curve must start at its alignment, a built-in correctness check),
and ``lam -> 0`` returns the discredited from-scratch fit. Anything above the ``lam -> inf`` end is real
adaptation rather than an artefact of the sweep.

This is also the deployable version of the question. It is the latent-space analogue of the on-rig
re-identification that took the robotics action loop from 54.8% to 94.7%, and it is a strictly larger
model class than the gain calibration that ``latent_crosstalk.py`` shows to be capped at the operator's
alignment: rescaling one operator cannot exceed 0.41, whereas refitting it per scene is not bounded
that way. The practical reading is "given a handful of real observations in a new scene, how well can
velocity be steered there?"

**Ridge is standardized here** (``A = XtX + lambda diag(XtX)``) for the reason
``resolve_ridge_sweep.py`` documents at length: ``command_features_pos`` spans six orders of magnitude
in scale, and a uniform penalty on such features shrinks anisotropically, which rotates the edit rather
than merely shortening it. Lambda is swept and the curve is reported in full, since with 24 training
pairs against 27 features the regularisation genuinely matters and a single guessed value would be the
weakest link in the conclusion.

Reported per scene and aggregated as a median:

  ``align``   cos between the predicted edit and the true held-out displacement -- the number to hold
              against the 0.80 bar, and against the deployed operator's 0.660.
  ``gain``    the same edit as a fraction of the true displacement along it, for continuity with
              ``latent_crosstalk.py``.
  ``leak``    its component along the orthogonalised spin direction, so a scene-conditioned operator
              that buys alignment by dragging spin is visible rather than hidden.

    PYTHONPATH=. python experiments/threads/restitution-spin/07_eval/within_scene_oracle.py \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --layers 18 --out .../analysis/spin_ball3d/within_scene_oracle.json
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="18")
    ap.add_argument("--num_scenes", type=int, default=24)
    ap.add_argument("--n_vel", type=int, default=4)
    ap.add_argument("--n_spin", type=int, default=4)
    ap.add_argument("--max_cached_shards", type=int, default=2)
    ap.add_argument("--lambdas", default="1e-4,1e-2,1,10,100,1000,10000,1e6")
    ap.add_argument("--canon", type=int, default=1)
    ap.add_argument("--global_operators", required=True,
                    help="operators dir supplying B_global (the shrinkage target). Its XtX/XtY are "
                         "re-solved here at --global_ridge rather than reusing its stored B_, so the "
                         "prior is the BEST global fit and not whatever lambda it happened to ship.")
    ap.add_argument("--global_ridge", type=float, default=1e-4)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    L = layers[0]
    lams = [float(x) for x in args.lambdas.split(",") if x]

    # The shrinkage target. Re-solved from the persisted normal equations at the ridge the sweep in
    # resolve_ridge_sweep.py selected, since the stored B_ was solved at the fit's own lambda=1.
    gz = np.load(Path(args.global_operators) / f"operator_vel.npz")
    gXtX, gXtY = gz[f"XtX_{L}"].astype(np.float64), gz[f"XtY_{L}"]
    if gXtX.shape[0] != vo.COMMAND_FEATURE_DIM_POS:
        # A conditioned operator's features include the scene code, which this per-scene refit does not
        # build. Rather than silently fit a different feature set than the prior was estimated on --
        # which would make the lam->inf end of the sweep disagree with the global operator and destroy
        # the correctness check the whole design rests on -- refuse.
        raise SystemExit(f"--global_operators must be a command-only (cond_dim=0) fit; "
                         f"got p={gXtX.shape[0]}")
    B_global = np.linalg.solve(gXtX + args.global_ridge * np.eye(gXtX.shape[0]),
                               gXtY).astype(np.float32)
    del gz, gXtY
    print(f"[oracle] B_global from {Path(args.global_operators).name} "
          f"at ridge {args.global_ridge:g}: {B_global.shape}", flush=True)

    ds = LatentDataset(args.test_dir, layers=layers, max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]
    print(f"[oracle] {len(sids)} scenes, layer {L}, lambdas={lams}", flush=True)

    per_lam: dict[float, list[dict]] = {lam: [] for lam in lams}
    for n, s in enumerate(sids):
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        sam = {c: ds[i] for c, i in cells.items()}
        grid = tuple(int(x) for x in sam[(0, 0)]["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sam[(0, 0)]), grid)

        def flat(c):
            f = vo.layer_flat(sam[c]["layers"][L])
            return vo.roll_layer(f, grid, sh) if args.canon else f

        H = {c: flat(c) for c in sam}                      # 16 x 2.1M, ~134 MB

        def feats(ca, cb):
            return vo.command_features_pos(vo.clip_velocity(sam[ca]), vo.clip_velocity(sam[cb]),
                                           vo.clip_start_pos(sam[ca]))

        for held in range(args.n_vel):
            # Train: every matched-spin velocity pair whose BOTH cells avoid the held-out velocity index.
            train = [((va, sp), (vb, sp))
                     for sp in range(args.n_spin)
                     for va in range(args.n_vel) for vb in range(args.n_vel)
                     if va != vb and va != held and vb != held]
            # Test: a pair that moves INTO the held-out velocity cell, at spin 0, from velocity cell 0
            # (or 1 when 0 is the held-out one) -- a command no training pair contained.
            src = 0 if held != 0 else 1
            ca, cb = (src, 0), (held, 0)
            X = np.stack([feats(a, b) for a, b in train])              # (24, 27)
            Y = np.stack([H[b] - H[a] for a, b in train])              # (24, 2.1M)
            XtX = X.T @ X
            XtY = X.T @ Y
            del X, Y

            D_vel = H[cb] - H[ca]
            D_spin = H[(src, args.n_spin - 1)] - H[ca]
            nv = np.linalg.norm(D_vel)
            u_v = D_vel / (nv + 1e-12)
            perp = D_spin - (D_spin @ u_v) * u_v
            n_perp = np.linalg.norm(perp)
            u_s = perp / (n_perp + 1e-12)
            x = feats(ca, cb)

            for lam in lams:
                # Shrinkage toward B_global, not toward zero: the lam*B_global term in the right-hand
                # side is the whole difference between this and the discredited from-scratch fit.
                B = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]),
                                    XtY + lam * B_global)
                e = x @ B
                en = np.linalg.norm(e)
                per_lam[lam].append({"scene": int(s), "held": held,
                                     "align": float(e @ u_v / (en * 1.0 + 1e-12)),
                                     "gain": float(e @ u_v / (nv + 1e-12)),
                                     "leak": float(e @ u_s / (n_perp + 1e-12)),
                                     "norm_rel": float(en / (nv + 1e-12))})
            del XtX, XtY
        del sam, H
        if (n + 1) % 4 == 0:
            print(f"[oracle] scene {n + 1}/{len(sids)}", flush=True)

    summary = {"layer": L, "canon": bool(args.canon), "n_scenes": len(sids),
               "n_folds_total": len(per_lam[lams[0]]), "lambdas": lams, "by_lambda": {}}
    for lam in lams:
        rr = per_lam[lam]
        summary["by_lambda"][f"{lam:g}"] = {
            k: round(float(np.median([r[k] for r in rr])), 4)
            for k in ("align", "gain", "leak", "norm_rel")}
    best = max(summary["by_lambda"], key=lambda k: summary["by_lambda"][k]["align"])
    summary["best_lambda"] = best
    summary["oracle_align"] = summary["by_lambda"][best]["align"]
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": per_lam[float(best)]}, indent=1))

    print(f"\n# Within-scene oracle, layer {L} "
          f"({summary['n_folds_total']} held-out-velocity-cell folds)\n")
    print("| lambda | align | gain | leak | norm_rel |")
    print("|---|---|---|---|---|")
    for lam in lams:
        d = summary["by_lambda"][f"{lam:g}"]
        mark = "  <- best" if f"{lam:g}" == best else ""
        print(f"| {lam:g} | {d['align']:.3f} | {d['gain']:.3f} | {d['leak']:+.3f} | "
              f"{d['norm_rel']:.3f} |{mark}")
    print(f"\nbest per-scene-adapted alignment = {summary['oracle_align']:.3f} at lambda {best}")
    print("CHECK: the largest lambda must reproduce the global operator's alignment -- that end of the")
    print("sweep IS B_global by construction, so a disagreement there means the prior or the feature")
    print("set is wired up wrong and no other cell in this table can be trusted.")
    print("Anything ABOVE that end is genuine per-scene adaptation: a strictly larger model class than")
    print("gain calibration, which latent_crosstalk.py caps at the operator's own alignment.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
