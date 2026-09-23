

#!/usr/bin/env python
"""Fit the VELOCITY and SPIN command operators on the crosstalk scene, each blind to the other quantity.

Latent-only (no decoder). Produces the three operators that ``experiments/threads/restitution-spin/06_probes/crosstalk_eval.py`` uses to
answer the PI's question -- *does steering one physical quantity move another?*

  ``W_V``  fitted ONLY on pairs whose spin cell is identical, so ``d_omega`` is exactly 0 in every
           training pair. Features: the 27-dim position-aware ``command_features_pos`` that won on
           ``rolling_ball3d``, so the translation half of this experiment stays comparable to that
           baseline.
  ``W_S``  fitted ONLY on pairs whose velocity cell is identical, so ``d_v`` is exactly 0 in every
           training pair. Features: the 12-dim ``spin_command_features``.
  ``W_J``  fitted on the DIAGONAL pairs (both cells differ), on the concatenated 39-dim feature set.
           Not used for the crosstalk measurement itself -- it is the reference for the separate
           question of whether composing two single-quantity operators is as good as fitting the
           two-quantity edit directly.

The blindness is the load-bearing property. If ``W_V`` had ever seen a pair in which omega moved, any
spin leakage it produced at test time could be dismissed as leakage from the fit. Because the factorial
is complete, both training sets exist exactly, with no reweighting and no residualization.

**``--canon`` (position canonicalization) -- switched ON by default, and it is the difference between an
operator that works and one that does not.** The first version of this fit was a single GLOBAL linear
map, and ``experiments/threads/restitution-spin/06_probes/latent_crosstalk.py`` measured it at gain 0.156 with edit norm 0.398 of the true
displacement -- an alignment of 0.39, i.e. ~67 degrees off the true direction, with ~92% of the edit
landing off BOTH physical axes. That is not ridge shrinkage (lambda=1 is negligible against an ``XtX``
accumulated over 12,288 pairs); it is a placement failure, and this project already knows the cause.
``velocity_ops.direction_bin`` says it outright: *"the velocity edit's spatial footprint depends on where
the ball is, so a single global operator cannot place it correctly."* The latent displacement for a
velocity change is a spatially LOCALIZED pattern sitting on the tokens the ball occupies; scenes start
the ball in different cells, so the correct edit lives at a different token offset in every scene. No
linear function of the command features can translate a spatial pattern -- so the global fit regresses
toward the average over start positions, which is a smear that points nowhere in particular.

The fix is the one ``fit_command_operators_accel_canon.py`` already uses on the accel scene: roll every
clip's token grid so the ball's frame-0 cell sits at the grid centre BEFORE differencing, fit in that
canonical frame, and roll the predicted edit back by ``-shift`` at test time. All ranks in a scene share
``pos0``, so it is one shift per scene and ``roll(H_b) - roll(H_a) == roll(dH)`` exactly.

**``--cond_dim k`` (scene-conditioned operator).** A command-only operator is a function of the command
alone, so it must emit the SAME edit for the same command in every scene, and its ceiling is how well
the across-scene average edit represents any individual scene (measured directly by
``experiments/threads/restitution-spin/04_operators/operator_ceiling.py``). If that ceiling is low, no amount of refitting helps and the
model class itself has to change.

Conditioning is the change. The features gain ``z``, a ``k``-dim random projection of the BASE clip's
latent, plus ``outer(z, dv)``, which is what lets the map apply a scene-dependent transform to the
command instead of one global transform. ``z`` is drawn from clip ``a`` only -- never ``H_b`` -- so this
remains a command-only operator at test time in the sense that matters: nothing about the target clip
is used. A random projection rather than a PCA basis, deliberately: it needs no extra pass over the
data, and Johnson-Lindenstrauss says a random ``k``-dim projection preserves the relative geometry that
distinguishes scenes, which is all the conditioning needs. The projection matrix is saved with the
operator, because an operator applied under a DIFFERENT random basis than it was fit with would produce
confident nonsense.

Also saves the raw normal equations (``XtX``, ``XtY``) alongside each operator. Ridge enters only at
``solve()``, so with these on disk any lambda can be re-solved in seconds instead of re-running the
12-minute fit.

    PYTHONPATH=. python experiments/threads/restitution-spin/04_operators/fit_spin_operators.py \
        --train_dir .../latents/spin_ball3d/train/vjepa2_large \
        --layers 6,12,18,23 --output_dir .../analysis/spin_ball3d/operators
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

VEL_DIM = vo.COMMAND_FEATURE_DIM_POS      # 27
SPIN_DIM = so.SPIN_FEATURE_DIM            # 12


def _scene_cells(scenes: dict[int, dict[int, int]], s: int, n_spin: int,
                 n_vel: int) -> dict[tuple[int, int], int]:
    """``(vel_index, spin_index) -> dataset index`` for one scene, decoded from the rank convention.

    Incomplete scenes are rejected outright rather than fitted on what happens to be present: a missing
    cell would silently unbalance the factorial, and an unbalanced factorial reintroduces exactly the
    velocity/spin correlation the design exists to eliminate.
    """
    cells = {divmod(int(rank), n_spin): idx for rank, idx in scenes[s].items()}
    if len(cells) != n_vel * n_spin:
        raise ValueError(f"scene {s}: {len(cells)} cells, expected the full {n_vel}x{n_spin} factorial")
    return cells


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--num_scenes", type=int, default=0, help="0 = all scenes in the cache")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--n_vel", type=int, default=4)
    p.add_argument("--n_spin", type=int, default=4)
    p.add_argument("--max_cached_shards", type=int, default=2)
    p.add_argument("--canon", type=int, default=1,
                   help="1 = fit in the ball-centred (position-canonicalized) frame; 0 = old global fit")
    p.add_argument("--cond_dim", type=int, default=0,
                   help="k>0 = SCENE-CONDITIONED operator: augment the command features with a k-dim "
                        "random projection of the BASE clip's latent (see module docstring)")
    p.add_argument("--cond_basis", default="",
                   help="npz from experiments/threads/restitution-spin/04_operators/fit_scene_basis.py. Swaps the random projection for a "
                        "PCA basis of the scene distribution and changes NOTHING else, so an A/B at "
                        "the same --cond_dim isolates the basis from capacity, ridge, layer and frame.")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Bounded shard cache: this pass streams the WHOLE split, and the default unbounded cache grows
    # to the entire 4096-clip x 4-layer set (~136 GB) and gets OOM-killed. 2 is enough to read each
    # shard exactly once, since a shard holds 128 clips = 8 consecutive scenes and scenes are visited
    # in order.
    ds = LatentDataset(args.train_dir, layers=layers, max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)
    if args.num_scenes:
        sids = sids[:args.num_scenes]
    print(f"[fit] {len(sids)} scenes, layers={layers}", flush=True)

    fits: dict[str, dict[int, vo.LinearLS]] = {}
    counts = {"vel": 0, "spin": 0, "both": 0}
    k = int(args.cond_dim)
    # Conditioning adds z (k) and outer(z, dv) (2k) to the velocity features, and z plus z*dw (k + k)
    # to the spin ones -- in each case the raw scene code plus its interaction with the commanded step.
    cond_v, cond_s = (k + 2 * k, k + k) if k else (0, 0)
    dims = {"vel": VEL_DIM + cond_v, "spin": SPIN_DIM + cond_s,
            "both": VEL_DIM + cond_v + SPIN_DIM + cond_s}
    # One projection per layer, fixed seed, saved with the operator. The projection is over a layer's
    # flat dimension, which is not known until the first sample is read, so it is built lazily.
    proj: dict[int, np.ndarray] = {}
    proj_mu: dict[int, np.ndarray] = {}
    proj_rng = np.random.default_rng(12345)
    if args.cond_basis:
        bz = np.load(args.cond_basis)
        bk = int(bz["k"][0])
        if bk != k:
            # Silently conditioning on a different number of dimensions than requested would make the
            # A/B against the random-basis operator a comparison of two things at once.
            raise SystemExit(f"--cond_basis has k={bk} but --cond_dim={k}")
        for L in layers:
            if f"P_{L}" not in bz.files:
                raise SystemExit(f"--cond_basis lacks P_{L}; it was built for other layers")
            proj[L] = bz[f"P_{L}"].astype(np.float64)
            proj_mu[L] = bz[f"mu_{L}"].astype(np.float64)
        print(f"[fit] conditioning on the PCA basis {Path(args.cond_basis).name} "
              f"(explained variance {float(bz['explained_variance_ratio'][0]):.3f})", flush=True)

    def _code(flat_by_layer: dict[int, np.ndarray]) -> np.ndarray:
        """k-dim scene code for a base clip: random projection of its latent, scale-normalized.

        Normalized by the latent's own norm so the code describes the scene's DIRECTION in latent space
        rather than its magnitude -- otherwise a globally brighter scene would shift every coefficient
        and the conditioning would spend its capacity on overall scale.
        """
        parts = []
        for L in sorted(flat_by_layer):
            f = flat_by_layer[L]
            if L not in proj:
                proj[L] = proj_rng.standard_normal((f.shape[0], k)) / np.sqrt(f.shape[0])
            # The PCA basis is defined on CENTRED latents, so the same mean must be removed here or the
            # code is dominated by the (large, scene-independent) mean latent's coordinates and carries
            # almost no scene information -- the opposite of the point.
            parts.append(((f - proj_mu[L]) if L in proj_mu else f) @ proj[L])
        z = np.mean(parts, axis=0)
        return z / (np.linalg.norm(z) + 1e-12)

    for n, s in enumerate(sids):
        cells = _scene_cells(scenes, s, args.n_spin, args.n_vel)
        pairs = so.enumerate_pairs(cells)
        # Load this scene's clips ONCE and form every pair in RAM: the pair count per scene (240) is
        # 15x the clip count (16), so re-reading per pair would turn a 190 GB pass into a 2.8 TB one.
        samples = {idx: ds[idx] for idx in cells.values()}
        flats = {idx: {L: vo.layer_flat(samples[idx]["layers"][L]) for L in layers} for idx in samples}

        # Position canonicalization. Every rank in a scene shares the frame-0 ball centre, so ONE shift
        # serves the whole scene, and rolling each clip once is equivalent to rolling every pair's dH
        # (roll is linear: roll(H_b) - roll(H_a) == roll(H_b - H_a)) at 1/15th the cost, since a scene has
        # 240 pairs but 16 clips. Rolling here means the fit sees every scene's edit at the same token
        # location instead of averaging a moving spatial pattern into a smear.
        if args.canon:
            s_ref = samples[sorted(samples)[0]]
            grid = tuple(int(x) for x in s_ref["grid"])
            sh = vo.canon_shift(vo.clip_start_pos(s_ref), grid)
            flats = {idx: {L: vo.roll_layer(f, grid, sh) for L, f in per_l.items()}
                     for idx, per_l in flats.items()}

        # --- accumulate this scene's normal equations ------------------------------------------------
        # The naive route -- materialise ``dH = H_b - H_a`` per pair and rank-1 update a (p x 2.1M)
        # matrix -- is memory-bandwidth bound and takes hours: a scene has 240 pairs but only 16 clips,
        # so every clip's 2.1M-dim latent gets touched 15 times over.
        #
        # It is exactly refactorable. Writing X for the (n_pairs x p) feature matrix,
        #     X^T dH = sum_p x_p (f_{b(p)} - f_{a(p)})^T = sum_clips c_clip f_clip^T = C^T F
        # with ``c_clip = sum_{p: b(p)=clip} x_p - sum_{p: a(p)=clip} x_p``. So we accumulate a tiny
        # (16 x p) coefficient matrix over the pairs, then do ONE (p x 16)(16 x 2.1M) BLAS matmul per
        # layer per scene. Same numbers, ~15x less traffic through the big array.
        idx_list = sorted(samples)
        pos_of = {idx: j for j, idx in enumerate(idx_list)}
        feat_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

        def _feats(ia: int, ib: int) -> dict[str, np.ndarray]:
            if (ia, ib) not in feat_cache:
                sa, sb = samples[ia], samples[ib]
                va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb)
                wa, wb = so.clip_spin(sa), so.clip_spin(sb)
                fv = vo.command_features_pos(va, vb, vo.clip_start_pos(sa))
                fs = so.spin_command_features(wa, wb, so.clip_phi0(sa))
                if k:
                    # Scene code from clip a ONLY. Using clip b here would leak the target into the
                    # operator and make every downstream steering number meaningless.
                    z = _code(flats[ia])
                    fv = np.concatenate([fv, z, np.outer(z, np.asarray(vb) - np.asarray(va)).ravel()])
                    fs = np.concatenate([fs, z, z * (float(wb) - float(wa))])
                feat_cache[(ia, ib)] = {"vel": fv, "spin": fs, "both": np.concatenate([fv, fs])}
            return feat_cache[(ia, ib)]

        for kind, plist in pairs.items():
            if not plist:
                continue
            p_dim = dims[kind]
            C = np.zeros((len(idx_list), p_dim))
            XtX = np.zeros((p_dim, p_dim))
            for ia, ib in plist:
                x = _feats(ia, ib)[kind]
                XtX += np.outer(x, x)
                C[pos_of[ib]] += x
                C[pos_of[ia]] -= x
                counts[kind] += 1

            if kind not in fits:
                fits[kind] = {L: vo.LinearLS(p_dim, flats[idx_list[0]][L].shape[0], args.ridge)
                              for L in layers}
            for L in layers:
                F = np.stack([flats[i][L] for i in idx_list])          # (16, n_tok*D)
                op = fits[kind][L]
                op.XtX += XtX
                op.XtY += C.T @ F
                op.n += len(plist)
                del F

        del samples, flats, feat_cache
        if (n + 1) % 10 == 0:
            print(f"[fit] scene {n + 1}/{len(sids)}  pairs={counts}", flush=True)

    meta = {"layers": layers, "num_scenes": len(sids), "ridge": args.ridge,
            "canon": bool(args.canon), "cond_dim": k,
            "canon_note": ("edits are predicted in the BALL-CENTRED frame; a consumer MUST roll the "
                           "prediction back by -canon_shift(start_pos) before adding it to a latent"),
            "pair_counts": counts, "dims": dims,
            "feature_sets": {"vel": "velocity_ops.command_features_pos (27)",
                             "spin": "spin_ops.spin_command_features (12)",
                             "both": "concat(vel, spin) (39)"}}
    for kind in fits:
        arrs = {}
        for L in layers:
            arrs[f"B_{L}"] = fits[kind][L].solve().astype(np.float32)
            # Ridge enters ONLY at solve(), so persisting the normal equations makes the whole lambda
            # axis a seconds-long re-solve instead of a 12-minute refit. XtX is (p x p) and free; XtY is
            # the size of B itself, which is the real cost and still worth it.
            arrs[f"XtX_{L}"] = fits[kind][L].XtX.astype(np.float64)
            arrs[f"XtY_{L}"] = fits[kind][L].XtY.astype(np.float32)
            # The conditioning basis travels WITH the operator: applying it under a different random
            # projection than it was fit under yields confident nonsense, and nothing downstream could
            # detect that from the numbers alone.
            if k:
                arrs[f"P_{L}"] = proj[L].astype(np.float32)
                # The centring vector travels with the basis for the same reason the basis travels
                # with the operator: a PCA code rebuilt without it is not a smaller error than one
                # rebuilt under the wrong projection, and nothing downstream could detect either.
                if L in proj_mu:
                    arrs[f"mu_{L}"] = proj_mu[L].astype(np.float32)
        # Uncompressed: these are dense float32 operators (up to 1.3 GB for the 39-dim joint one) and
        # deflate buys almost nothing on them while dominating the runtime of the whole fit.
        np.savez(out / f"operator_{kind}.npz", **arrs)
        print(f"[fit] wrote operator_{kind}.npz  ({counts[kind]} pairs)", flush=True)
    (out / "operators_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[fit] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
