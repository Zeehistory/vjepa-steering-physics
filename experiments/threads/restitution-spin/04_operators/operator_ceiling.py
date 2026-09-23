

#!/usr/bin/env python
"""How well could ANY command-only operator steer this latent? The ceiling, measured directly.

Two hypotheses have now been tested and rejected as explanations for the velocity operator's low
alignment (0.156 gain at 0.398 edit norm, i.e. ~0.39 aligned with the true displacement):

  * **ridge shrinkage** -- rejected by arithmetic: lambda=1 is negligible against an ``XtX`` summed
    over 12,288 pairs, and the edit's NORM is 0.398 of truth rather than the ~0.16 pure shrinkage
    would imply. Shrinkage scales an edit down; it does not rotate it.
  * **spatial placement** -- rejected by experiment: fitting in the ball-centred frame
    (``canon_shift`` + ``roll_layer``, the fix that worked on the accel scene) moved gain 0.156 ->
    0.108, slightly WORSE, with leak/gain unchanged at -0.50.

So before proposing a third fix, measure whether a fix is available at all. A command-only operator is
a function of the command alone: two scenes issued the SAME command must receive the SAME edit. The
ceiling is therefore set by how well displacements agree BETWEEN SCENES THAT SHARE A COMMAND. If two
scenes given identical commands need different edits, no function of the command can serve both, and
that gap bounds every command-only model -- linear or otherwise.

**The first version of this script got that wrong, and its numbers must not be reused.** It compared
each scene's displacement against the leave-one-out MEAN over all other scenes and reported ~0.00 at
every layer. That is not a ceiling. The corner cells ``(0,0) -> (n_vel-1, 0)`` name the same GRID
POSITION in every scene but not the same velocity COMMAND, because each scene draws its own velocity
values -- so the mean averaged unlike commands and cancelled the signal. The tell was immediate and
arithmetic: the fitted operator scores alignment 0.41 on held-out scenes, and nothing can beat its own
ceiling. A ceiling below the achieved value falsifies the ceiling, not the operator.

What is measured instead, per quantity:

  * ``cmd_spread``       -- how much the corner command actually varies across scenes. Had it been
    constant, the old metric would have been valid; this reports the fact rather than assuming it.
  * ``ceiling_nn``       -- for each scene, the cosine against the scene whose COMMAND is nearest in
    feature space. Two near-identical commands that need near-orthogonal edits cannot both be served,
    so this bounds any command-only function, not merely a linear one.

    **It is on an r-SQUARED scale and must not be compared directly to an operator's alignment.**
    Writing each displacement as ``signal(command) + scene noise`` with correlation ``r``, an operator
    predicts the clean signal and so aligns at ``~r``, whereas this statistic correlates two NOISY
    displacements and lands at ``~r^2``. ``ceiling_nn_sqrt`` is therefore reported alongside and is the
    number to hold against alignment. Skipping that conversion makes a sound operator look like it beat
    its own ceiling -- which is how the first version of this script was caught.
  * ``ceiling_nn_canon`` -- the same in the ball-centred frame.
  * ``shared_component`` -- the old leave-one-out-mean number, kept but named honestly. It measures how
    much of the edit is command-INDEPENDENT, which is a real quantity and simply not a ceiling.

If ``ceiling_nn`` sits near the operator's achieved 0.41, the model class is the limit and the way
forward is a scene-conditioned operator. If it is far above, there is headroom in the fit itself.
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


def _nn_cos(mats: np.ndarray, cmds: np.ndarray) -> np.ndarray:
    """Cosine of each scene's displacement against that of its NEAREST-COMMAND neighbour.

    The bound on any command-only map: a function of the command must return nearly the same edit for
    two nearly-identical commands, so it can do no better than how much those scenes' true edits agree.
    Commands are standardized per column first, since ``dv`` and ``pos`` carry different units and an
    unscaled distance would let whichever has the larger numeric range choose the neighbour.
    """
    c = (cmds - cmds.mean(axis=0)) / (cmds.std(axis=0) + 1e-12)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(axis=1)
    out = []
    for i, j in enumerate(nn):
        a, b = mats[i], mats[j]
        out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return np.asarray(out)


def _loo_cos(mats: np.ndarray) -> np.ndarray:
    """Leave-one-out cosine of each row against the mean of the others."""
    n = len(mats)
    total = mats.sum(axis=0)
    out = []
    for i in range(n):
        other = (total - mats[i]) / max(n - 1, 1)
        d = mats[i]
        out.append(float(d @ other / (np.linalg.norm(d) * np.linalg.norm(other) + 1e-12)))
    return np.asarray(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--num_scenes", type=int, default=32)
    p.add_argument("--n_vel", type=int, default=4)
    p.add_argument("--n_spin", type=int, default=4)
    p.add_argument("--max_cached_shards", type=int, default=1)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    out = {}
    for L in layers:
        ds = LatentDataset(args.test_dir, layers=[L], max_cached_shards=args.max_cached_shards)
        scenes = vo.group_scenes(ds)
        sids = sorted(scenes)[: args.num_scenes]

        raw_v, raw_s, can_v, can_s, starts = [], [], [], [], []
        cmd_v, cmd_s = [], []
        for s in sids:
            cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
            if len(cells) != args.n_vel * args.n_spin:
                continue
            sq = so.commutation_square(cells, 0, 0, args.n_vel - 1, args.n_spin - 1)
            sam = {k: ds[i] for k, i in sq.items()}
            f = {k: vo.layer_flat(sam[k]["layers"][L]) for k in sam}
            d_v, d_s = f["vel_only"] - f["base"], f["spin_only"] - f["base"]
            grid = tuple(int(x) for x in sam["base"]["grid"])
            sh = vo.canon_shift(vo.clip_start_pos(sam["base"]), grid)
            starts.append(sh)
            raw_v.append(d_v); raw_s.append(d_s)
            can_v.append(vo.roll_layer(d_v, grid, sh))
            can_s.append(vo.roll_layer(d_s, grid, sh))
            # The COMMAND each scene actually issued at this grid corner, which is what the nearest-
            # neighbour ceiling matches on. Velocity: the commanded change dv plus the start position
            # the edit is relative to. Spin: the rate change and the marker's starting phase.
            cmd_v.append(np.concatenate([
                np.asarray(vo.clip_velocity(sam["vel_only"])) - np.asarray(vo.clip_velocity(sam["base"])),
                np.asarray(vo.clip_start_pos(sam["base"]))]))
            # phi0 enters as (cos, sin), NOT as a raw angle. It is circular, so a Euclidean distance on
            # the raw value calls phi0=0 and phi0=2pi maximally distant and picks the wrong neighbour --
            # which understates the spin ceiling by matching scenes whose marker phase is unrelated.
            phi = so.clip_phi0(sam["base"])
            cmd_s.append(np.array([so.clip_spin(sam["spin_only"]) - so.clip_spin(sam["base"]),
                                   float(np.cos(phi)), float(np.sin(phi))]))

        CV, CS = np.stack(cmd_v), np.stack(cmd_s)
        res = {
            "n_scenes": len(raw_v),
            "distinct_canon_shifts": len(set(starts)),
            # If these are ~0 the corner command is shared across scenes and the leave-one-out-mean
            # metric would have been valid after all; they are reported so that stays checkable.
            "cmd_spread_vel": [float(x) for x in CV.std(axis=0)],
            "cmd_spread_spin": [float(x) for x in CS.std(axis=0)],
            "ceiling_nn_vel": float(np.median(_nn_cos(np.stack(raw_v), CV))),
            "ceiling_nn_spin": float(np.median(_nn_cos(np.stack(raw_s), CS))),
            # The r-scale conversion: this is what compares against an operator's alignment.
            "ceiling_nn_sqrt_vel": float(np.sqrt(max(np.median(_nn_cos(np.stack(raw_v), CV)), 0.0))),
            "ceiling_nn_sqrt_spin": float(np.sqrt(max(np.median(_nn_cos(np.stack(raw_s), CS)), 0.0))),
            "ceiling_nn_canon_vel": float(np.median(_nn_cos(np.stack(can_v), CV))),
            "ceiling_nn_canon_spin": float(np.median(_nn_cos(np.stack(can_s), CS))),
            # NOT a ceiling -- the command-independent share of the edit. Kept for continuity with the
            # first run, renamed so it can never be quoted as a bound again.
            "shared_component_vel": float(np.median(_loo_cos(np.stack(raw_v)))),
            "shared_component_spin": float(np.median(_loo_cos(np.stack(raw_s)))),
        }
        out[f"L{L}"] = res
        print(f"[ceiling] L{L}: n={res['n_scenes']} "
              f"cmd_spread_vel={np.round(res['cmd_spread_vel'],3).tolist()} "
              f"cmd_spread_spin={np.round(res['cmd_spread_spin'],3).tolist()}", flush=True)
        print(f"           ceiling_nn_SQRT (compare to operator alignment): "
              f"vel {res['ceiling_nn_sqrt_vel']:.3f}  spin {res['ceiling_nn_sqrt_spin']:.3f}", flush=True)
        print(f"           ceiling_nn  vel {res['ceiling_nn_vel']:+.3f} "
              f"(canon {res['ceiling_nn_canon_vel']:+.3f}) | "
              f"spin {res['ceiling_nn_spin']:+.3f} (canon {res['ceiling_nn_canon_spin']:+.3f}) | "
              f"shared-cmpt vel {res['shared_component_vel']:+.3f}", flush=True)
        del ds

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("READ: ceiling_nn bounds ANY command-only map. Compare against the fitted operator alignment "
          "of 0.406: near it => the model class is the limit; far above => the fit has headroom.")


if __name__ == "__main__":
    main()
