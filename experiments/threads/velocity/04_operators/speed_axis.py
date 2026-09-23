#!/usr/bin/env python
"""Supervised SPEED axes for the velocity subspace (2026-09-15 magnitude fix, step 2).

Why. After the standardized refit ([[velocity-magnitude-not-steered]]) cmd-U8 tracks heading at ~6 deg but
magnitude only at corr ~0.47 in 2-D, and regressing the achieved speed on the command shows why:

    |v_ach| = 0.18 |v_b| + 0.64 |v_a|   (cmd_U8, gain 1)      -- the edit leaves the anchor's speed in place
    |v_ach| = 0.32 |v_b| + 0.55 |v_a|   (subspace_U8 = the TRUE dH projected onto U8)

Even the oracle delta loses its speed change when projected onto U8: the global PCA basis is fit on raw dH
over headings spanning 360 deg, so its top components carry heading; the speed change (a ~2x range) is
spread over the PCA tail. This script fits the speed-writing directions directly: a streaming least squares
of dH on z = [1, ds, ds*u_bx, ds*u_by] (ds = |v_b| - |v_a|, u_b the target heading), takes the three
coefficient rows as axes, reports how much of each already lies inside U8, orthogonalizes them against U8
and saves the augmented basis

    global_basis_spd_L{L}.npy = [U8 (8 rows); speed axes (3 rows)]     (11, D), orthonormal

which fit_command_operators.py --basis_tag spd and steer_velocity2d.py --basis_tag spd consume unchanged.

    python experiments/threads/velocity/04_operators/speed_axis.py \
        --train_dir .../train/vjepa2_large --layers 6,12,18,23 \
        --artifacts_dir outputs/analysis/moving_ball_v2d/subspace
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

AXIS_NAMES = ["ds", "ds*u_bx", "ds*u_by"]


def speed_features(va, vb):
    sa, sb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    ub = np.asarray(vb, float) / (sb + 1e-9)
    ds = sb - sa
    return np.array([1.0, ds, ds * ub[0], ds * ub[1]], float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True, help="dir with global_basis_L*.npy; output lands here")
    p.add_argument("--ku", type=int, default=8, help="rows of the PCA basis to keep in front of the speed axes")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--quantity", choices=["velocity", "accel"], default="velocity",
                   help="accel -> axes for |a| (clip_acceleration), written to the accel artifacts dir")
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir)
    U = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64)[: args.ku] for L in layers}
    qfn = vo.clip_acceleration if args.quantity == "accel" else vo.clip_velocity

    tr = LatentDataset(args.train_dir, layers=layers)
    scenes = vo.group_scenes(tr)
    if args.max_scenes:
        scenes = {s: scenes[s] for s in sorted(scenes)[: args.max_scenes]}
    P = 4
    ls = {L: None for L in layers}
    n = 0
    for s in sorted(scenes):
        ranks = sorted(scenes[s]); a = ranks[0]
        sa = tr[scenes[s][a]]; va = qfn(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[scenes[s][b]]; vb = qfn(sb)
            z = speed_features(va, vb).reshape(1, P)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                if ls[L] is None:
                    ls[L] = vo.LinearLS(P, dH.size, args.ridge)
                ls[L].add(z, dH.reshape(1, -1))
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[spd]   {n}/{len(scenes)} scenes", flush=True)

    summary = {"layers": layers, "ku": args.ku, "quantity": args.quantity, "n_train_scenes": len(scenes), "axes": AXIS_NAMES,
               "per_layer": {}}
    for L in layers:
        B = ls[L].solve(standardize=True)            # (4, D); rows 1..3 = speed axes
        axes = B[1:].astype(np.float64)
        rows = list(U[L])
        info = []
        for name, a in zip(AXIS_NAMES, axes):
            nrm = float(np.linalg.norm(a))
            inU = float(np.linalg.norm(U[L] @ a)) / (nrm + 1e-30)          # fraction of the axis inside U8
            # Gram-Schmidt against everything kept so far
            r = a.copy()
            for q in rows:
                r -= (q @ r) * q
            rn = float(np.linalg.norm(r))
            info.append({"axis": name, "norm": nrm, "frac_in_U": inU, "resid_frac": rn / (nrm + 1e-30)})
            if rn > 1e-6 * nrm:
                rows.append(r / rn)
        basis = np.stack(rows, 0).astype(np.float32)
        np.save(art / f"global_basis_spd_L{L}.npy", basis)
        summary["per_layer"][str(L)] = {"rows": int(basis.shape[0]), "axes": info}
        print(f"[spd] L{L}: " + "  ".join(f"{i['axis']}: |a|={i['norm']:.3g} inU8={i['frac_in_U']:.2f}" for i in info)
              + f"  -> basis {basis.shape}", flush=True)
        del ls[L]; gc.collect()
    (art / "speed_axis_meta.json").write_text(json.dumps(summary, indent=2))
    print(f"[spd] saved global_basis_spd_L*.npy + speed_axis_meta.json -> {art}")


if __name__ == "__main__":
    main()
