

#!/usr/bin/env python
"""Fit COMMAND-ONLY acceleration operators whose edit is a SPLINE IN TIME (Step 2 "Bravo", spline arm).

Every command-only accel operator so far synthesizes a single global edit applied with equal weight at
every temporal token. Acceleration is a time-varying effect, and the measured edit's magnitude grows
monotonically along the clip (curvature_summary: ``dH_norm_per_t`` 449 -> 878 at L12), so a
constant-in-t edit has the wrong SHAPE. Here the edit is instead a smooth curve over the T=8 temporal
tokens, expanded in a clamped B-spline basis with K control points:

    Delta R(t) = sum_j C_j B_j(t),     C = W_K . phi(a_a, a_b)

``W_K`` is a ridge map from the 13-dim command features to the K*D spline control points, fit on TRAIN
by streaming normal equations (same math as fit_command_operators_accel.py), per layer, per K.

K IS THE ABLATION. The basis is a partition of unity, so:

    K = 1   constant in t  == the classical one-global-vector operator (the ~13-15deg plateau)
    K = 2   linear ramp in t
    K = 3,4 the low-order curved / piecewise-linear regime
    K = 8   unconstrained per-token profile == the existing ``cmd_prof`` operator

Same features, same ridge, same training pairs at every K — only the temporal degrees of freedom change,
so any difference is attributable to temporal smoothness alone and nothing else.

Training pairs: anchor = rank 0, targets = ranks 1..7, i.e. 7 pairs per scene (3500 on the 500-scene
train split) rather than the single extreme pair. This enrichment is applied identically at every K, and
K=8 is the internal control that isolates the basis from the extra data.

Writes ``spline_W_K{K}_L{L}.npy`` + ``spline_operator_meta.json`` into --output_dir, consumed by
``experiments/threads/acceleration/05_steering/steer_accel_spline.py``. Held-out TEST gate = cos(predicted profile, true profile); the decode
is the decisive test.

    python experiments/threads/acceleration/04_operators/fit_accel_spline_operator.py \
        --train_dir .../moving_ball_scene_accel2d_mixed/train/vjepa2_large \
        --test_dir  .../moving_ball_scene_accel2d_mixed/test/vjepa2_large \
        --layers 6,12,18,23 --knots 1,2,3,4,6,8 \
        --output_dir outputs/analysis/moving_ball_accel2d_mixed/spline
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


def _profiles(sample, layers, grid):
    """Per-layer spatially pooled temporal profile ``(T, D)`` of one clip."""
    return {L: sp.temporal_profile(vo.layer_flat(sample["layers"][L]), grid) for L in layers}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--knots", default="1,2,3,4,6,8",
                   help="control-point counts K to fit; 1 = constant-in-t (global-vector baseline), "
                        "T = unconstrained per-token profile (cmd_prof equivalent)")
    p.add_argument("--degree", type=int, default=3, help="B-spline degree (auto-lowered when K is small)")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0, help="0 = all; smoke-test knob")
    p.add_argument("--quantity", choices=["accel", "angvel"], default="accel")
    args = p.parse_args()

    qfn = vo.clip_angvel if args.quantity == "angvel" else vo.clip_acceleration
    layers = [int(x) for x in args.layers.split(",")]
    Ks = [int(x) for x in args.knots.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    P = vo.COMMAND_FEATURE_DIM

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}

    grid = tuple(int(x) for x in tr[tr_scenes[sorted(tr_scenes)[0]][0]]["grid"])
    T, D = grid[0], int(tr.records[0]["hidden_dim"])
    B = {K: sp.spline_basis(T, K, args.degree) for K in Ks}
    print(f"[spline-fit] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; "
          f"grid={grid} T={T} D={D}; K={Ks}", flush=True)

    ls = {K: {L: vo.LinearLS(P, K * D, args.ridge) for L in layers} for K in Ks}

    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s])
        sa = tr[tr_scenes[s][ranks[0]]]
        aa = qfn(sa)
        Ra = _profiles(sa, layers, grid)
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            phi = vo.command_features(aa, qfn(sb)).reshape(1, P)
            for L in layers:
                dR = sp.temporal_profile(vo.layer_flat(sb["layers"][L]), grid) - Ra[L]
                for K in Ks:
                    C = sp.project_profile(dR, B[K])            # (K, D)
                    ls[K][L].add(phi, C.reshape(1, -1))
            del sb
        del Ra, sa
        gc.collect()
        n += 1
        if n % 50 == 0:
            print(f"[spline-fit]   {n}/{len(tr_scenes)} scenes", flush=True)

    W = {K: {L: ls[K][L].solve() for L in layers} for K in Ks}   # (P, K*D)
    del ls; gc.collect()
    for K in Ks:
        for L in layers:
            np.save(out / f"spline_W_K{K}_L{L}.npy", W[K][L].astype(np.float32))
    print("[spline-fit] operators saved; running held-out latent gate", flush=True)

    # ------------------------------------------------------------------ held-out latent gate
    # Two numbers per (K, layer):
    #   pred_cos  cos(predicted profile, TRUE profile)  -- can the command synthesize the edit?
    #   proj_cos  cos(K-knot projection of the TRUE profile, TRUE profile) -- is K enough to EXPRESS it?
    # proj_cos is the representational ceiling for that K; pred_cos is what the operator achieves.
    gate = {K: {L: {"pred_cos": [], "proj_cos": []} for L in layers} for K in Ks}
    m = 0
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s])
        sa = te[te_scenes[s][ranks[0]]]
        aa = qfn(sa)
        Ra = _profiles(sa, layers, grid)
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]
            phi = vo.command_features(aa, qfn(sb))
            for L in layers:
                dR = sp.temporal_profile(vo.layer_flat(sb["layers"][L]), grid) - Ra[L]
                flat_true = dR.reshape(-1)
                for K in Ks:
                    pred = (phi @ W[K][L]).reshape(K, D)
                    gate[K][L]["pred_cos"].append(
                        vo.cosine(sp.reconstruct_profile(pred, B[K]).reshape(-1), flat_true))
                    gate[K][L]["proj_cos"].append(
                        vo.cosine(sp.smooth_profile(dR, B[K]).reshape(-1), flat_true))
            del sb
        del Ra, sa
        gc.collect()
        m += 1
        if m % 25 == 0:
            print(f"[spline-fit]   gated {m}/{len(te_scenes)} scenes", flush=True)

    summary = {
        "train_dir": args.train_dir, "test_dir": args.test_dir, "quantity": args.quantity,
        "layers": layers, "knots": Ks, "degree": args.degree, "ridge": args.ridge,
        "grid": list(grid), "T": T, "D": D, "command_feature_dim": P,
        "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes),
        "pairs_per_scene": 7,
        "per_knot": {
            str(K): {str(L): {
                "pred_cos": round(float(np.mean(gate[K][L]["pred_cos"])), 4),
                "proj_cos_ceiling": round(float(np.mean(gate[K][L]["proj_cos"])), 4),
            } for L in layers} for K in Ks},
        "artifacts": "spline_W_K{K}_L{L}.npy  (P=13 -> K*D control points)",
        "note": "K=1 is constant-in-t (the classical global-vector operator); K=T is the unconstrained "
                "per-token profile. Decode (experiments/threads/acceleration/05_steering/steer_accel_spline.py) is the decisive test.",
    }
    (out / "spline_operator_meta.json").write_text(json.dumps(summary, indent=2))

    print("\n[spline-fit] held-out latent gate (mean cosine vs true profile):")
    print(f"  {'K':>3}  " + "  ".join(f"L{L}:pred/ceil" for L in layers))
    for K in Ks:
        cells = "  ".join(f"{np.mean(gate[K][L]['pred_cos']):.3f}/"
                          f"{np.mean(gate[K][L]['proj_cos']):.3f}" for L in layers)
        print(f"  {K:>3}  {cells}")
    print(f"[spline-fit] -> {out}/spline_operator_meta.json")


if __name__ == "__main__":
    main()
