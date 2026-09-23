

#!/usr/bin/env python
"""Is decoder-in-the-loop TTO already doing SPLINE steering? (CPU-only, no decoder, no GPU.)

Hypothesis: "spline steering has the same spirit as our TTO -- we modify the latent space
gradually, just presented in a nicer way." That is testable with data already on disk. TTO
(``experiments/threads/acceleration/05_steering/steer_accel_decopt.py --mode free``) reaches 5.07deg by optimizing a full-D per-token edit
against the frozen decoder, and ``--dump_dir`` saved the winning edit e*(scene) for 500 TRAIN scenes.

This script asks, of those winning edits, three questions no GPU is needed to answer:

  1. HOW SPLINE-LIKE IS e* IN TIME?  Project e*'s T=8 temporal axis onto a clamped K-knot B-spline
     basis and report cos(smoothed, e*) for K=1..8. K=8 is the identity (cos == 1 by construction, and
     serves as the arithmetic check); K=1 is a constant-in-t edit. If cos is already ~0.95 at K=2-3,
     the edit TTO discovers is, to that accuracy, a low-order spline in time -- the hypothesis, in a
     number. If it stays low until K=7, TTO is exploiting temporal detail no spline can carry.

  2. HOW MUCH OF e* IS SPATIALLY UNIFORM?  The command-only spline operator broadcasts a (T,D) profile
     over all 16x16 spatial tokens (``spline_ops.broadcast_profile``), so it can only ever express the
     spatially-uniform part of e*. The energy fraction in that part is a hard ceiling on how much of
     TTO's solution that parametrization could ever reproduce.

  3. IS THE SPLINE REPRESENTATION PREDICTABLE FROM THE COMMAND?  Distilling the RAW e* into a
     feed-forward student failed badly (43.87 / 46.37 / 71.54deg, all near the 49deg no-op floor). If
     the low-K control points of the spatially-uniform part ARE predictable from phi(a_a,a_b) under
     cross-validation while the raw edit is not, the spline basis is the missing prior and amortizing
     TTO becomes worth GPU time. If they are equally unpredictable, e* is scene-idiosyncratic and no
     reparametrization rescues distillation -- equally worth knowing before spending anything.

Reports per layer; the layers are NOT interchangeable (``plan_d2/d2_support.json``: zeroing the
optimized edit at L23 costs 43.34deg vs 11.16deg at L6).

    python experiments/threads/acceleration/05_steering/analyze_tto_spline_content.py \
        --edits_dir .../moving_ball_accel2d_mixed/decopt_edits \
        --out .../moving_ball_accel2d_mixed/tto_spline_content.json
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

import numpy as np

from src.analysis import spline_ops as sp
from src.analysis import velocity_ops as vo


def cv_ridge_cos(X: np.ndarray, Y: np.ndarray, folds: int = 5, ridge: float = 1.0) -> float:
    """Mean held-out cos(predicted, true) of a ridge map X -> Y under contiguous k-fold CV.

    Contiguous folds (not shuffled) because the scene ids are already an arbitrary order and a fixed
    split makes the number reproducible without carrying an rng seed.
    """
    n = X.shape[0]
    edges = np.linspace(0, n, folds + 1).round().astype(int)
    cos = []
    for f in range(folds):
        te = np.zeros(n, dtype=bool)
        te[edges[f]: edges[f + 1]] = True
        tr = ~te
        A = X[tr].T @ X[tr] + ridge * np.eye(X.shape[1])
        W = np.linalg.solve(A, X[tr].T @ Y[tr])
        P = X[te] @ W
        for p, y in zip(P, Y[te]):
            cos.append(vo.cosine(p, y))
    return float(np.mean(cos))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--grid", default="8,16,16", help="T,H,W of the token grid")
    p.add_argument("--knots", default="1,2,3,4,5,6,7,8")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--max_scenes", type=int, default=0, help="0 = all; smoke-test knob")
    p.add_argument("--ridge", type=float, default=1.0)
    args = p.parse_args()

    T, H, W = (int(x) for x in args.grid.split(","))
    Ks = [int(x) for x in args.knots.split(",")]
    B = {K: sp.spline_basis(T, K, args.degree) for K in Ks}
    # least-squares smoother S_K = B (B^T B)^-1 B^T, applied along the temporal axis
    S = {K: B[K] @ np.linalg.pinv(B[K]) for K in Ks}

    files = sorted(Path(args.edits_dir).glob("scene*.npz"))
    if args.max_scenes:
        files = files[: args.max_scenes]
    if not files:
        raise SystemExit(f"no scene*.npz under {args.edits_dir}")

    z0 = np.load(files[0])
    layers = sorted(int(k[1:]) for k in z0.files if k.startswith("L") and k[1:].isdigit())
    D = int(z0[f"L{layers[0]}"].size // (T * H * W))
    print(f"[tto-spline] {len(files)} edits, layers={layers}, grid=({T},{H},{W}), D={D}", flush=True)

    full_cos = {L: {K: [] for K in Ks} for L in layers}   # spline-in-t, full spatial freedom
    prof_cos = {L: {K: [] for K in Ks} for L in layers}   # spline-in-t of the uniform part
    unif_frac = {L: [] for L in layers}                   # energy fraction that is spatially uniform
    phis, ctrl = [], {L: {K: [] for K in Ks} for L in layers}
    acc_errF = []

    for i, f in enumerate(files):
        z = np.load(f)
        aa, ab = z["a_a"].astype(np.float64), z["a_b"].astype(np.float64)
        phis.append(vo.command_features(aa, ab))
        acc_errF.append(float(z["acc_errF"]))
        for L in layers:
            E = z[f"L{L}"].astype(np.float32).reshape(T, H * W * D)
            tot = float(E.ravel() @ E.ravel()) + 1e-30
            # (1) spline-in-time with full spatial freedom
            for K in Ks:
                Es = (S[K].astype(np.float32) @ E)
                full_cos[L][K].append(float(E.ravel() @ Es.ravel()) /
                                      (np.sqrt(tot) * np.linalg.norm(Es) + 1e-30))
                del Es
            # (2) spatially-uniform part: the (T,D) profile, broadcast back
            R = E.reshape(T, H * W, D).mean(axis=1).astype(np.float64)      # (T,D)
            unif_frac[L].append(float(H * W * (R * R).sum()) / tot)
            rn = np.linalg.norm(R) + 1e-30
            for K in Ks:
                Rs = S[K] @ R
                prof_cos[L][K].append(float((R * Rs).sum()) / (rn * (np.linalg.norm(Rs) + 1e-30)))
                if K <= 4:
                    ctrl[L][K].append(sp.project_profile(R, B[K]).reshape(-1))
            del E, R
        if (i + 1) % 50 == 0:
            print(f"[tto-spline]   {i+1}/{len(files)}", flush=True)

    X = np.stack(phis, 0)
    pred = {L: {} for L in layers}
    for L in layers:
        for K in Ks:
            if K <= 4 and ctrl[L][K]:
                Y = np.stack(ctrl[L][K], 0)
                pred[L][str(K)] = round(cv_ridge_cos(X, Y, ridge=args.ridge), 4)

    summary = {
        "edits_dir": args.edits_dir, "n_edits": len(files), "grid": [T, H, W], "D": D,
        "layers": layers, "knots": Ks, "degree": args.degree, "ridge": args.ridge,
        "mean_acc_errF": round(float(np.mean(acc_errF)), 4),
        "spline_cos_full": {str(L): {str(K): round(float(np.mean(full_cos[L][K])), 4) for K in Ks}
                            for L in layers},
        "spline_cos_profile": {str(L): {str(K): round(float(np.mean(prof_cos[L][K])), 4) for K in Ks}
                               for L in layers},
        "uniform_energy_frac": {str(L): round(float(np.mean(unif_frac[L])), 4) for L in layers},
        "cv_pred_cos_control_points": {str(L): pred[L] for L in layers},
        "reads": {
            "spline_cos_full": "cos(e*, K-knot temporal smoothing of e*) -- how spline-like TTO's "
                               "winning edit already is in time. K=8 is the identity (== 1.0).",
            "spline_cos_profile": "same, on the spatially-uniform (T,D) part only.",
            "uniform_energy_frac": "hard ceiling on what a spatially-broadcast profile operator can "
                                   "reproduce of e*.",
            "cv_pred_cos_control_points": "5-fold held-out cos of ridge phi(13) -> control points. "
                                          "Raw-edit distillation scored 43.87-71.54 deg (floor 49).",
        },
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n[tto-spline] cos(e*, K-knot temporal smoothing)   [full spatial freedom]")
    print("   K  " + "  ".join(f"L{L:>2}" for L in layers))
    for K in Ks:
        print(f"  {K:>2}  " + "  ".join(f"{np.mean(full_cos[L][K]):.3f}" for L in layers))
    print("\n[tto-spline] spatially-uniform energy fraction of e*:")
    for L in layers:
        print(f"   L{L:<3} {np.mean(unif_frac[L]):.4f}")
    print("\n[tto-spline] 5-fold held-out cos, command -> control points:")
    for L in layers:
        print(f"   L{L:<3} " + "  ".join(f"K{K}:{v}" for K, v in pred[L].items()))
    print(f"[tto-spline] -> {out}")


if __name__ == "__main__":
    main()
