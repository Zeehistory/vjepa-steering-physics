

#!/usr/bin/env python
"""Re-solve the saved operators under a SCALED ridge, and measure the reachable-subspace ceiling.

**Why.** ``fit_spin_operators.py`` penalises every feature with the same lambda=1. The features are not
on the same scale, and not remotely: on ``operators_cond512`` the command block's ``XtX`` diagonal runs
0.0047 -> 1.2e4 while the conditioning block's sits at a median of 0.011, because ``z`` is a unit-norm
k-vector (so each component is ~1/sqrt(k)) and ``outer(z, dv)`` is smaller still. A single uniform
lambda is therefore negligible against the command features and dominant against the conditioning ones.
Measured as effective degrees of freedom, ``tr(XtX (XtX + lambda I)^-1)``:

    operators_canon    p=  27   edf =   7.7  (29% of p)
    operators_cond128  p= 411   edf =  98.1  (24%)
    operators_cond512  p=1563   edf = 162.7  (10%)

So the conditioned operator that took 8 hours to fit is using a tenth of the parameters it paid for,
and every doubling of ``k`` adds features that ridge then shrinks by ~99%. That is a plausible cause of
the apparent saturation of the alignment-vs-capacity curve (0.477 -> 0.660), which was read as a
property of the model class.

**It also reopens a question this project considered closed.** ``latent_crosstalk.py`` argues ridge was
ruled out because "shrinkage scales an edit down; it does not rotate it", supported by alignment ==
norm-ratio. That argument is valid for ISOTROPIC shrinkage. Here the per-feature shrinkage factors
``d_j/(d_j+lambda)`` span 0.01 to 0.9999, which is strongly anisotropic -- and anisotropic shrinkage
rotates. "Mis-aimed rather than merely small" is the predicted symptom of this bug, not evidence
against it.

**What is re-solved.** Ridge enters only at ``solve()``, and the fit persisted ``XtX``/``XtY``, so a
different penalty is an exact refit at zero fitting cost:

    uniform        A = XtX + lambda * I                 (what was fitted)
    standardized   A = XtX + lambda * diag(diag(XtX))   (equivalent to standardizing every feature)

**Why the sweep is cheap.** The scored quantities are three scalars per scene -- ``e.u_v``, ``e.u_s``,
``|e|`` -- so the predicted edit ``e = XtY^T w`` never has to be materialised in its 2.1M dims:

    e . u   = w . (XtY u)        one p-vector per scene per axis
    |e|^2   = w^T G w            with G = XtY XtY^T, a (p x p) matrix computed ONCE

After those two objects exist, the whole (scaling x lambda) grid is p-dimensional algebra and runs in
seconds, so the lambda axis can be swept densely instead of guessed.

**The ceiling this also buys.** ``G`` and ``a = XtY u_v`` give, for free, the best alignment ANY choice
of coefficients could reach in this feature class -- estimation error excluded:

    max over w of (w.a)/sqrt(w^T G w)  =  sqrt(a^T G^-1 a)  =  |proj of u_v onto rowspace(XtY)|

This separates the two possible diagnoses, which call for opposite work:

  * ceiling near 1.0 -> the reachable subspace CONTAINS the true displacement direction and the gap is
    pure estimation (regularisation, feature scaling, sample size). Fixable without new features.
  * ceiling near the achieved alignment -> the span itself is too poor, and no amount of retuning helps;
    the features have to change.

**Canonicalization is handled by rolling the TARGETS, not the predictions.** A canon operator predicts
in the ball-centred frame, and the consumer normally rolls the prediction back by ``-shift``. Rolling is
a permutation, hence orthogonal, so every cosine, gain and norm here is invariant to which side it is
applied on. Rolling each scene's ``D_vel``/``D_spin`` FORWARD once is identical arithmetic to rolling
each of many predictions back, at 1/(scenes x lambdas) the cost.

**Lambda is selected on one half of the test scenes and reported on the other.** Sweeping lambda and
quoting the best value on the same scenes would be selection on the test set; the split keeps the
reported number honest, and both halves are printed so the gap is visible.

    PYTHONPATH=. python experiments/threads/restitution-spin/07_eval/resolve_ridge_sweep.py \
        --operators_dir .../analysis/spin_ball3d/operators_cond512 \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --layers 18 --out .../analysis/spin_ball3d/ridge_sweep_cond512.json
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


def _gram(XtY: np.ndarray, chunk: int = 262144) -> np.ndarray:
    """``XtY XtY^T`` in float64, accumulated over column chunks.

    Chunked because ``XtY`` is float32 and (p x 2.1M); casting the whole thing to float64 at once would
    double an already 13 GB array for no benefit, while the (p x p) result is tiny. float64 for the
    accumulation because ``G`` is inverted downstream and its condition number is large.
    """
    p, N = XtY.shape
    G = np.zeros((p, p), dtype=np.float64)
    for i in range(0, N, chunk):
        C = XtY[:, i:i + chunk].astype(np.float64)
        G += C @ C.T
    return G


def main() -> None:
    p_ = argparse.ArgumentParser(description=__doc__)
    p_.add_argument("--operators_dir", required=True)
    p_.add_argument("--test_dir", required=True)
    p_.add_argument("--out", required=True)
    p_.add_argument("--layers", default="18")
    p_.add_argument("--num_scenes", type=int, default=48)
    p_.add_argument("--n_vel", type=int, default=4)
    p_.add_argument("--n_spin", type=int, default=4)
    p_.add_argument("--max_cached_shards", type=int, default=2)
    p_.add_argument("--lambdas", default="1e-4,1e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,100,1000")
    args = p_.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    if len(layers) != 1:
        # The G-trick is per-operator-matrix, and a multi-layer operator stores one XtY per layer with
        # its own row space. Supporting it means summing scores across layers, which is a different
        # (and untested) contract -- refuse rather than silently score only the first.
        raise SystemExit("this script scores ONE layer at a time (--layers 18)")
    L = layers[0]
    lams = [float(x) for x in args.lambdas.split(",") if x]

    ops_dir = Path(args.operators_dir)
    meta = json.loads((ops_dir / "operators_meta.json").read_text())
    canon = bool(meta.get("canon", False))
    cond_dim = int(meta.get("cond_dim", 0))
    print(f"[sweep] {ops_dir.name}: canon={canon} cond_dim={cond_dim} layer={L}", flush=True)

    # --- the two operators, as normal equations ------------------------------------------------------
    store = {}
    for kind in ("vel", "spin"):
        z = np.load(ops_dir / f"operator_{kind}.npz")
        XtX = z[f"XtX_{L}"].astype(np.float64)
        XtY = z[f"XtY_{L}"]                      # (p, N) float32, left as-is: 13 GB at cond512
        print(f"[sweep] {kind}: XtX {XtX.shape}, XtY {XtY.shape} -- building Gram", flush=True)
        G = _gram(XtY)
        store[kind] = {"XtX": XtX, "XtY": XtY, "G": G}
        if cond_dim and kind == "vel":
            store["P"] = z[f"P_{L}"].astype(np.float64)
            # Present only for a PCA basis; a random-projection operator has no centring vector and
            # must not be given one.
            store["mu"] = z[f"mu_{L}"].astype(np.float64) if f"mu_{L}" in z.files else None
        print(f"[sweep] {kind}: Gram done", flush=True)

    # --- test scenes: the ground-truth frame, in the operator's own (canonical) frame -----------------
    ds = LatentDataset(args.test_dir, layers=layers, max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]
    rows = []
    for n, s in enumerate(sids):
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        sq = so.commutation_square(cells, 0, 0, args.n_vel - 1, args.n_spin - 1)
        sam = {k: ds[i] for k, i in sq.items()}
        grid = tuple(int(x) for x in sam["base"]["grid"])
        sh = vo.canon_shift(vo.clip_start_pos(sam["base"]), grid)

        def flat(key):
            f = vo.layer_flat(sam[key]["layers"][L])
            # Roll the TARGETS into the canonical frame rather than rolling predictions out of it.
            # roll is a permutation, so all gains/cosines/norms below are unchanged.
            return vo.roll_layer(f, grid, sh) if canon else f

        base = flat("base")
        D_vel = flat("vel_only") - base
        D_spin = flat("spin_only") - base

        nv = np.linalg.norm(D_vel)
        u_v = D_vel / (nv + 1e-12)
        perp = D_spin - (D_spin @ u_v) * u_v
        n_perp = np.linalg.norm(perp)
        u_s = perp / (n_perp + 1e-12)

        va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
        wa, wb = so.clip_spin(sam["base"]), so.clip_spin(sam["spin_only"])
        fv = vo.command_features_pos(va, vb, vo.clip_start_pos(sam["base"]))
        fs = so.spin_command_features(wa, wb, so.clip_phi0(sam["base"]))
        if cond_dim:
            # Exactly the code the fit used: projection of the BASE clip's latent, in the same frame,
            # unit-normalized. A different z than the fit saw would produce confident nonsense.
            zc = (base - store["mu"] if store.get("mu") is not None else base) @ store["P"]
            zc = zc / (np.linalg.norm(zc) + 1e-12)
            fv = np.concatenate([fv, zc, np.outer(zc, np.asarray(vb) - np.asarray(va)).ravel()])
            fs = np.concatenate([fs, zc, zc * (float(wb) - float(wa))])

        # The per-scene projections onto the frame. These are the ONLY places the 2.1M-dim vectors are
        # touched; everything after this is p-dimensional.
        r = {"scene": int(s), "nv": float(nv), "n_perp": float(n_perp),
             "cos_Dvel_Dspin": float(D_vel @ D_spin / (nv * np.linalg.norm(D_spin) + 1e-12)),
             "fv": fv, "fs": fs}
        for kind in ("vel", "spin"):
            XtY = store[kind]["XtY"]
            r[f"a_v_{kind}"] = XtY @ u_v.astype(np.float32)
            r[f"a_s_{kind}"] = XtY @ u_s.astype(np.float32)
        rows.append(r)
        if (n + 1) % 8 == 0:
            print(f"[sweep] scene {n + 1}/{len(sids)}", flush=True)

    # --- reachable-subspace ceiling ------------------------------------------------------------------
    # sqrt(a^T G^-1 a) with a = XtY u_v is the length of u_v's projection onto rowspace(XtY): the best
    # alignment any coefficient vector could achieve, before any question of estimating it. G is
    # ill-conditioned (the feature scales that motivate this whole script), so a small jitter is added
    # relative to its own trace rather than absolutely.
    for kind in ("vel", "spin"):
        G = store[kind]["G"]
        store[kind]["Ginv"] = np.linalg.pinv(G + np.eye(G.shape[0]) * (np.trace(G) / G.shape[0]) * 1e-10)
    for r in rows:
        # Each operator is scored against the axis it is SUPPOSED to move: the velocity operator against
        # u_v, the spin operator against u_s. Using u_v for both would report the spin operator's
        # ceiling for producing a velocity change, which is not a quantity anyone wants.
        for kind, axis in (("vel", "a_v_vel"), ("spin", "a_s_spin")):
            a = np.asarray(r[axis], dtype=np.float64)
            r[f"ceiling_{kind}"] = float(np.sqrt(max(a @ store[kind]["Ginv"] @ a, 0.0)))

    # --- the sweep -----------------------------------------------------------------------------------
    half = len(rows) // 2
    sel, rep = rows[:half], rows[half:]          # lambda picked on `sel`, quoted on `rep`
    _eig_cache: dict = {}

    def score(kind: str, scaling: str, lam: float, subset, rank: int = 0) -> dict:
        XtX, G = store[kind]["XtX"], store[kind]["G"]
        D = np.diag(np.diag(XtX)).copy() if scaling == "standardized" else np.eye(XtX.shape[0])
        A = XtX + lam * D
        # REDUCED-RANK regression, and it costs nothing extra. Truncating B = A^-1 XtY to its top-r
        # output directions is B_r = U_r U_r^T B with U_r the leading eigenvectors of
        # B B^T = A^-1 G A^-1 -- a (p x p) matrix we already have. So the truncation acts entirely on
        # the p-dim coefficient vector (x -> U_r U_r^T x) and the 2.1M-dim operator is never formed.
        # Worth sweeping because the gap being chased here is variance, not bias: the reachable
        # subspace already contains the target direction, and dropping the output directions that are
        # mostly estimation noise is the standard way to buy that back.
        U = None
        if rank:
            # Cached per (kind, scaling, lambda): the eigenbasis does not depend on the rank or on
            # which scenes are being scored, and a p=1563 eigh is seconds -- recomputing it inside the
            # 176-cell grid would dominate the whole run.
            ck = (kind, scaling, lam)
            if ck not in _eig_cache:
                M = np.linalg.inv(A)
                evals, evecs = np.linalg.eigh(M @ G @ M)
                _eig_cache[ck] = evecs[:, np.argsort(evals)[::-1]]
            U = _eig_cache[ck][:, :rank]
        out = {"gain": [], "leak": [], "align": [], "norm": []}
        feat = "fv" if kind == "vel" else "fs"
        for r in subset:
            x = r[feat] if U is None else U @ (U.T @ r[feat])
            w = np.linalg.solve(A, x)
            e_uv = float(w @ r[f"a_v_{kind}"])
            e_us = float(w @ r[f"a_s_{kind}"])
            e_n = float(np.sqrt(max(w @ G @ w, 0.0)))
            # gain/leak follow latent_crosstalk.py exactly: fractions of the two REAL displacements.
            out["gain"].append(e_uv / (r["nv"] + 1e-12))
            out["leak"].append(e_us / (r["n_perp"] + 1e-12))
            out["align"].append(e_uv / (e_n + 1e-12))
            out["norm"].append(e_n / (r["nv"] + 1e-12))
        return {k: float(np.median(v)) for k, v in out.items()}

    p_dim = store["vel"]["XtX"].shape[0]
    ranks = [r for r in (0, 4, 8, 16, 32, 64, 128, 256) if r == 0 or r < p_dim]
    results = {}
    for scaling in ("uniform", "standardized"):
        for lam in lams:
            for rk in ranks:
                key = f"{scaling}@{lam:g}" + (f"/r{rk}" if rk else "")
                results[key] = {
                    "vel_sel": score("vel", scaling, lam, sel, rk),
                    "vel_rep": score("vel", scaling, lam, rep, rk),
                }

    best = max(results, key=lambda k: results[k]["vel_sel"]["align"])
    fitted = results["uniform@1"]
    summary = {
        "operators": ops_dir.name, "layer": L, "cond_dim": cond_dim, "canon": canon,
        "n_scenes": len(rows), "n_select": len(sel), "n_report": len(rep),
        "lambdas": lams,
        "as_fitted": {"key": "uniform@1", **fitted},
        "best_by_selection": {"key": best, **results[best]},
        "ceiling_vel_median": float(np.median([r["ceiling_vel"] for r in rows])),
        "ceiling_spin_median": float(np.median([r["ceiling_spin"] for r in rows])),
        "grid": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))

    print(f"\n# Ridge re-solve, {ops_dir.name} layer {L} "
          f"({len(sel)} select / {len(rep)} report scenes, {len(results)} grid cells)\n")
    print("| estimator | align (sel) | align (rep) | gain (rep) | leak (rep) | norm (rep) |")
    print("|---|---|---|---|---|---|")
    # The full grid is 176 cells; printing it all buries the finding. Show the as-fitted baseline, the
    # selection-chosen winner, and the next best few -- with the selection column alongside so a cell
    # that only looks good on the reporting half is visible as such.
    order = sorted(results, key=lambda k: -results[k]["vel_sel"]["align"])
    for k in ["uniform@1"] + [x for x in order[:8] if x != "uniform@1"]:
        a, b = results[k]["vel_sel"], results[k]["vel_rep"]
        mark = "  <- as fitted" if k == "uniform@1" else ("  <- best" if k == best else "")
        print(f"| {k} | {a['align']:.3f} | {b['align']:.3f} | "
              f"{b['gain']:.3f} | {b['leak']:+.3f} | {b['norm']:.3f} |{mark}")
    print(f"\nreachable-subspace ceiling (velocity): {summary['ceiling_vel_median']:.3f}")
    print("  = |projection of the true direction onto the span the operator can emit|;")
    print("  an upper bound over ALL coefficient choices, estimation error excluded.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
