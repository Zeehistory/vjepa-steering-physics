"""Vector-valued outcome steering: command a ball VELOCITY (vx, vy), not just a speed.

Everything certified so far steers ONE scalar -- the outgoing speed along the table axis -- through a
one-parameter law, ``v_out - v_in = alpha (v_p - v_in)``. That is a real result but it is a weak
claim: a scalar map is trivially invertible and a reviewer can fairly ask whether the controller has
learned anything more than a gain. This script asks whether the same machinery steers a 2-D outcome.

THE SECOND ACTUATION AXIS IS ALREADY IN THE RIG
----------------------------------------------
No new geometry. ``simulate`` already takes ``y_b0``, the ball's lateral offset from the blade's
centreline -- it exists as a hardware-realism probe (an aiming ERROR to be budgeted). Read the other
way it is a control input: where on the striker you make contact.

That reading is only physical on a CONVEX striker. On the abstract paddle the striking face is a
flat box, the contact normal is along x for every offset, and an off-centre hit should give
essentially zero lateral velocity -- i.e. the paddle is expected to have NO second axis. On the
dynamically-actuated arm the striker is a 50 mm sphere, so an off-centre hit tilts the contact
normal and genuinely throws the ball sideways. Running both is the point: the paddle is the negative
control that shows the lateral response is contact geometry rather than a fitting artefact. If the
paddle also shows a large lateral gain, something is wrong with the measurement, not with physics.

WHAT IS FIT, AND WHAT WOULD FALSIFY IT
--------------------------------------
Sweep a grid over the action pair ``a = (v_p, y_b0)`` and record the outcome pair
``u = (v_out_x, v_out_y)``. Then:

1. Fit a FAMILY of forward maps (see ``MODELS``): linear ``u = A a + b``, plus ``y_b0^2``,
   ``v_p*y_b0`` and both. Score every one OUT OF SAMPLE on a held-out half of the grid -- an extra
   term always lowers the in-sample residual, so in-sample numbers cannot adjudicate this. This is
   the honest test of "is it still linear": the scalar law is, but a lateral throw off a sphere is a
   geometric sine scaled by impact speed, and there is no reason for that to stay linear.
2. Invert the linear map and the best held-out map, on the SAME held-out targets drawn inside the
   reachable set, and EXECUTE each one in MuJoCo. Score pass@5% on the VECTOR error
   ``||u - u*|| / ||u*||`` -- not per-axis, because per-axis lets a controller that only ever gets
   the big component right look good. Running both inverses against identical requests is what turns
   "the residual is lower" into "the controller is actually better".
3. Report the conditioning of ``A``. A near-singular ``A`` means the second axis is nominally
   present but useless: tiny outcome changes need enormous action changes, and the "2-D" claim is
   cosmetic. ``cond(A)`` and the achievable lateral fraction are the numbers that decide whether
   this strengthens the claim or refutes it. Either result is publishable; a quiet failure is not.

SIGN CONVENTION -- NOW MEASURED, PREVIOUSLY FABRICATED
------------------------------------------------------
``sim["v_out_lateral"]`` is a MAGNITUDE -- ``max |hypot(v_y, v_z)|`` over the post window -- so it
discards direction AND mixes in the vertical axis. An earlier version of this script fell back to
``sign(y_b0) * magnitude`` when no signed key existed, and NO SIGNED KEY EXISTED: the fallback was
the only path ever taken. That is fatal, and the two guards meant to catch it were both vacuous
under it, which is worse than having no guards at all:

* ``sign(y)*|.|`` is monotone in ``y`` BY CONSTRUCTION, so the monotonicity check always passed;
* at ``y_b0 = 0`` it is exactly 0 because ``sign(0) = 0``, so the zero-crossing check always passed.

The JSON would have looked clean while the entire second outcome axis was manufactured out of a
folded absolute value. The rig now records the signed y component directly
(``v_out_lateral_signed``, mean over the post window, in :mod:`src.data.paddle_strike`), and this
script HARD-FAILS if the fallback is ever re-entered rather than reporting a guarded number. The
monotonicity and zero-crossing checks are kept, because against genuinely signed data they finally
test something.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/05_steering/vector_steer.py \
        --output_dir /path/out --embodiment franka_dynamic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.paddle_strike import build_striker

LATERAL_SIGNED_KEYS = ("v_out_lateral_signed", "v_out_y_world", "v_out_lat_world")


def _outcome(sim: dict, y_b0: float) -> tuple[float, float, bool]:
    """(v_out_x, v_out_y, used_fallback_sign) for one rollout."""
    vx = float(sim["v_out_world"])
    for k in LATERAL_SIGNED_KEYS:
        if k in sim:
            return vx, float(sim[k]), False
    mag = float(sim.get("v_out_lateral", 0.0))
    return vx, float(np.sign(y_b0) * mag), True


# Model family over the action ``a = (v_p, y_b0)``. Every model carries the two linear terms and an
# intercept; the named extras are what distinguishes them. Column order is fixed and the inverse
# below depends on it: [v_p, y, (y^2), (v_p*y), 1].
#
# "bilinear" exists because of the physics, not to pad the comparison. An off-centre hit on a sphere
# throws the ball sideways by roughly v_p * sin(theta) with sin(theta) ~ y/R -- the lateral outcome
# is a PRODUCT of impact speed and offset. A model that is quadratic in y alone cannot express that
# at any order, which is exactly what the first smoke test showed: y^2 cut the v_out_x residual by
# 2.8x and left v_out_y untouched at ~9% of range.
MODELS = {
    "linear":    (),
    "quadratic": ("y2",),
    "bilinear":  ("vpy",),
    "full":      ("y2", "vpy"),
}


def _design(a: np.ndarray, terms: tuple = ()) -> np.ndarray:
    """Design matrix for action rows ``a = (v_p, y_b0)`` with the named extra terms."""
    cols = [a]
    if "y2" in terms:
        cols.append(a[:, 1:2] ** 2)
    if "vpy" in terms:
        cols.append(a[:, 0:1] * a[:, 1:2])
    return np.concatenate(cols + [np.ones((len(a), 1))], axis=1)


def _fit(A_in: np.ndarray, U: np.ndarray, terms: tuple = ()):
    """Least-squares outcome map with intercept. Returns (coef, predict_fn, resid_frac)."""
    W, *_ = np.linalg.lstsq(_design(A_in, terms), U, rcond=None)
    pred = _design(A_in, terms) @ W
    rng = U.max(axis=0) - U.min(axis=0)
    resid = np.abs(pred - U).max(axis=0) / np.maximum(rng, 1e-12)
    return W, (lambda a: _design(a, terms) @ W), resid


def _heldout_resid(A_in: np.ndarray, U: np.ndarray, terms: tuple, train: np.ndarray):
    """Residual on points the fit never saw, as a fraction of each outcome's full range.

    In-sample residual cannot adjudicate between these models: an extra term always fits better on
    the data it was fit to, so a lower in-sample residual is not evidence of curvature. This is the
    comparison that decides whether the second axis is genuinely nonlinear.
    """
    test = ~train
    W, *_ = np.linalg.lstsq(_design(A_in[train], terms), U[train], rcond=None)
    pred = _design(A_in[test], terms) @ W
    rng = U.max(axis=0) - U.min(axis=0)
    return np.abs(pred - U[test]).max(axis=0) / np.maximum(rng, 1e-12)


def _invert_linear(A_mat: np.ndarray, b_vec: np.ndarray, u_star: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(A_mat) @ (u_star - b_vec)


def _coeffs(W: np.ndarray, terms: tuple) -> tuple:
    """Unpack a fitted W into (c_vp, c_y, c_yy, c_vpy, c_1) per outcome, zero-filling absent terms."""
    idx, n = {"vp": 0, "y": 1}, 2
    for t in ("y2", "vpy"):
        if t in terms:
            idx[t] = n
            n += 1
    idx["1"] = n
    z = np.zeros(W.shape[1])
    g = lambda k: W[idx[k]] if k in idx else z
    return g("vp"), g("y"), g("y2"), g("vpy"), g("1")


def _invert(W: np.ndarray, terms: tuple, u_star: np.ndarray, y_max: float,
            a_lin: np.ndarray) -> np.ndarray:
    """Inverse of the fitted forward map: solve ``u(v_p, y) = u_star`` for the action.

    Eliminating ``v_p`` between the two outcome equations leaves a single polynomial in ``y``:

        v_p * (A_vp + A_vpy y) = u_x - A_y y - A_yy y^2 - A_1   =: Px(y)
        (B_vp + B_vpy y) Px(y) + (A_vp + A_vpy y)(B_y y + B_yy y^2 + B_1 - u_y) = 0

    which is cubic once a ``v_p*y`` term is present and quadratic without it. Rooted exactly with
    ``np.roots`` rather than by Newton, so there is no iteration to converge or fail silently, and
    the degenerate low-order cases fall out of trimming leading zeros. Among real roots the one
    nearest the linear solution is taken: a far-side contact can throw the ball the same way, and
    that is not the branch the linear controller is being compared against.
    """
    A_vp, A_y, A_yy, A_vpy, A_1 = (c[0] for c in _coeffs(W, terms))
    B_vp, B_y, B_yy, B_vpy, B_1 = (c[1] for c in _coeffs(W, terms))

    # Px(y) = -A_yy y^2 - A_y y + (u_x - A_1), highest power first
    px = np.array([-A_yy, -A_y, u_star[0] - A_1])
    poly = (np.polyadd(np.polymul([B_vpy, B_vp], px),
                       np.polymul([A_vpy, A_vp], [B_yy, B_y, B_1 - u_star[1]])))
    poly = np.trim_zeros(np.atleast_1d(poly), "f")
    if len(poly) < 2:                                  # no solvable equation in y
        return a_lin
    roots = np.roots(poly)
    roots = roots[np.abs(roots.imag) < 1e-9].real
    if not len(roots):                                 # target off the reachable branch
        return a_lin

    inside = roots[np.abs(roots) <= y_max]
    cand = inside if len(inside) else roots
    y = float(cand[np.abs(cand - a_lin[1]).argmin()])
    denom = A_vp + A_vpy * y
    if abs(denom) < 1e-12:                             # v_p cannot move u_x at this offset
        return a_lin
    a = np.array([float((u_star[0] - A_y * y - A_yy * y * y - A_1) / denom), y])

    # Elimination is exact but ill-posed when an outcome axis is degenerate: the paddle's lateral
    # gain is ~1e-14, so the eliminated polynomial is numerically all-noise and its roots are
    # garbage, while the pseudo-inverse quietly returns the sensible minimum-norm answer. Rather
    # than special-casing that, check the answer: whichever action the FITTED model says lands
    # closer to the target is the one returned. Observed catching exactly this -- paddle pass@5%
    # 100% -> 0% with median error 75% before the check.
    pred = _design(np.stack([a, a_lin]), terms) @ W
    return a if np.linalg.norm(pred[0] - u_star) <= np.linalg.norm(pred[1] - u_star) else a_lin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="franka_dynamic")
    ap.add_argument("--v_in", type=float, default=1.0)
    ap.add_argument("--n_vp", type=int, default=9)
    ap.add_argument("--n_y", type=int, default=9)
    ap.add_argument("--vp_lo", type=float, default=-1.45)
    ap.add_argument("--vp_hi", type=float, default=0.25)
    ap.add_argument("--y_max", type=float, default=0.030,
                    help="half-width of the lateral offset sweep, metres. Kept well inside the "
                         "striker's radius so every grid point is still a real face-on contact "
                         "rather than a grazing edge hit that MuJoCo resolves differently.")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = build_striker(args.embodiment, image_size=128)

    # -- 1. sweep the action grid (render=False: no EGL, no GPU needed) -------------------------
    vps = np.linspace(args.vp_lo, args.vp_hi, args.n_vp)
    ys = np.linspace(-args.y_max, args.y_max, args.n_y)
    A_rows, U_rows, fallback = [], [], False
    for vp in vps:
        for y in ys:
            s = gen.simulate(args.v_in, float(vp), render=False, y_b0=float(y))
            vx, vy, fb = _outcome(s, float(y))
            fallback = fallback or fb
            A_rows.append([vp, y])
            U_rows.append([vx, vy])
    A_in = np.array(A_rows)
    U = np.array(U_rows)
    print(f"swept {len(A_in)} (v_p, y_b0) points; "
          f"v_out_x in [{U[:,0].min():.3f},{U[:,0].max():.3f}], "
          f"v_out_y in [{U[:,1].min():.3f},{U[:,1].max():.3f}]", flush=True)

    # -- 2. the sign convention: hard gate, not a reported flag ---------------------------------
    # A fabricated direction must not be allowed to reach the JSON looking like a measurement, so
    # this exits rather than annotating. See the module docstring for why the guards below cannot
    # catch the fallback on their own.
    if fallback:
        raise SystemExit(
            "FATAL: no signed lateral key in the sim output -- fell back to sign(y_b0)*|lateral|, "
            f"which fabricates the second outcome axis. Looked for {LATERAL_SIGNED_KEYS}. "
            "src.data.paddle_strike must record 'v_out_lateral_signed'; refusing to fit.")

    # Against genuinely signed data these finally test something: the lateral response must reverse
    # with the offset and vanish on the centreline.
    vp_ref = A_in[np.abs(A_in[:, 0] - np.median(vps)).argmin(), 0]
    sel = np.isclose(A_in[:, 0], vp_ref)
    y_sel, vy_sel = A_in[sel, 1], U[sel, 1]
    d = np.diff(vy_sel[np.argsort(y_sel)])
    monotone = bool(np.all(d >= -1e-6) or np.all(d <= 1e-6))
    zero_cross = float(np.abs(vy_sel[np.abs(y_sel).argmin()]))
    print(f"sign check: monotone={monotone}  |v_y| at y_b0=0 = {zero_cross:.5f} m/s", flush=True)

    # -- 3. the model family, scored OUT OF SAMPLE ----------------------------------------------
    # Alternate grid points into train/test so both halves span the full (v_p, y_b0) box -- a
    # contiguous split would ask the fit to extrapolate and flatter the richer models for the wrong
    # reason.
    train = (np.arange(len(A_in)) % 2 == 0)
    fits, ho = {}, {}
    for name, terms in MODELS.items():
        W, _f, resid = _fit(A_in, U, terms)
        fits[name] = W
        ho[name] = _heldout_resid(A_in, U, terms, train)
        print(f"{name:>9}: in-sample {np.array2string(resid, precision=4)}   "
              f"held-out {np.array2string(ho[name], precision=4)}", flush=True)

    A_mat = fits["linear"][:2, :].T           # d(outcome)/d(action)
    b_vec = fits["linear"][-1, :]
    cond = float(np.linalg.cond(A_mat))
    print(f"A = {A_mat.tolist()}  cond(A) = {cond:.2f}", flush=True)

    # The winner on held-out error, summed over both outcome axes -- but a richer model has to EARN
    # the swap by a clear margin, not win a tie on noise. Without this the paddle picks a nonlinear
    # model on the strength of differences in the 4th decimal of an axis whose entire range is
    # ~1e-15 m/s, then inverts it and destroys a controller that was at 100%: the negative control's
    # second axis is degenerate, so any ranking over it is meaningless and linear must be the
    # default. Observed doing exactly this before the guard: paddle pass@5% 100% -> 40%.
    cheapest = min(MODELS, key=lambda m: ho[m].sum())
    best = cheapest if ho[cheapest].sum() < 0.9 * ho["linear"].sum() else "linear"
    print(f"best held-out model: {best}"
          f"{'' if best == cheapest else f' (ranked {cheapest}, but margin too small to swap)'}",
          flush=True)

    # -- 4. closed loop on held-out 2-D targets, EXECUTED, both inverses ------------------------
    # Targets are drawn inside the convex hull of the swept outcomes (shrunk 20%) so a failure is a
    # control failure and not a request outside the reachable set. The SAME targets go through both
    # controllers: whether the best nonlinear inverse beats the linear one on identical requests is
    # closed-loop nonlinearity result, which a residual comparison alone cannot deliver.
    rng = np.random.default_rng(args.seed)
    c = U.mean(axis=0)
    lo, hi = U.min(axis=0), U.max(axis=0)
    targets = c + 0.80 * (rng.uniform(lo, hi, size=(args.n_test, 2)) - c)

    def run_loop(invert, label):
        rs = []
        for u_star in targets:
            a_lin = _invert_linear(A_mat, b_vec, u_star)
            a = invert(u_star, a_lin)
            a = np.array([np.clip(a[0], args.vp_lo, args.vp_hi),
                          np.clip(a[1], -args.y_max, args.y_max)])
            s = gen.simulate(args.v_in, float(a[0]), render=False, y_b0=float(a[1]))
            vx, vy, _ = _outcome(s, float(a[1]))
            u = np.array([vx, vy])
            err = float(np.linalg.norm(u - u_star) / max(np.linalg.norm(u_star), 1e-9))
            rs.append({"target": u_star.tolist(), "action": a.tolist(),
                       "achieved": u.tolist(), "vector_rel_err": err,
                       "err_x": float(abs(vx - u_star[0]) / max(abs(u_star[0]), 1e-9)),
                       "err_y": float(abs(vy - u_star[1]) / max(abs(u_star[1]), 1e-9))})
        e = np.array([r["vector_rel_err"] for r in rs])
        p5 = float((e <= 0.05).mean())
        print(f"{label:>9} closed loop: pass@5% = {p5:.1%}  median {np.median(e):.2%}  "
              f"worst {e.max():.2%}  (n={len(e)})", flush=True)
        return rs, e, p5

    print("", flush=True)
    recs, errs, pass5 = run_loop(lambda u, a_lin: a_lin, "LINEAR")
    if best == "linear":
        # No nonlinear model earned the swap, so there is no second controller to run. Re-running
        # the linear map through the general inverse would burn 40 rollouts to reproduce the row
        # above -- and, on a degenerate axis, would not even reproduce it.
        print("no nonlinear model beat linear out of sample -- second loop skipped", flush=True)
        recs_q, errs_q, pass5_q = recs, errs, pass5
    else:
        recs_q, errs_q, pass5_q = run_loop(
            lambda u, a_lin: _invert(fits[best], MODELS[best], u, args.y_max, a_lin), best.upper())

    lat_frac = float(np.abs(U[:, 1]).max() / max(np.abs(U[:, 0]).max(), 1e-9))
    result = {
        "embodiment": args.embodiment, "v_in": args.v_in,
        "grid": {"n_vp": args.n_vp, "n_y": args.n_y, "y_max": args.y_max},
        "sign_convention": {"fallback_sign_from_y_b0": fallback,
                            "monotone_in_y": monotone,
                            "abs_vy_at_y0": zero_cross},
        "forward_linear": {"A": A_mat.tolist(), "b": b_vec.tolist(), "cond_A": cond},
        "models": {m: {"terms": list(MODELS[m]), "W": fits[m].tolist(),
                       "heldout_resid_frac_of_range": ho[m].tolist()} for m in MODELS},
        # The nonlinearity verdict, in one place. Out-of-sample residual is the honest comparison;
        # the closed-loop pass@5% delta is the one that survives a reviewer asking "so what".
        "nonlinearity": {
            "best_heldout_model": best,
            "heldout_resid_improvement_frac": (ho["linear"] - ho[best]).tolist(),
            "closed_loop_pass_delta": pass5_q - pass5,
            "beats_linear_heldout": bool(np.any(ho[best] < 0.9 * ho["linear"])),
        },
        "reachable": {"v_out_x_range": [float(U[:, 0].min()), float(U[:, 0].max())],
                      "v_out_y_range": [float(U[:, 1].min()), float(U[:, 1].max())],
                      "max_lateral_fraction": lat_frac},
        "closed_loop": {"n": len(errs), "pass_at_5pct": pass5,
                        "median_vector_rel_err": float(np.median(errs)),
                        "worst_vector_rel_err": float(errs.max()),
                        "records": recs},
        "closed_loop_nonlinear": {"n": len(errs_q), "pass_at_5pct": pass5_q,
                                  "median_vector_rel_err": float(np.median(errs_q)),
                                  "worst_vector_rel_err": float(errs_q.max()),
                                  "records": recs_q},
    }
    (out / "vector_steer.json").write_text(json.dumps(result, indent=2))

    # -- 5. figure: reachable set + where the closed loop landed --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax.scatter(U[:, 0], U[:, 1], s=12, c="#c9d6e4", label="swept outcomes")
    for r in recs:
        t, a_ = r["target"], r["achieved"]
        ax.plot([t[0], a_[0]], [t[1], a_[1]], "-", lw=0.7, color="#999", zorder=2)
    ax.scatter([r["target"][0] for r in recs], [r["target"][1] for r in recs],
               s=22, marker="x", c="#c43e2e", label="requested", zorder=3)
    ax.scatter([r["achieved"][0] for r in recs], [r["achieved"][1] for r in recs],
               s=18, c="#2f6fb0", label="achieved", zorder=4)
    ax.set_xlabel("$v_{out,x}$ (m/s)")
    ax.set_ylabel("$v_{out,y}$ (m/s)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    bins = np.linspace(0, max(errs.max(), errs_q.max()) * 100, 21)
    ax2.hist(errs * 100, bins=bins, color="#2f6fb0", alpha=0.75, label="linear inverse")
    ax2.hist(errs_q * 100, bins=bins, histtype="step", lw=1.4, color="#1b7f5a",
             label=f"{best} inverse")
    ax2.axvline(5.0, color="#c43e2e", lw=1.2, ls=":")
    ax2.set_xlabel("vector relative error (%)")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, lw=0.5, axis="y")
    fig.suptitle(f"{args.embodiment}: 2-D velocity steering  "
                 f"(pass@5% linear {pass5:.0%} / {best} {pass5_q:.0%}, cond(A) = {cond:.1f})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "vector_steer.png", dpi=170)
    plt.close(fig)
    print(f"wrote {out}/vector_steer.json, vector_steer.png")


if __name__ == "__main__":
    main()
