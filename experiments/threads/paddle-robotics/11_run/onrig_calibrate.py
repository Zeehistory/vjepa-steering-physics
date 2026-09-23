"""Calibrate the strike law on a PHYSICAL rig, and accept or reject it against the 5% bar.

This is the missing step between "certified in simulation" and "validated system". The certificate
(``experiments/pipeline/01_data/validate_paddle_strike.py``) proves the action->outcome map is invertible for the
SIMULATOR's contact parameters. Those parameters were chosen, not measured, so on a real paddle the
law has different coefficients and every commanded speed inherits that error. ``hardware_specs.py``
quantifies what each imperfection COSTS; this script is what an operator actually runs to fix the one
imperfection that can be fixed, by measuring the rig instead of guessing it.

WHAT PRECISION THE CALIBRATION NEEDS -- the number this whole script exists to check
-----------------------------------------------------------------------------------
The law has one free parameter, not three. A ball and striker moving at a common velocity ``u`` cannot
collide, so any correct law must return ``v_out = u``, which forces ``alpha + beta = 1`` and
``gamma = 0`` (see ``StrikeInverse.fit_constrained``; the free 3-parameter fit lands at
``alpha + beta = 1.00098``, so the constraint is not an assumption being smuggled in -- it is
confirmed). The law is therefore::

    v_out - v_in = alpha * (v_p - v_in)

Command a target with an estimate ``alpha_hat``, and the executed action is
``v_p = v_in + (v_target - v_in) / alpha_hat``. The rig responds with its TRUE alpha, so::

    v_out = v_in + (alpha_true / alpha_hat) * (v_target - v_in)

which gives a relative outcome error that is exact and rig-independent::

    |v_out - v_target| / |v_target| = |alpha_true/alpha_hat - 1| * |v_target - v_in| / |v_target|

For the head-on convention here (``v_in = +1``, ``v_target = -r*v_in``) the geometric factor is
``(r+1)/r``, which is WORST at the slowest ratio. So holding a tolerance ``TOL`` across a ratio range
requires::

    |d_alpha / alpha|  <=  TOL * r_min / (r_min + 1)

At ``TOL = 5%`` over ratios 0.5..3.0 that is **1.67% relative precision on alpha** (set by r=0.5;
r=3.0 alone would only need 3.75%). That is the acceptance criterion below, and note which way it
cuts: the slow end of the range is the expensive end to calibrate, which is the same place the sim
already fails (0.75x sits at 0% pass), so the slow end is where a real rig will hurt first.

USAGE
-----
Against real measurements -- a CSV with columns ``v_in,v_p,v_out`` in m/s, one row per strike, signs
in world convention (ball arriving is +, returning is -)::

    PYTHONPATH=. python experiments/threads/paddle-robotics/11_run/onrig_calibrate.py \\
        --measurements rig_strikes.csv --output_dir outputs/onrig

With no rig available, ``--selftest`` synthesises measurements from MuJoCo, including a mode where the
simulated "rig" is deliberately given DIFFERENT contact physics from the nominal model. That is the
end-to-end check of the procedure itself: it must recover the rig's own alpha, and it must REJECT the
nominal sim model when the rig disagrees with it. A calibration script that cannot fail that test
would pass any rig, including a broken one::

    PYTHONPATH=. python experiments/threads/paddle-robotics/11_run/onrig_calibrate.py --selftest --output_dir outputs/onrig
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS, build_striker

TOL = 0.05                     # the outcome bar, relative
RATIOS = (0.5, 1.0, 2.0, 3.0)  # certified command range
V_IN_NOMINAL = 1.0
N_BOOT = 2000


# -- the spec ---------------------------------------------------------------------------------------

def alpha_precision_required(ratio_min: float, tol: float = TOL) -> float:
    """Relative precision on ``alpha`` needed to hold ``tol`` down to ``ratio_min``. See module docs.

    Closed form, so it transfers to any rig rather than being a fact about this simulator.
    """
    return tol * ratio_min / (ratio_min + 1.0)


def predicted_rel_err(alpha_hat: float, alpha_true: float, ratio: float) -> float:
    """Exact relative outcome error from commanding with ``alpha_hat`` on a rig with ``alpha_true``."""
    return abs(alpha_true / alpha_hat - 1.0) * (ratio + 1.0) / ratio


# -- fitting ----------------------------------------------------------------------------------------

def _alpha_constrained(v_in: np.ndarray, v_p: np.ndarray, v_out: np.ndarray) -> float:
    """One-parameter least squares for ``v_out - v_in = alpha (v_p - v_in)``, through the origin."""
    x, y = v_p - v_in, v_out - v_in
    denom = float(x @ x)
    if denom <= 0:
        raise ValueError("degenerate calibration set: every strike has v_p == v_in")
    return float(x @ y) / denom


def bootstrap_alpha(v_in: np.ndarray, v_p: np.ndarray, v_out: np.ndarray,
                    n_boot: int = N_BOOT, seed: int = 0) -> dict[str, float]:
    """Resample strikes with replacement to get a CI on alpha.

    Bootstrapping the MEASUREMENTS is the honest uncertainty here: it folds in sensing noise and the
    particular strikes that happened to be collected, which are the two things an operator controls.
    """
    rng = np.random.default_rng(seed)
    n = len(v_in)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            draws.append(_alpha_constrained(v_in[i], v_p[i], v_out[i]))
        except ValueError:
            continue
    a = np.asarray(draws)
    point = _alpha_constrained(v_in, v_p, v_out)
    return {"alpha": point,
            "alpha_boot_mean": float(a.mean()),
            "alpha_std": float(a.std(ddof=1)),
            "alpha_lo95": float(np.quantile(a, 0.025)),
            "alpha_hi95": float(np.quantile(a, 0.975)),
            # Half-width of the 95% interval as a fraction of alpha: the quantity the spec bounds.
            "rel_precision": float((np.quantile(a, 0.975) - np.quantile(a, 0.025)) / 2 / abs(point)),
            "n_strikes": int(n)}


# -- selftest rig -----------------------------------------------------------------------------------

def synth_measurements(solref: tuple[float, float], n: int, noise: float,
                       seed: int = 0) -> tuple[dict[str, np.ndarray], float]:
    """Synthesise rig measurements from MuJoCo with the given contact physics and sensing noise.

    Returns the noisy measurement set and the rig's TRUE alpha (fitted on the clean sweep), which a
    real rig obviously does not hand you -- it is used only to score the procedure.
    """
    rng = np.random.default_rng(seed)
    gen = build_striker("paddle", solref=solref)
    grid = sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20))
    alpha_true = _alpha_constrained(grid["v_in"], grid["v_p"], grid["v_out"])
    idx = rng.choice(len(grid["v_out"]), size=n, replace=False)
    meas = {
        # v_p is COMMANDED, so it is known exactly; v_in and v_out are SENSED, so both carry noise.
        "v_in": grid["v_in"][idx] + rng.normal(0, noise, n),
        "v_p": grid["v_p"][idx],
        "v_out": grid["v_out"][idx] + rng.normal(0, noise, n),
    }
    return meas, alpha_true


def verify_on_rig(solref: tuple[float, float], alpha_hat: float) -> list[dict[str, Any]]:
    """EXECUTE the calibrated commands on the simulated rig and measure what comes back.

    The bootstrap CI is a statement about the fit; this is the falsifiable part. On a real rig this is
    the acceptance test you run after calibrating, with the same commands.
    """
    gen = build_striker("paddle", solref=solref)
    rows = []
    for r in RATIOS:
        target = -r * V_IN_NOMINAL
        v_p = V_IN_NOMINAL + (target - V_IN_NOMINAL) / alpha_hat
        try:
            achieved = float(gen.simulate(V_IN_NOMINAL, v_p, render=False)["v_out_world"])
            err = abs(achieved - target) / abs(target)
        except RuntimeError:                     # action outside the rig's envelope
            achieved, err = float("nan"), float("inf")
        rows.append({"ratio": r, "target": target, "v_p": v_p,
                     "achieved": achieved, "rel_err": err, "pass": bool(err <= TOL)})
    return rows


# -- report -----------------------------------------------------------------------------------------

def assess(boot: dict[str, float], ratio_min: float) -> dict[str, Any]:
    need = alpha_precision_required(ratio_min)
    got = boot["rel_precision"]
    # Strikes scale as 1/sqrt(n), so the shortfall converts directly into a strike count.
    n_needed = int(np.ceil(boot["n_strikes"] * (got / need) ** 2)) if got > need else boot["n_strikes"]
    return {"alpha_precision_required": need, "alpha_precision_achieved": got,
            "calibration_sufficient": bool(got <= need),
            "n_strikes_have": boot["n_strikes"], "n_strikes_needed_estimate": n_needed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--measurements", help="CSV with columns v_in,v_p,v_out (m/s, world signs)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthesise measurements from MuJoCo instead of reading a rig")
    ap.add_argument("--selftest_noise", type=float, default=0.02,
                    help="sensing noise std in m/s for --selftest (0.028 is the tracker's spec)")
    ap.add_argument("--selftest_n", type=int, default=20)
    ap.add_argument("--ratio_min", type=float, default=min(RATIOS))
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"tol": TOL, "ratios": list(RATIOS)}

    # The nominal simulator law, for reference: this is what you would use if you did NOT calibrate.
    nominal_solref = (-120000.0, -18.0)
    gen = build_striker("paddle")
    nominal = StrikeInverse.fit(sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15],
                                             np.linspace(0.25, -1.45, 20)), BALL_MASS, PADDLE_MASS)
    grid_nom = sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20))
    alpha_nominal = _alpha_constrained(grid_nom["v_in"], grid_nom["v_p"], grid_nom["v_out"])
    report["nominal"] = {"alpha_free": nominal.alpha, "beta_free": nominal.beta,
                         "alpha_constrained": alpha_nominal}

    print(f"nominal simulator: alpha_free={nominal.alpha:.5f} beta_free={nominal.beta:.5f} "
          f"(sum={nominal.alpha + nominal.beta:.5f}), alpha_constrained={alpha_nominal:.5f}")
    need = alpha_precision_required(args.ratio_min)
    print(f"spec: alpha must be known to {need:.2%} relative to hold {TOL:.0%} down to "
          f"ratio {args.ratio_min:g}\n")

    cases: list[tuple[str, tuple[float, float]]] = []
    if args.selftest:
        # Case A: the rig IS the nominal sim. Sanity -- calibration must agree with nominal.
        # Case B: the rig has softer, lossier contact, i.e. a genuinely different alpha. This is the
        # case that matters, and the one a procedure that merely echoes the sim would fail.
        cases = [("rig_matches_sim", nominal_solref), ("rig_differs", (-45000.0, -32.0))]
    elif args.measurements:
        cases = [("rig", nominal_solref)]
    else:
        raise SystemExit("pass --measurements CSV or --selftest")

    report["cases"] = {}
    for name, solref in cases:
        if args.selftest:
            meas, alpha_true = synth_measurements(solref, args.selftest_n, args.selftest_noise)
        else:
            arr = np.genfromtxt(args.measurements, delimiter=",", names=True)
            meas = {k: np.atleast_1d(arr[k]).astype(float) for k in ("v_in", "v_p", "v_out")}
            alpha_true = float("nan")

        boot = bootstrap_alpha(meas["v_in"], meas["v_p"], meas["v_out"])
        verdict = assess(boot, args.ratio_min)
        entry: dict[str, Any] = {"solref": list(solref), "alpha_true": alpha_true,
                                 **boot, **verdict}

        # How far off the NOMINAL model would have been on this rig, and how far the CALIBRATED one is.
        if np.isfinite(alpha_true):
            entry["nominal_alpha_rel_offset"] = abs(alpha_nominal / alpha_true - 1.0)
            entry["predicted_err_if_uncalibrated"] = {
                f"{r:g}": predicted_rel_err(alpha_nominal, alpha_true, r) for r in RATIOS}
            entry["predicted_err_calibrated"] = {
                f"{r:g}": predicted_rel_err(boot["alpha"], alpha_true, r) for r in RATIOS}
            entry["executed_calibrated"] = verify_on_rig(solref, boot["alpha"])
            entry["executed_uncalibrated"] = verify_on_rig(solref, alpha_nominal)

        report["cases"][name] = entry

        print(f"--- {name} (solref={solref}) ---")
        print(f"  alpha_true          {alpha_true:.5f}")
        print(f"  alpha_calibrated    {boot['alpha']:.5f}  "
              f"[{boot['alpha_lo95']:.5f}, {boot['alpha_hi95']:.5f}]  "
              f"from {boot['n_strikes']} strikes")
        print(f"  precision           {boot['rel_precision']:.2%}  (need <= {need:.2%})  -> "
              f"{'SUFFICIENT' if verdict['calibration_sufficient'] else 'INSUFFICIENT, want ~' + str(verdict['n_strikes_needed_estimate']) + ' strikes'}")
        if np.isfinite(alpha_true):
            print(f"  nominal model off by {entry['nominal_alpha_rel_offset']:.2%} on this rig")
            for row, row_u in zip(entry["executed_calibrated"], entry["executed_uncalibrated"]):
                print(f"    ratio {row['ratio']:<4g} calibrated {row['rel_err']:7.2%} "
                      f"{'PASS' if row['pass'] else 'FAIL'}   |   "
                      f"uncalibrated {row_u['rel_err']:7.2%} "
                      f"{'PASS' if row_u['pass'] else 'FAIL'}")
        print()

    (out / "calibration.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {out}/calibration.json")


if __name__ == "__main__":
    main()
