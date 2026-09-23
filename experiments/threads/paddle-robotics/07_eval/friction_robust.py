

#!/usr/bin/env python
"""Make the strike law survive a table with friction -- the largest sim-to-real gap.

WHY THIS EXISTS
---------------
``SPECS.md`` section 6 reports the certified controller failing at any realistic table friction:
mu=0.005 gives 8.39% outcome error, mu=0.01 gives 17.05%, mu>=0.05 misses contact entirely, against
a 5% bar. Every real table has friction, so as it stands the certificate covers a world that does
not exist. That section also notes the first post-contact sample is barely better than the window
mean (7.16% vs 8.39% at mu=0.005), which rules out "just measure in a shorter window" as the fix and
is why this needed its own investigation rather than a one-line change.

WHAT IS ACTUALLY BROKEN
-----------------------
Three distinct things, and only the third is the strike law's fault:

  1. ``v_in`` is wrong. The controller is handed the LAUNCH speed, but the ball decelerates over the
     ~1 m of approach, so the speed the blade actually meets is lower. The law is fed an input it
     never sees.
  2. ``v_out`` is wrong. The label is the mean over the 16-frame post window, and the ball is
     decelerating all the way down it, so the mean sits below the speed the strike produced.
  3. Whatever is left after 1 and 2 are referred to the contact instant is the genuine effect of
     friction ON THE IMPACT, which is the only part a better law would have to model.

The fix for 1 and 2 is the same idea: stop referring velocities to the window and refer them to the
CONTACT INSTANT, by fitting a line to each window's velocity trace and extrapolating to contact.
That is rig-realizable -- a tracker that can measure speed at all can fit a line through its own
samples -- which is the test every candidate fix here has to pass. Re-fitting alpha/beta against
ground truth would not be.

WHAT IS REPORTED
----------------
* ``diagnose``     -- decomposes the error into the three sources above, per mu.
* ``identify``     -- refits the one-parameter law on contact-referred quantities at each mu and
                      checks alpha is mu-INVARIANT. If it is, one calibration covers every table.
* ``closed_loop``  -- plans with the contact-referred law and executes, scoring against both the
                      contact-referred outcome and the old window mean, per mu and ratio.

A note on what "the outcome" means once friction exists. With a frictionless table the post-contact
speed is constant, so "speed at contact" and "mean over the window" are the same number and the
distinction never arises. With friction they genuinely differ and the target has to pick one. This
script controls the speed AT CONTACT, because that is the property of the strike; the window mean is
then a downstream consequence of how long you look and how rough the table is. Both are reported so
the choice is visible rather than buried.

Usage:
    python experiments/threads/paddle-robotics/07_eval/friction_robust.py --output_dir <dir> [--quick]
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
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paddle.hardware_specs import PerturbedStrike, rel_err  # noqa: E402
from src.control.strike_inverse import StrikeInverse, sweep_strikes  # noqa: E402
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS  # noqa: E402

TOL = 0.05
RATIOS = (0.5, 1.0, 2.0, 3.0)
V_IN = 1.0
# 0 is the certified world; 0.005-0.02 is the band SPECS.md already showed failing; 0.05+ is where
# the swing stops reaching the ball at all, which is a different failure and reported as such.
MUS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)


# -- contact-referred measurement --------------------------------------------------------------

def window_times(r: dict, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame-time of every pre- and post-window velocity sample.

    The pre window is frames 0..T-1. The post window does NOT start at T: the swing gap sits in
    between and is never rendered, so it starts at T + gap_frames. Getting this offset wrong biases
    the post extrapolation by the whole gap (~2-6 frames), which is larger than the effect being
    measured, so it is derived from the episode rather than assumed.
    """
    t = int(num_frames)
    t_pre = np.arange(t, dtype=float)
    t_post = t + float(r["gap_frames"]) + np.arange(t, dtype=float)
    return t_pre, t_post


def contact_referred(r: dict, num_frames: int) -> dict[str, float]:
    """Extrapolate both windows' velocity traces to the contact instant.

    Linear, not higher-order, and deliberately so: a sliding ball under Coulomb friction decelerates
    at a constant rate, so a line is the correct model during sliding, and once it is rolling the
    slope is ~0 and a line is still correct. A quadratic would fit the sliding-to-rolling knee but
    would also happily extrapolate a curve out of tracker noise. ``slope`` is returned so the caller
    can see which regime it was in rather than trusting the fit blind.
    """
    t = int(num_frames)
    v = np.asarray(r["ball_v"], dtype=float)
    t_pre, t_post = window_times(r, t)
    v_pre, v_post = v[:t], v[t:2 * t]

    m_pre, c_pre = np.polyfit(t_pre, v_pre, 1)
    m_post, c_post = np.polyfit(t_post, v_post, 1)
    t_c0 = float(r["contact_frame"])
    t_c1 = float(r["contact_end_frame"])

    return {
        "v_in_launch": float(r["v_in_cmd"]),
        "v_in_window_mean": float(v_pre.mean()),
        "v_in_at_contact": float(m_pre * t_c0 + c_pre),
        "v_out_window_mean": float(v_post.mean()),
        "v_out_first_sample": float(v_post[0]),
        "v_out_at_contact": float(m_post * t_c1 + c_post),
        "pre_slope_per_frame": float(m_pre),
        "post_slope_per_frame": float(m_post),
        "contact_frame": t_c0,
        "contact_end_frame": t_c1,
        "v_p_actual": float(r["v_p_actual"]),
    }


def measure(gen: Any, v_in: float, v_p: float) -> dict[str, float] | None:
    try:
        r = gen.simulate(float(v_in), float(v_p), render=False)
    except RuntimeError:
        return None
    return contact_referred(r, gen.num_frames)


# -- part 1: where does the error come from -----------------------------------------------------

def diagnose(inv0: StrikeInverse, mus: tuple[float, ...]) -> list[dict]:
    """Split the failure into approach decay, window decay, and residual impact error."""
    rows = []
    for mu in mus:
        pert = PerturbedStrike(table_friction=mu)
        for ratio in RATIOS:
            target = -ratio * V_IN
            # what the CURRENT pipeline does: plan from the launch speed with the nominal law
            v_p = float(np.asarray(inv0.action_for(V_IN, target, mode="linear")).ravel()[0])
            m = measure(pert, V_IN, v_p)
            if m is None:
                rows.append({"mu": mu, "ratio": ratio, "missed_contact": True})
                continue
            # what the law PREDICTS given the speed the blade actually met
            pred_at_contact = inv0.forward(m["v_in_at_contact"], m["v_p_actual"])
            rows.append({
                "mu": mu, "ratio": ratio, "missed_contact": False, "target": target,
                "v_in_launch": V_IN, "v_in_at_contact": m["v_in_at_contact"],
                "approach_loss_m_s": V_IN - m["v_in_at_contact"],
                "v_out_at_contact": m["v_out_at_contact"],
                "v_out_window_mean": m["v_out_window_mean"],
                "window_loss_m_s": abs(m["v_out_at_contact"]) - abs(m["v_out_window_mean"]),
                # the three errors, same target, increasingly fair measurement
                "err_window_mean": rel_err(m["v_out_window_mean"], target),
                "err_first_sample": rel_err(m["v_out_first_sample"], target),
                "err_at_contact": rel_err(m["v_out_at_contact"], target),
                # what is left once the law is given the true input AND read at contact: the part
                # friction changes about the IMPACT itself, which is the only irreducible piece
                "err_law_residual": rel_err(m["v_out_at_contact"], float(pred_at_contact)),
                "post_slope_per_frame": m["post_slope_per_frame"],
            })
    return rows


# -- part 2: is the contact-referred law the same law at every mu -------------------------------

def identify(mus: tuple[float, ...], v_ins: list[float], v_ps: np.ndarray) -> list[dict]:
    """Refit ``v_out - v_in = alpha (v_p - v_in)`` on contact-referred quantities, per mu.

    If alpha comes out mu-invariant the friction problem is entirely a measurement problem and one
    calibration covers every table. If it drifts with mu, friction is changing the impact and the
    rig has to be calibrated on its own table.
    """
    rows = []
    for mu in mus:
        pert = PerturbedStrike(table_friction=mu)
        xs, ys, n_missed = [], [], 0
        for v_in in v_ins:
            for v_p in v_ps:
                m = measure(pert, v_in, float(v_p))
                if m is None:
                    n_missed += 1
                    continue
                # the one-parameter form, so alpha+beta=1 is imposed rather than hoped for
                xs.append(m["v_p_actual"] - m["v_in_at_contact"])
                ys.append(m["v_out_at_contact"] - m["v_in_at_contact"])
        if len(xs) < 3:
            rows.append({"mu": mu, "n": len(xs), "n_missed": n_missed, "alpha": float("nan")})
            continue
        x, y = np.asarray(xs), np.asarray(ys)
        alpha = float(x @ y / (x @ x))
        resid = np.abs(alpha * x - y)
        rows.append({
            "mu": mu, "n": int(len(x)), "n_missed": int(n_missed), "alpha": alpha,
            "beta": 1.0 - alpha, "e_eff": alpha - 1.0,
            "resid_max": float(resid.max()),
            "resid_frac_of_range": float(resid.max() / (y.max() - y.min())),
        })
    return rows


# -- part 3: close the loop with the contact-referred law ---------------------------------------

def closed_loop(inv_by_mu: dict[float, float], mus: tuple[float, ...],
                v_ins: list[float]) -> list[dict]:
    """Plan with the contact-referred law and a MEASURED approach speed; execute; score.

    Two things a rig can do are assumed, and nothing else: it can see the ball's speed shortly
    before contact, and it can see it shortly after. Both come from the same tracker that already
    has to exist. In particular the launch speed is NOT assumed known -- that is what broke before.

    The approach speed is taken from a probe strike at the same launch speed. On a rig this is just
    "watch one ball come in", and it is needed because the action has to be chosen BEFORE contact,
    so the controller cannot use the contact-referred input from the strike it is planning.
    """
    rows = []
    for mu in mus:
        alpha = inv_by_mu[mu]
        if not np.isfinite(alpha):
            continue
        pert = PerturbedStrike(table_friction=mu)
        for v_in in v_ins:
            # probe: one incoming ball, no strike commanded, to learn the approach decay
            probe = measure(pert, v_in, 0.0)
            v_in_c = probe["v_in_at_contact"] if probe else v_in
            for ratio in RATIOS:
                target = -ratio * v_in
                # one-parameter inverse: v_out - v_in = alpha (v_p - v_in)
                v_p = v_in_c + (target - v_in_c) / alpha
                m = measure(pert, v_in, float(v_p))
                if m is None:
                    rows.append({"mu": mu, "v_in": v_in, "ratio": ratio, "missed_contact": True})
                    continue
                rows.append({
                    "mu": mu, "v_in": v_in, "ratio": ratio, "missed_contact": False,
                    "target": target, "v_p": float(v_p),
                    "v_in_at_contact_probe": v_in_c,
                    "achieved_at_contact": m["v_out_at_contact"],
                    "achieved_window_mean": m["v_out_window_mean"],
                    "err_at_contact": rel_err(m["v_out_at_contact"], target),
                    "err_window_mean": rel_err(m["v_out_window_mean"], target),
                })
    return rows


# -- report --------------------------------------------------------------------------------------

def pct(x: float) -> str:
    return "MISS" if not np.isfinite(x) else f"{x * 100:.2f}%"


def write_report(out: Path, diag: list[dict], ident: list[dict], loop: list[dict]) -> None:
    L = ["# Making the strike law friction-robust", "",
         f"Bar: relative outcome error <= {TOL:.0%}. `SPECS.md` section 6 had the certified "
         "controller at",
         "8.39% by mu=0.005 and 17.05% by mu=0.01, i.e. failing on any real table.", "",
         "## 1. Where the error actually comes from", "",
         "Same episodes, same target, three increasingly fair measurements. `approach loss` is how "
         "much",
         "speed the ball sheds before the blade meets it; `err at contact` is what is left once both",
         "velocities are referred to the contact instant; `law residual` is the contact-referred",
         "outcome against what the frictionless law predicts from the inputs it actually saw.", "",
         "| mu | ratio | approach loss | err window mean | err first sample | err at contact | "
         "law residual |",
         "|---|---|---|---|---|---|---|"]
    for r in diag:
        if r.get("missed_contact"):
            L.append(f"| {r['mu']} | {r['ratio']} | MISS | MISS | MISS | MISS | MISS |")
            continue
        L.append(f"| {r['mu']} | {r['ratio']} | {r['approach_loss_m_s']:.4f} m/s | "
                 f"{pct(r['err_window_mean'])} | {pct(r['err_first_sample'])} | "
                 f"{pct(r['err_at_contact'])} | {pct(r['err_law_residual'])} |")

    L += ["", "## 2. Is it the same law on every table", "",
          "The one-parameter law `v_out - v_in = alpha (v_p - v_in)` refitted on contact-referred",
          "quantities at each mu. If alpha is flat in mu, friction never touched the impact and the",
          "whole problem was measurement.", "",
          "| mu | n strikes | alpha | e_eff | residual (frac of range) | missed contact |",
          "|---|---|---|---|---|---|"]
    for r in ident:
        if not np.isfinite(r.get("alpha", float("nan"))):
            L.append(f"| {r['mu']} | {r['n']} | n/a | n/a | n/a | {r['n_missed']} |")
            continue
        L.append(f"| {r['mu']} | {r['n']} | {r['alpha']:.5f} | {r['e_eff']:.4f} | "
                 f"{r['resid_frac_of_range'] * 100:.3f}% | {r['n_missed']} |")
    fin = [r["alpha"] for r in ident if np.isfinite(r.get("alpha", float("nan")))]
    if len(fin) > 1:
        spread = (max(fin) - min(fin)) / np.mean(fin)
        L += ["", f"alpha spread across mu: **{spread * 100:.2f}%** of its mean "
                  f"({min(fin):.5f} to {max(fin):.5f}).",
              "", "The on-rig calibration analysis requires alpha to 1.67% relative to hold the bar "
                  "down to",
              "ratio 0.5, so compare the spread against that number, not against zero."]

    L += ["", "## 3. Closing the loop with the contact-referred law", "",
          "Planned with the refitted law and a MEASURED approach speed (one probe ball), executed in "
          "the",
          "frictional world. Scored two ways against the same target: the speed the strike produced,",
          "and the old 16-frame window mean.", "",
          "| mu | pass@5% at contact | median err at contact | pass@5% window mean | "
          "median err window mean |",
          "|---|---|---|---|---|"]
    for mu in sorted({r["mu"] for r in loop}):
        sub = [r for r in loop if r["mu"] == mu]
        ec = np.array([r.get("err_at_contact", np.inf) for r in sub], dtype=float)
        ew = np.array([r.get("err_window_mean", np.inf) for r in sub], dtype=float)
        fc, fw = ec[np.isfinite(ec)], ew[np.isfinite(ew)]
        L.append(f"| {mu} | {(ec <= TOL).mean():.0%} | "
                 f"{pct(np.median(fc)) if len(fc) else 'MISS'} | {(ew <= TOL).mean():.0%} | "
                 f"{pct(np.median(fw)) if len(fw) else 'MISS'} |")

    L += ["", "Read the two outcome columns as different questions, not as a good and a bad result.",
          "`at contact` is whether the strike did what it was told. `window mean` is what a camera",
          "averaging 16 frames will report afterwards, and it falls away from the target as mu rises",
          "for a reason that has nothing to do with control: the ball is slowing down while being",
          "watched. A rig that needs the window mean to hit a target has to be told a target for the",
          "window mean.", ""]
    (out / "FRICTION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="smaller identification sweep, for a smoke test")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mus = MUS[:4] if args.quick else MUS
    v_ins_id = [0.95, 1.05] if args.quick else [0.85, 0.95, 1.05, 1.15]
    v_ps = np.linspace(0.25, -1.45, 6 if args.quick else 14)
    v_ins_loop = [1.0] if args.quick else [0.9, 1.0, 1.1]

    print("[1/4] fitting the nominal (frictionless) law ...", flush=True)
    base = PerturbedStrike(table_friction=0.0)
    inv0 = StrikeInverse.fit_constrained(
        sweep_strikes(base, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20)),
        BALL_MASS, PADDLE_MASS)
    print(f"      alpha={inv0.alpha:.5f} beta={inv0.beta:.5f}", flush=True)

    print(f"[2/4] diagnosing the error split over mu={mus} ...", flush=True)
    diag = diagnose(inv0, mus)

    print("[3/4] re-identifying the law on contact-referred quantities ...", flush=True)
    ident = identify(mus, v_ins_id, v_ps)
    inv_by_mu = {r["mu"]: r.get("alpha", float("nan")) for r in ident}

    print("[4/4] closing the loop ...", flush=True)
    loop = closed_loop(inv_by_mu, mus, v_ins_loop)

    (out / "friction.json").write_text(json.dumps(
        {"nominal": inv0.to_dict(), "diagnose": diag, "identify": ident, "closed_loop": loop,
         "config": {"mus": list(mus), "ratios": list(RATIOS), "tol": TOL,
                    "v_ins_identify": v_ins_id, "v_ins_loop": v_ins_loop}},
        indent=2, default=float))
    write_report(out, diag, ident, loop)
    print(f"\nwrote {out}/FRICTION.md and friction.json", flush=True)


if __name__ == "__main__":
    main()
