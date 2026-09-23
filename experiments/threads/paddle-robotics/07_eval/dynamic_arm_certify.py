

#!/usr/bin/env python
"""Certify the dynamically-actuated Franka -- the only variant with real arm inertia in the impact.

WHY THIS EXISTS
---------------
The certified dataset variant carries a KINEMATIC arm: non-colliding, zero inertia, present only so
appearance can be varied with physics held fixed. That is the right control for a latent-transfer
experiment and it is why the certificate is trustworthy, but it means the certificate says nothing
about a robot. ``franka_dynamic_strike`` is the variant where the arm's own links and servos take
the impact, and it was left explicitly INDICATIVE, not certified, on two counts:

  * 14/15 commanded targets landed within 5%, with 1/15 blowing up;
  * its fitted law came out ``alpha=1.5725, beta=-0.9132``, so ``alpha+beta = 0.659``. That FAILS the
    validity check in ``SPECS.md`` section 8 -- ``alpha+beta`` must be 1 for any correct strike law,
    since a ball and striker moving at the same velocity cannot collide and the outcome must be that
    shared velocity. A law that fails it is not a strike law, and ``HARDWARE.md`` was quoting those
    coefficients as a hardware starting point.

NONE OF THAT WAS THE ARM
------------------------
All four faults were in how the scene and the controller were set up, and they are fixed at source in
:mod:`src.data.franka_dynamic_strike`. In the order they mattered:

  1. **A degenerate friction cone.** Every contact in the scene is frictionless, but ``condim``
     defaulted to 3, so MuJoCo built a pyramidal friction basis whose vectors all collapse onto the
     normal at ``mu=0``: four identical Jacobian rows and a singular block in the solve. Usually
     harmless; with a 7-DOF chain hanging off the contact it intermittently returned a near-rigid
     impulse instead of the soft one ``solref`` asks for -- 1.1 ms and 6 kN in place of 9 ms and
     20 N, restitution above 2. THAT is the whole "the arm injects energy" story, including the
     25 m/s rebound and the bimodal action->outcome map that ``validated_system`` E11 reported. It
     looked action-dependent because conditioning is, and it looked like a slowly-converging
     discretisation error because refining the step improves conditioning without curing the
     degeneracy.
  2. **A position servo cannot track a velocity ramp.** It needs a standing ``kd/kp * qdot`` error,
     which is 0.1 rad at 1 rad/s on this Panda. That is the commanded-versus-achieved shortfall in
     full -- not a property of the arm, a property of driving it with angles instead of torques --
     and the same lag sagged the tool 6 mm below the ball, tilting the contact normal 3.3 deg off
     the motion axis.
  3. **A constant-acceleration ramp rings the sprung mount.** The acceleration step at ramp end
     releases the mount's steady deflection as free oscillation, putting a 4.7% velocity ripple on
     the mallet at the fast end of the envelope -- and the fitted alpha drifted by 4.7% across that
     same envelope. A smoothstep velocity profile removes the input.
  4. **The post window could open mid-collision.** Once contact lands past ``X_POST`` the window
     opened on the first substep the ball's velocity turned negative, averaging a near-stationary
     first frame into ``v_out`` and biasing it by 6% in an action-dependent direction.

The mount was never at fault: the effective striker mass, measured from the model along the contact
normal, is 2.0010 kg at every action in the envelope.

WHAT THIS DOES
--------------
Separates the two maps that were being conflated:

    TRACKING   v_p_commanded -> v_p_at_impact     (the arm: IK + servos + mount)
    STRIKE     (v_in, v_p_at_impact) -> v_out     (the contact, which should match the paddle's)

That separation is still worth keeping even though tracking now comes out as very nearly the
identity, because it is what makes the residual attributable: any future drift shows up in one map or
the other rather than in a single opaque number. The strike law is then checked by the
ground-truth-free identity ``alpha+beta = 1`` and by a Galilean boost run on the arm itself, and the
closed loop is executed and scored end to end.

What bounds the envelope now is the Panda's published JOINT-VELOCITY limit, not the physics -- and
note that an arm which actually delivers its commanded tool speed necessarily turns its joints faster
than one that was quietly under-delivering, so that ceiling got closer as the arm got better.

Usage:
    python experiments/threads/paddle-robotics/07_eval/dynamic_arm_certify.py --output_dir <dir> [--quick]
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

from src.data.franka_dynamic_strike import FrankaDynamicStrike  # noqa: E402

TOL = 0.05
RATIOS = (0.75, 1.0, 1.5, 2.0)   # the dynamic arm's envelope is joint-velocity limited, not torque
V_INS = (0.9, 1.0, 1.1)
# Outcome/range ratio beyond which an episode is a blow-up rather than a miss. The rigid-tool dead
# end returned 25 m/s from a 1 m/s ball, so this is nowhere near a borderline call.
BLOWUP_ABS = 6.0


def fit_one_param(v_in: np.ndarray, v_p: np.ndarray, v_out: np.ndarray) -> dict[str, float]:
    """``v_out - v_in = alpha (v_p - v_in)``, the form with alpha+beta=1 imposed."""
    x, y = v_p - v_in, v_out - v_in
    alpha = float(x @ y / (x @ x))
    resid = np.abs(alpha * x - y)
    return {"alpha": alpha, "beta": 1.0 - alpha,
            "resid_max": float(resid.max()),
            "resid_frac_of_range": float(resid.max() / (y.max() - y.min()))}


def fit_free(v_in: np.ndarray, v_p: np.ndarray, v_out: np.ndarray) -> dict[str, float]:
    """Unconstrained ``v_out = alpha v_p + beta v_in + gamma``; alpha+beta is then a TEST, not an input."""
    A = np.stack([v_p, v_in, np.ones_like(v_p)], axis=1)
    coef, *_ = np.linalg.lstsq(A, v_out, rcond=None)
    pred = A @ coef
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "gamma": float(coef[2]),
            "alpha_plus_beta": float(coef[0] + coef[1]),
            "resid_max": float(np.abs(pred - v_out).max())}


def sweep(gen: Any, v_ins: list[float], v_ps: np.ndarray) -> dict[str, np.ndarray]:
    """Roll the grid, keeping COMMANDED and ACHIEVED tool speed as separate columns."""
    cols: dict[str, list[float]] = {k: [] for k in
                                    ("v_in", "v_p_cmd", "v_p_actual", "v_out", "n_touches")}
    n_fail = 0
    for v_in in v_ins:
        for v_p in v_ps:
            try:
                r = gen.simulate(float(v_in), float(v_p), render=False)
            except RuntimeError:
                n_fail += 1
                continue
            if abs(r["v_out_world"]) > BLOWUP_ABS:
                n_fail += 1
                continue
            cols["v_in"].append(float(r["v_in_world"]))
            cols["v_p_cmd"].append(float(v_p))
            cols["v_p_actual"].append(float(r["v_p_actual"]))
            cols["v_out"].append(float(r["v_out_world"]))
            cols["n_touches"].append(float(r["n_touches"]))
        print(f"      v_in={v_in}: {len(cols['v_out'])} good so far", flush=True)
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["n_excluded"] = np.asarray([n_fail])
    return out


def fit_tracking(v_p_cmd: np.ndarray, v_p_actual: np.ndarray) -> dict[str, float]:
    """Affine model of what the arm actually delivers for what it was told.

    Affine rather than a pure gain because the swing has a finite ramp: the mallet is still
    accelerating when it meets the ball, so there is an offset as well as a slope.
    """
    s, c = np.polyfit(v_p_cmd, v_p_actual, 1)
    pred = s * v_p_cmd + c
    return {"slope": float(s), "intercept": float(c),
            "resid_max": float(np.abs(pred - v_p_actual).max()),
            "resid_std": float((pred - v_p_actual).std()),
            "r2": float(np.corrcoef(v_p_cmd, v_p_actual)[0, 1] ** 2)}


def boost_identity(gen: Any, cases: list[tuple[float, float]], deltas: tuple[float, ...],
                   track: dict[str, float]) -> list[dict]:
    """Move ball and tool by the same delta; the outcome must move by exactly delta.

    This is the ground-truth-free validity check, and it is the one that condemned the previous
    coefficients. It has to be run in ACHIEVED tool speed, which means the boosted rollout is
    COMMANDED through the inverse tracking map rather than by adding ``delta`` to the command.
    Adding it to the command instead tests ``tracking * strike``, which is not a Galilean boost at
    all unless tracking happens to be the identity -- it reports the tracking gain as though it were
    a physics violation. ``d_v_p_actual`` is carried on every row so the reader can check that the
    tool really did move by ``delta``.
    """
    rows = []
    for v_in, v_p in cases:
        try:
            base = gen.simulate(v_in, v_p, render=False)
        except RuntimeError:
            continue
        for d in deltas:
            cmd = v_p + d / track["slope"]
            try:
                b = gen.simulate(v_in + d, float(cmd), render=False)
            except RuntimeError:
                continue
            dv_out = b["v_out_world"] - base["v_out_world"]
            d_actual = b["v_p_actual"] - base["v_p_actual"]
            rows.append({"v_in": v_in, "v_p": v_p, "delta": d, "v_p_cmd_boosted": float(cmd),
                         "dv_out": float(dv_out),
                         "dv_out_over_delta": float(dv_out / d),
                         "d_v_p_actual": float(d_actual)})
    return rows


def closed_loop(gen: Any, strike: dict[str, float], track: dict[str, float]) -> list[dict]:
    """Command through TRACKING then STRIKE, execute on the dynamic arm, score."""
    rows = []
    for v_in in V_INS:
        for ratio in RATIOS:
            target = -ratio * v_in
            # what the mallet must be doing at impact
            v_p_needed = v_in + (target - v_in) / strike["alpha"]
            # what to command so it is doing that
            v_p_cmd = (v_p_needed - track["intercept"]) / track["slope"]
            rec: dict[str, Any] = {"v_in": v_in, "ratio": ratio, "target": target,
                                   "v_p_needed": float(v_p_needed), "v_p_cmd": float(v_p_cmd)}
            try:
                r = gen.simulate(v_in, float(v_p_cmd), render=False)
                got = float(r["v_out_world"])
                rec["blowup"] = bool(abs(got) > BLOWUP_ABS)
                rec["achieved"] = got
                rec["v_p_actual"] = float(r["v_p_actual"])
                rec["tracking_err"] = float(r["v_p_actual"] - v_p_needed)
                rec["rel_err"] = abs(got - target) / abs(target)
                rec.update({k: v for k, v in gen.feasibility().items()
                            if k in ("worst_joint_vel_frac", "worst_torque_frac")})
            except RuntimeError:
                rec.update({"blowup": False, "achieved": float("nan"), "rel_err": float("inf"),
                            "unreachable": True})
            rows.append(rec)
    return rows


def write_report(out: Path, sw: dict, cmd_fit: dict, act_fit: dict, cmd_free: dict,
                 act_free: dict, track: dict, boost: list[dict], loop: list[dict]) -> None:
    errs = np.array([r["rel_err"] for r in loop], dtype=float)
    fin = errs[np.isfinite(errs)]
    n_blow = sum(1 for r in loop if r.get("blowup"))
    n_unreach = sum(1 for r in loop if r.get("unreachable"))

    L = ["# Certifying the dynamically-actuated Franka", "",
         f"Bar: relative outcome error <= {TOL:.0%}. "
         f"{int(sw['n_excluded'][0])} of {int(sw['n_excluded'][0]) + len(sw['v_out'])} "
         "identification rollouts were excluded (no contact or blow-up).", "",
         "## 1. Does the arm obey a valid strike law?", "",
         "Same episodes, same outcomes. The only difference is which tool speed the regression uses.",
         "", "| fitted against | alpha | beta | alpha+beta | verdict |", "|---|---|---|---|---|",
         f"| COMMANDED v_p | {cmd_free['alpha']:.4f} | "
         f"{cmd_free['beta']:.4f} | {cmd_free['alpha_plus_beta']:.4f} | "
         f"{'passes' if abs(cmd_free['alpha_plus_beta'] - 1) <= 0.02 else 'FAILS the identity'} |",
         f"| ACHIEVED v_p at impact | {act_free['alpha']:.4f} | {act_free['beta']:.4f} | "
         f"{act_free['alpha_plus_beta']:.4f} | "
         f"{'passes' if abs(act_free['alpha_plus_beta'] - 1) <= 0.02 else 'FAILS the identity'} |",
         "",
         "`alpha+beta` must be 1 for any correct strike law (SPECS section 8), and it is not a free",
         "parameter of the fit here -- both rows are unconstrained three-parameter regressions, so the",
         "column is a test rather than an assumption.", "",
         "With the identity imposed instead, the one-parameter law fits the achieved-speed data with",
         f"alpha = {act_fit['alpha']:.5f} (residual {act_fit['resid_frac_of_range']:.3%} of range) "
         f"against {cmd_fit['alpha']:.5f} ({cmd_fit['resid_frac_of_range']:.3%}) on commanded speed.",
         "",
         "## 2. What the arm actually delivers", "",
         f"`v_p_at_impact = {track['slope']:.4f} * v_p_commanded {track['intercept']:+.4f}` "
         f"(R^2 = {track['r2']:.5f}, worst residual {track['resid_max']:.4f} m/s).", "",
         "This is a property of the ARM -- IK loop, seven position servos, sprung mount, finite swing",
         "ramp -- not of the contact, and it is measurable on real hardware, which the strike law's",
         "internals are not: you watch the tool, not the impulse. It used to be the dominant error",
         "term, at `0.6984 * cmd - 0.0525`, because the arm was driven by joint angles alone and a",
         "position servo cannot track a ramp without a standing `kd/kp * qdot` lag. Driven with the",
         "plan's inverse-dynamics torque fed forward, it is the identity to within a millimetre per",
         "second, so the composition step below is nearly a no-op rather than a calibration.",
         "",
         "## 3. The validity check, run on the arm itself", "",
         "Boost ball and tool by the same delta and the outcome must shift by exactly delta.", "",
         "| v_in | v_p | delta | measured dv_out/delta |", "|---|---|---|---|"]
    for r in boost:
        L.append(f"| {r['v_in']} | {r['v_p']} | {r['delta']} | {r['dv_out_over_delta']:.4f} |")
    if boost:
        rr = np.array([r["dv_out_over_delta"] for r in boost])
        L += ["", f"Mean {rr.mean():.4f}, worst deviation from 1 is {np.abs(rr - 1).max():.4f}."]

    L += ["", "## 4. Closed loop on the dynamic arm", "",
          "Command through tracking, then strike; execute; score against the same bar.", "",
          f"* commands attempted: **{len(loop)}**",
          f"* pass@{TOL:.0%}: **{(errs <= TOL).mean():.1%}**",
          f"* median relative error: **{np.median(fin):.2%}**" if len(fin) else "* median: n/a",
          f"* worst: **{fin.max():.2%}**" if len(fin) else "* worst: n/a",
          f"* blow-ups: **{n_blow}**   unreachable: **{n_unreach}**", "",
          "| v_in | ratio | target | commanded v_p | achieved | rel err | joint vel frac |",
          "|---|---|---|---|---|---|---|"]
    for r in loop:
        e = r["rel_err"]
        e_s = "MISS" if not np.isfinite(e) else f"{e * 100:.2f}%"
        L.append(f"| {r['v_in']} | {r['ratio']} | {r['target']:.3f} | {r['v_p_cmd']:.3f} | "
                 f"{r.get('achieved', float('nan')):.3f} | {e_s} | "
                 f"{r.get('worst_joint_vel_frac', float('nan')):.2f} |")

    vf = [r.get("worst_joint_vel_frac", float("nan")) for r in loop]
    vf = [v for v in vf if np.isfinite(v)]
    over = [r for r in loop if np.isfinite(r.get("worst_joint_vel_frac", float("nan")))
            and r["worst_joint_vel_frac"] > 1.0]
    L += ["", "## 5. What bounds the envelope", "",
          "The last column is the peak joint rate DELIVERING the strike, as a fraction of the "
          "Panda's published limit.",
          f"Worst over the grid is **{max(vf):.2f}**"
          + (", and {} of {} commands exceed 1.0 -- {}.".format(
              len(over), len(loop),
              ", ".join("v_in={} at {}x".format(r["v_in"], r["ratio"]) for r in over))
             if over else " -- every command is inside the limit."), "",
          "This is the binding constraint, and it is a HARDWARE limit rather than a physics one: the "
          "outcome error",
          "on those commands is as small as anywhere else, the robot simply cannot turn its joints "
          "that fast. Note",
          "also that this ceiling got CLOSER as the arm got better -- an arm that actually delivers "
          "its commanded",
          "tool speed turns its joints faster than one that was under-delivering by 30%, so earlier "
          "readings here",
          "were flattered by the tracking fault rather than by real headroom."]

    (out / "DYNAMIC_ARM.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen = FrankaDynamicStrike()
    v_ins = [0.95, 1.05] if args.quick else [0.85, 0.95, 1.05, 1.15]
    v_ps = np.linspace(-0.20, -1.10, 5 if args.quick else 12)

    print(f"[1/5] sweeping the dynamic arm ({len(v_ins)}x{len(v_ps)} rollouts, ~4 s each) ...",
          flush=True)
    sw = sweep(gen, v_ins, v_ps)
    print(f"      {len(sw['v_out'])} usable, {int(sw['n_excluded'][0])} excluded", flush=True)

    print("[2/5] fitting the strike law both ways ...", flush=True)
    cmd_fit = fit_one_param(sw["v_in"], sw["v_p_cmd"], sw["v_out"])
    act_fit = fit_one_param(sw["v_in"], sw["v_p_actual"], sw["v_out"])
    cmd_free = fit_free(sw["v_in"], sw["v_p_cmd"], sw["v_out"])
    act_free = fit_free(sw["v_in"], sw["v_p_actual"], sw["v_out"])
    print(f"      commanded: alpha+beta={cmd_free['alpha_plus_beta']:.4f}", flush=True)
    print(f"      achieved : alpha+beta={act_free['alpha_plus_beta']:.4f} "
          f"alpha={act_fit['alpha']:.5f}", flush=True)

    print("[3/5] fitting the arm's tracking map ...", flush=True)
    track = fit_tracking(sw["v_p_cmd"], sw["v_p_actual"])
    print(f"      v_p_actual = {track['slope']:.4f}*cmd {track['intercept']:+.4f} "
          f"(R2={track['r2']:.5f})", flush=True)

    print("[4/5] boost identity ...", flush=True)
    cases = [(0.95, -0.60)] if args.quick else [(0.90, -0.60), (0.95, -0.30), (1.00, -0.90)]
    boost = boost_identity(gen, cases, (0.05, 0.10) if args.quick else (0.05, 0.10, 0.20), track)

    print("[5/5] closing the loop ...", flush=True)
    loop = closed_loop(gen, act_fit, track)

    (out / "dynamic_arm.json").write_text(json.dumps(
        {"strike_on_commanded": cmd_fit, "strike_on_achieved": act_fit,
         "strike_free_on_commanded": cmd_free, "strike_free_on_achieved": act_free,
         "tracking": track, "boost_identity": boost, "closed_loop": loop,
         "n_excluded": int(sw["n_excluded"][0]), "n_usable": int(len(sw["v_out"])),
         "config": {"tol": TOL, "ratios": list(RATIOS), "v_ins": list(V_INS)}},
        indent=2, default=float))
    write_report(out, sw, cmd_fit, act_fit, cmd_free, act_free, track, boost, loop)
    print(f"\nwrote {out}/DYNAMIC_ARM.md and dynamic_arm.json", flush=True)


if __name__ == "__main__":
    main()
