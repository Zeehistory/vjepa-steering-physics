"""Can a real Franka Panda execute this strike? Feasibility frontier over (v_in, target v_out).

This is the hardware go/no-go. The controllability certificate
(``experiments/pipeline/01_data/validate_paddle_strike.py``) proves the *task* is well posed with an idealised striker.
This script asks the separate question of whether a 7-DOF Panda, driven by its own position servos
with its real link inertia, can actually deliver the required tool speed without exceeding the
robot's published joint velocity and torque limits.

Method:

1. Sweep the dynamic arm's action -> outcome map over a grid of (incoming ball speed, commanded tool
   speed), recording for every rollout the achieved outgoing speed, whether the contact was clean,
   and peak joint velocity / torque as fractions of the Panda's limits (measured up to contact,
   since delivering the strike is what has to be within limits -- the follow-through is not).
2. Read the **frontier** straight off that sweep: per incoming speed, the fastest outgoing ball
   produced by a rollout that was both clean and inside the robot's limits.
3. Fit the inverse model on the feasible rollouts ONLY, then command a few targets back through it
   to confirm the arm hits them.

Step 3's filter is not cosmetic. Fitting on the raw sweep gives a meaningless model: outside its
feasible envelope the arm saturates and the contact degrades, and those points dominate the
regression (alpha 0.095 and 86%-of-range residual, versus a true alpha near 1.9). The arm can only
be commanded where the arm actually works.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/07_eval/franka_feasibility.py --output_dir outputs/franka_feas
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.control.strike_inverse import StrikeInverse
from src.data import franka_dynamic_strike as fd
from src.data import paddle_strike as ps
from src.data.franka_dynamic_strike import FrankaDynamicStrike

TOL = 0.05          # same relative-error tolerance as the controllability certificate
LIMIT = 1.0         # fraction of a published Panda limit that counts as feasible


def _rollout(gen: FrankaDynamicStrike, v_in: float, v_p: float) -> dict | None:
    try:
        r = gen.simulate(v_in, v_p, render=False)
    except RuntimeError:
        return None
    f = gen.feasibility()
    # ENERGY BOUND. An elastic bounce off an infinitely heavy striker moving at v_tool toward a ball
    # closing at v_in cannot exceed v_in + 2*v_tool. Anything above that is the solver injecting
    # energy, not physics. This gate is essential and not implied by the others: an explosive
    # rebound is still a single touch at a perfectly constant post-contact speed, so it sails
    # through the cleanliness checks. Without it the frontier reported 39 m/s off a 0.61 m/s tool.
    bound = abs(r["v_in_world"]) + 2.0 * abs(r["v_p_actual"]) + 0.05
    physical = bool(abs(r["v_out_world"]) <= bound)
    clean = bool(r["n_touches"] == 1 and r["v_out_world_std"] < 1e-3
                 and r["v_out_lateral_ratio"] < 0.10 and physical)
    within = bool(f["worst_joint_vel_frac"] <= LIMIT and f["worst_joint_torque_frac"] <= LIMIT)
    return {
        "v_in": float(v_in), "v_p": float(v_p), "v_out": float(r["v_out_world"]),
        "tool_speed": f["peak_tool_speed_mps"],
        "joint_vel_frac": f["worst_joint_vel_frac"],
        "joint_torque_frac": f["worst_joint_torque_frac"],
        "mount_mm": 1000 * f["peak_mount_deflection_m"],
        "lateral_pct": 100 * r["v_out_lateral_ratio"],
        "n_touches": int(r["n_touches"]), "post_std": r["v_out_world_std"],
        "physical": physical, "energy_bound": bound,
        "clean": clean, "within_limits": within, "feasible": bool(clean and within),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--v_in", default="0.85,1.0,1.25,1.5,2.0")
    ap.add_argument("--n_vp", type=int, default=16)
    ap.add_argument("--vp_lo", type=float, default=0.1)
    ap.add_argument("--vp_hi", type=float, default=-1.5)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gen = FrankaDynamicStrike()

    v_in_grid = [float(x) for x in args.v_in.split(",")]
    v_p_grid = np.linspace(args.vp_lo, args.vp_hi, args.n_vp)

    # ---- 1. sweep --------------------------------------------------------------------------------
    print(f"[1/3] sweeping {len(v_in_grid)}x{len(v_p_grid)} rollouts ...", flush=True)
    rows = []
    for v_in in v_in_grid:
        for v_p in v_p_grid:
            r = _rollout(gen, v_in, float(v_p))
            if r is not None:
                rows.append(r)
        ok = [r for r in rows if r["v_in"] == v_in and r["feasible"]]
        print(f"      v_in={v_in:.2f}: {len(ok)} feasible of {len(v_p_grid)}"
              + (f", best |v_out|={max(abs(r['v_out']) for r in ok):.3f} m/s" if ok else ""),
              flush=True)

    # ---- 2. frontier -----------------------------------------------------------------------------
    frontier = {}
    for v_in in v_in_grid:
        ok = [r for r in rows if r["v_in"] == v_in and r["feasible"]]
        if ok:
            best = max(ok, key=lambda r: abs(r["v_out"]))
            frontier[f"{v_in:.2f}"] = {
                "max_v_out": abs(best["v_out"]), "ratio": abs(best["v_out"]) / v_in,
                "v_p": best["v_p"], "tool_speed": best["tool_speed"],
                "joint_vel_frac": best["joint_vel_frac"],
                "joint_torque_frac": best["joint_torque_frac"],
            }
        else:
            frontier[f"{v_in:.2f}"] = None

    # ---- 3. fit the inverse on FEASIBLE rollouts, then command targets back through it -----------
    feas = [r for r in rows if r["feasible"]]
    verify = []
    inv_d = None
    if len(feas) >= 6:
        sweep = {"v_in": np.array([r["v_in"] for r in feas]),
                 "v_p": np.array([r["v_p"] for r in feas]),
                 "v_out": np.array([r["v_out"] for r in feas])}
        inv = StrikeInverse.fit(sweep, ps.BALL_MASS, ps.PADDLE_MASS)
        inv.save(out / "franka_dynamic_inverse.json")
        inv_d = inv.to_dict()
        print(f"[3/3] inverse fit on {len(feas)} feasible rollouts: alpha={inv.alpha:.4f} "
              f"beta={inv.beta:.4f} residual={100 * inv.forward_max_resid / inv.forward_range:.2f}%",
              flush=True)
        for v_in in v_in_grid:
            fr = frontier[f"{v_in:.2f}"]
            if fr is None:
                continue
            for frac in (0.6, 0.8, 1.0):
                tgt = fr["max_v_out"] * frac
                v_p = float(np.asarray(inv.action_for(v_in, -tgt, mode="linear")).ravel()[0])
                r = _rollout(gen, v_in, v_p)
                if r is None:
                    continue
                rel = abs(abs(r["v_out"]) - tgt) / tgt
                verify.append({"v_in": v_in, "target": tgt, "achieved": abs(r["v_out"]),
                               "rel_err": rel, "joint_vel_frac": r["joint_vel_frac"],
                               "joint_torque_frac": r["joint_torque_frac"],
                               "ok": bool(rel <= TOL and r["feasible"])})
                print(f"      v_in={v_in:.2f} target={tgt:.2f} -> {abs(r['v_out']):.3f} "
                      f"({rel:.1%}), joints {r['joint_vel_frac']:.0%}/"
                      f"{r['joint_torque_frac']:.0%}{'  OK' if verify[-1]['ok'] else ''}", flush=True)

    report = {
        "tolerance": TOL, "limit_fraction": LIMIT,
        "arm_base": list(map(float, fd.ARM_BASE)),
        "joint_vel_limit": fd.JOINT_VEL_LIMIT.tolist(),
        "joint_torque_limit": fd.JOINT_TORQUE_LIMIT.tolist(),
        "frontier": frontier, "inverse": inv_d, "verify": verify, "sweep": rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "feasibility.json").write_text(json.dumps(report, indent=2))

    lines = ["# Franka feasibility: can the arm actually deliver the strike?", "",
             "A rollout counts as feasible when the contact is clean (one touch, constant "
             "post-contact speed, <10% lateral, and outgoing speed inside the elastic energy bound "
             "v_in + 2*v_tool) AND peak joint velocity and torque are both within the Panda's "
             "published limits, measured up to the moment of contact.", "",
             "## Frontier: fastest outgoing ball the arm can produce, within limits", "",
             "| v_in (m/s) | max \\|v_out\\| | ratio | tool speed | joint vel | joint torque |",
             "|---|---|---|---|---|---|"]
    for k, v in frontier.items():
        if v is None:
            lines.append(f"| {k} | none | - | - | - | - |")
        else:
            lines.append(f"| {k} | {v['max_v_out']:.2f} | {v['ratio']:.2f}x | "
                         f"{v['tool_speed']:.2f} m/s | {v['joint_vel_frac']:.0%} | "
                         f"{v['joint_torque_frac']:.0%} |")
    if verify:
        lines += ["", "## Commanding targets back through the fitted inverse", "",
                  "| v_in | target | achieved | err | joint vel | verdict |", "|---|---|---|---|---|---|"]
        for v in verify:
            lines.append(f"| {v['v_in']:.2f} | {v['target']:.2f} | {v['achieved']:.3f} | "
                         f"{v['rel_err']:.1%} | {v['joint_vel_frac']:.0%} | "
                         f"{'OK' if v['ok'] else 'FAIL'} |")
    (out / "feasibility.md").write_text("\n".join(lines) + "\n")

    print(f"\nwrote {out}/feasibility.md  ({report['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
