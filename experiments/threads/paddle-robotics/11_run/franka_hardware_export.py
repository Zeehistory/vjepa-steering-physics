"""Export strike trajectories in a form a real Franka Panda can execute, plus videos.

For each requested (incoming ball speed, target outgoing speed) this writes:

* ``<case>/trajectory.csv`` -- t, q1..q7, qd1..qd7 sampled at the controller rate (1 kHz default).
  This is the same joint reference the simulator feeds its servos, so the trajectory that gets
  shipped is the trajectory that was measured.
* ``<case>/meta.json`` -- the command, the simulated outcome, peak joint velocity and torque as
  fractions of the Panda's published limits, and the scene geometry the trajectory assumes.
* ``<case>/strike.mp4`` -- the continuous episode, including the swing gap the dataset never
  renders, so a human can actually watch the strike.

It also writes ``HARDWARE.md``, the protocol for taking this to a real robot. Read that before
running anything: the trajectory alone is not sufficient, because the strike law has to be
re-identified on the physical setup.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/11_run/franka_hardware_export.py \\
        --output_dir /path/to/out --cases 1.0:1.5,1.0:2.0,2.0:2.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.control.strike_inverse import StrikeInverse
from src.data import franka_dynamic_strike as fd
from src.data import paddle_strike as ps
from src.data.franka_dynamic_strike import FrankaDynamicStrike
from src.utils.video_io import save_video

CAL_V_IN = (0.85, 1.0, 1.25, 1.5, 2.0)
CAL_V_P = np.linspace(0.1, -1.1, 12)


def _calibrate(gen: FrankaDynamicStrike) -> StrikeInverse:
    """Fit the arm's own strike law on rollouts that are clean AND inside the robot's limits.

    Fitting on the raw sweep is meaningless: outside its feasible envelope the arm saturates and the
    contact degrades, and those points dominate the regression.
    """
    v_in, v_p, v_out = [], [], []
    for a in CAL_V_IN:
        for b in CAL_V_P:
            try:
                r = gen.simulate(float(a), float(b), render=False)
            except RuntimeError:
                continue
            f = gen.feasibility()
            bound = abs(r["v_in_world"]) + 2.0 * abs(r["v_p_actual"]) + 0.05
            if (r["n_touches"] == 1 and r["v_out_world_std"] < 1e-3
                    and abs(r["v_out_world"]) <= bound
                    and f["worst_joint_vel_frac"] <= 1.0 and f["worst_joint_torque_frac"] <= 1.0):
                v_in.append(r["v_in_world"]); v_p.append(float(b)); v_out.append(r["v_out_world"])
    print(f"      calibrated on {len(v_out)} feasible rollouts", flush=True)
    data = {"v_in": np.array(v_in), "v_p": np.array(v_p), "v_out": np.array(v_out)}
    # CONSTRAINED fit (alpha+beta=1, gamma=0). The feasibility filter above is not sufficient on its
    # own: even after it, the free three-parameter fit returned alpha=1.5725, beta=-0.9132, so
    # alpha+beta=0.659 -- a law claiming two bodies moving together at 1 m/s emerge at 0.659 m/s. That
    # is not a marginal fit, it is an impossible one, and it was being published in HARDWARE.md as the
    # starting point for a real robot. The residual is printed so a bad sweep is visible rather than
    # absorbed into confident wrong coefficients.
    inv = StrikeInverse.fit_constrained(data, ps.BALL_MASS, ps.PADDLE_MASS)
    free = StrikeInverse.fit(data, ps.BALL_MASS, ps.PADDLE_MASS)
    print(f"      constrained: alpha={inv.alpha:.4f} beta={inv.beta:.4f} "
          f"(a+b={inv.alpha + inv.beta:.4f}) resid={100 * inv.forward_max_resid / inv.forward_range:.2f}%"
          f" of range", flush=True)
    print(f"      unconstrained would have been: alpha={free.alpha:.4f} beta={free.beta:.4f} "
          f"(a+b={free.alpha + free.beta:.4f}) <- rejected if != 1", flush=True)
    return inv


HARDWARE_MD = """# Taking the paddle strike to a real Franka Panda

These trajectories are executable, but they are **not** a drop-in: the strike law depends on the
physical ball, tool and surface, and has to be re-identified on your setup. The protocol below is
the same one the simulator uses, run against hardware instead.

## 1. What the trajectory assumes

| | value |
|---|---|
| arm base, relative to the table centre | `x = {bx:.2f} m, y = {by:.2f} m, z = {bz:.2f} m` (on a plinth, facing the table) |
| tool | sphere, radius {mr:.3f} m, mass {mm:.1f} kg |
| tool mount | prismatic along the strike axis, spring {ms:.0f} N/m, damping {md:.0f} N.s/m, travel +-{mrg:.3f} m |
| ball | radius {br:.3f} m, mass {bm:.3f} kg, sliding on a low-friction surface |
| strike axis | world x; ball travels +x, tool moves -x, ball returns -x |
| controller | joint position, {rate} Hz |

The base orientation matters: the arm must **face the table**. Mounted the other way round, the
strike point sits behind the base and joint1 needs about 180 deg against a +-166 deg limit.

**The compliant mount is not optional.** It is what separates the ~2 ms impact from the ~100 ms
servo loop. With the tool bolted rigidly to the flange the simulated contact was unusable, and on
hardware a rigid tool puts the whole impulse straight into the joints.

## 2. Re-identify the strike law before trusting any command

The law is linear in the tool speed,

    v_out = alpha * v_tool + beta * v_in

with `alpha` set by the tool/ball mass ratio and restitution, and `beta` by restitution. In
simulation `alpha = {alpha:.3f}`, `beta = {beta:.3f}`. **Do not assume these transfer.** Your ball,
tool face and surface will give different numbers. Instead:

1. Roll the ball in at a repeatable speed and measure it (`v_in`).
2. Execute strikes across a range of tool speeds, spanning what you intend to command.
3. Measure `v_out` for each.
4. Least-squares fit `alpha`, `beta`. `src/control/strike_inverse.py::StrikeInverse.fit` does exactly
   this and is not simulator-specific -- feed it measured triples.
5. Command through the fitted inverse, and re-measure. In simulation this closes to ~1-2%.

## 3. Safety and limits

* Peak demands are in each case's `meta.json` as fractions of the Panda's published limits. **Joint
  velocity is the binding constraint, not torque** -- torque peaks around 38% while velocity reaches
  ~90% at the top of the envelope. Start with the low-speed cases.
* Everything here assumes free space around the swing. There is no obstacle avoidance and no
  collision checking against anything but the ball.
* The trajectory includes a follow-through that brakes the tool to rest over
  {brake:.2f} s after contact. Do not truncate it at contact, or the arm stops abruptly at speed.

## 4. What is still missing for a closed loop

The trajectories are open loop and assume the ball arrives at a known place at a known time. A real
run needs:

* **Ball sensing** -- the simulator reads position from ground truth. You need a tracker giving
  `v_in` and an arrival estimate.
* **A trigger** -- the swing must start at a fixed time before the ball reaches the strike point.
  Timing error maps directly to contact-point error.
* **The latent side of the loop is not here.** These trajectories come from the analytic inverse.
  Selecting the action from a V-JEPA prediction is the next stage and runs on top of this same interface:
  it only changes how the target speed is chosen, not how it is executed.

## 5. Known limitation of the simulation

The actuated-arm sim still shows occasional energy injection at contact (about 1 command in 15,
converging only slowly as the timestep shrinks). The envelope in `feasibility.md` is a good estimate
of what the robot can do, not a certified bound. The *dataset* variants do not have this problem --
they use an idealised striker and are certified 13/13.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cases", default="1.0:1.25,1.0:1.6,1.0:2.0,1.5:2.2,2.0:2.6",
                    help="comma-separated v_in:target_v_out pairs, m/s")
    ap.add_argument("--rate_hz", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = FrankaDynamicStrike()

    print("[1/2] calibrating the arm's strike law ...", flush=True)
    inv = _calibrate(gen)
    inv.save(out / "strike_law.json")
    print(f"      alpha={inv.alpha:.4f} beta={inv.beta:.4f}", flush=True)

    print("[2/2] exporting cases ...", flush=True)
    index = []
    for case in args.cases.split(","):
        v_in, v_tgt = (float(x) for x in case.split(":"))
        v_p = float(np.asarray(inv.action_for(v_in, -v_tgt, mode="linear")).ravel()[0])

        r = gen.simulate(v_in, v_p, render=True, render_all=True)
        f = gen.feasibility()
        traj = gen.plan_joint_trajectory(v_p, rate_hz=args.rate_hz)

        name = f"vin{v_in:.2f}_vout{v_tgt:.2f}".replace(".", "p")
        cdir = out / name
        cdir.mkdir(exist_ok=True)

        hdr = "t," + ",".join(f"q{i}" for i in range(1, 8)) + "," + \
              ",".join(f"qd{i}" for i in range(1, 8))
        np.savetxt(cdir / "trajectory.csv",
                   np.column_stack([traj["t"], traj["q"], traj["qd"]]),
                   delimiter=",", header=hdr, comments="", fmt="%.9g")

        achieved = abs(r["v_out_world"])
        meta = {
            "v_in_mps": v_in, "target_v_out_mps": v_tgt,
            "commanded_tool_speed_mps": v_p,
            "simulated_v_out_mps": achieved,
            "relative_error": abs(achieved - v_tgt) / v_tgt,
            "contact_time_s": traj["contact_time_s"],
            "control_rate_hz": args.rate_hz,
            "n_samples": int(len(traj["t"])),
            "peak_joint_vel_frac_of_limit": traj["peak_joint_vel_frac"].tolist(),
            "worst_joint_vel_frac_sim": f["worst_joint_vel_frac"],
            "worst_joint_torque_frac_sim": f["worst_joint_torque_frac"],
            "joint_vel_limit_rad_s": fd.JOINT_VEL_LIMIT.tolist(),
            "joint_torque_limit_nm": fd.JOINT_TORQUE_LIMIT.tolist(),
            "arm_base_xyz": list(map(float, fd.ARM_BASE)),
            "tool": {"radius_m": fd.MALLET_R, "mass_kg": fd.MALLET_MASS,
                     "reach_m": fd.MALLET_REACH,
                     "mount_stiffness_N_per_m": fd.MOUNT_STIFFNESS,
                     "mount_damping_Ns_per_m": fd.MOUNT_DAMPING,
                     "mount_travel_m": fd.MOUNT_RANGE},
            "ball": {"radius_m": ps.BALL_R, "mass_kg": ps.BALL_MASS},
            "strike_law_sim": {"alpha": inv.alpha, "beta": inv.beta},
            "caveat": "simulated outcome; re-identify alpha/beta on hardware (see HARDWARE.md)",
        }
        (cdir / "meta.json").write_text(json.dumps(meta, indent=2))

        clip = torch.from_numpy(r["frames_continuous"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        save_video(clip, cdir / "strike.mp4", fps=12)

        index.append({"case": name, **{k: meta[k] for k in
                                       ("v_in_mps", "target_v_out_mps", "simulated_v_out_mps",
                                        "relative_error", "worst_joint_vel_frac_sim",
                                        "worst_joint_torque_frac_sim")}})
        print(f"      {name}: target {v_tgt:.2f} -> sim {achieved:.3f} "
              f"({meta['relative_error']:.1%}), joints {f['worst_joint_vel_frac']:.0%} vel / "
              f"{f['worst_joint_torque_frac']:.0%} torque", flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=2))
    (out / "HARDWARE.md").write_text(HARDWARE_MD.format(
        bx=fd.ARM_BASE[0], by=fd.ARM_BASE[1], bz=fd.ARM_BASE[2],
        mr=fd.MALLET_R, mm=fd.MALLET_MASS, ms=fd.MOUNT_STIFFNESS, md=fd.MOUNT_DAMPING,
        mrg=fd.MOUNT_RANGE, br=ps.BALL_R, bm=ps.BALL_MASS, rate=args.rate_hz,
        alpha=inv.alpha, beta=inv.beta, brake=fd.BRAKE_S))
    print(f"\nwrote {out}/  ({len(index)} cases + HARDWARE.md)")


if __name__ == "__main__":
    main()
