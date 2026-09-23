#!/usr/bin/env python
"""Dump the frame stacks behind a robotics Real-vs-Steered filmstrip.

The counterfactual is the one the controllability certificate scores: the rig is asked for an
outcome speed, and we compare

  unsteered   the ONE fixed swing the rig always runs, whatever was requested
  steered     the swing the inverse model picks for THIS request

Both rollouts are rendered at the SAME wall-clock instants. The ball arrives on a fixed schedule and
the swing runs on a fixed schedule, so the same instant is the same phase of the same event in both
rows -- that is what makes the comparison honest rather than two flattering picks.

Unlike the latent-quantity filmstrips, nothing here is decoded: these are MuJoCo renders of an
executed action, and the speed annotated on each row is ``v_out_world`` -- the same world-state
measurement the certificate scores, never a number read off pixels.

    PYTHONPATH=. MUJOCO_GL=egl python experiments/threads/paddle-robotics/08_figures/dump_robotics_filmstrip.py \
        --output_dir OUT --embodiment paddle --ratio 0.5 --n_instants 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS, build_striker


def _instants(t: np.ndarray, c0: float, c1: float, lead: float, n: int) -> list[float]:
    """``n`` shared sample times spanning approach -> contact -> return.

    Contact is pinned to the midpoint of the interval MuJoCo reports rather than left to whichever
    frame lands nearest, so the strike is always actually in the strip; the rest are spread evenly
    before and after it.
    """
    t0, t1 = float(t.min()), float(t.max())
    tc = 0.5 * (c0 + c1)
    pre = max(t0, c0 - lead)
    n_pre = max(1, (n - 1) // 2)
    n_post = n - 1 - n_pre
    before = list(np.linspace(t0, pre, n_pre, endpoint=True))
    after = list(np.linspace(tc, t1, n_post + 1, endpoint=True))[1:]
    return before + [tc] + after


def _nearest(t: np.ndarray, want: float) -> int:
    return int(np.abs(t - want).argmin())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--v_in", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="requested outcome as a multiple of the incoming speed")
    ap.add_argument("--nominal_ratio", type=float, default=1.0,
                    help="the ONE swing the unsteered rig always runs, as an outcome ratio")
    ap.add_argument("--n_instants", type=int, default=6)
    ap.add_argument("--gap_slowdown", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--lead_frames", type=float, default=3.0)
    ap.add_argument("--wide", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    gen = build_striker(args.embodiment, image_size=args.image_size)
    if args.wide:
        if not hasattr(gen, "use_demo_camera"):
            raise SystemExit(f"--wide: embodiment '{args.embodiment}' has no wide demo camera")
        gen.use_demo_camera(True)

    inv = StrikeInverse.fit(sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15],
                                          np.linspace(0.25, -1.45, 20)),
                            BALL_MASS, PADDLE_MASS)

    def action(target: float) -> float:
        return float(np.asarray(inv.action_for(args.v_in, target, mode="quadratic")).ravel()[0])

    target = -args.ratio * args.v_in
    v_p_nom = action(-args.nominal_ratio * args.v_in)
    v_p_steer = action(target)

    sims = {}
    for tag, v_p in (("unsteered", v_p_nom), ("steered", v_p_steer)):
        s = gen.simulate(args.v_in, v_p, render=True, render_all=True,
                         gap_slowdown=args.gap_slowdown)
        sims[tag] = {"frames": np.asarray(s["frames_continuous"]),
                     "t": np.asarray(s["t_continuous"]),
                     "c0": float(s["contact_frame"]), "c1": float(s["contact_end_frame"]),
                     "v_out": float(s["v_out_world"]), "v_p": float(v_p)}

    u = sims["unsteered"]
    want = _instants(u["t"], u["c0"], u["c1"], args.lead_frames, args.n_instants)

    stacks, meta = {}, {}
    for tag, s in sims.items():
        idx = [_nearest(s["t"], w) for w in want]
        fr = s["frames"][idx]                                   # (n, H, W, C) uint8 or float
        if fr.dtype != np.uint8:
            fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
        stacks[tag] = fr
        meta[tag] = {"v_out_world": s["v_out"], "v_p": s["v_p"],
                     "t_sampled": [float(s["t"][i]) for i in idx],
                     "rel_err_vs_target": abs(s["v_out"] - target) / abs(target)}

    np.savez_compressed(out / f"{args.embodiment}_ratio{args.ratio:g}.npz", **stacks)
    (out / f"{args.embodiment}_ratio{args.ratio:g}.json").write_text(json.dumps({
        "embodiment": args.embodiment, "v_in": args.v_in, "ratio": args.ratio,
        "target_v_out": target, "nominal_ratio": args.nominal_ratio,
        "instants_requested": [float(w) for w in want],
        "contact_window": [u["c0"], u["c1"]],
        "note": "speeds are v_out_world (world state), never read off pixels",
        "rows": meta}, indent=1))
    print(f"[robo-filmstrip] {args.embodiment} ratio={args.ratio:g} target={target:.3f} "
          f"unsteered={meta['unsteered']['v_out_world']:.3f} steered={meta['steered']['v_out_world']:.3f} "
          f"-> {out}")


if __name__ == "__main__":
    main()
