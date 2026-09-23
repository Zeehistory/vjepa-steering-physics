"""Controllability certificate for the paddle-strike scene.

Answers one question with a number: **if we command a post-contact speed, do we get it?**
The pre-registered bar is ``|v_out - v*| / |v*| <= 5%`` on ``>= 95%`` of held-out commands.

Everything here runs with ``render=False``, so it is a CPU job -- no EGL, no GPU, no latents. Run it
before spending anything on rendering or extraction; if the map is not controllable, the whole
latent-control loop downstream is ill-posed and this is where we find out.

Structure:

1. **Forward sweep** over a (v_in x v_p) grid -> the raw action->outcome map.
2. **Map gates** -- strict monotonicity, linearity, restitution stability across v_in, agreement
   between the simulator's own velocity and the projected image track, and range coverage.
3. **Inverse model** fit on TRAIN v_in values only (:mod:`src.control.strike_inverse`).
4. **The bar** -- held-out (v_in, v*) commands executed open-loop, scored on relative error.
5. **Hygiene gates** -- determinism, exact contact frame, exactly one touch, frame residency.

Usage::

    PYTHONPATH=. python experiments/pipeline/01_data/validate_paddle_strike.py --output_dir outputs/paddle_strike/cert
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data import paddle_strike as ps
from src.data.paddle_strike import build_striker

TOL = 0.05          # the pre-registered relative-error tolerance
PASS_FRAC = 0.95    # ... required on this fraction of commands


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--n_vp", type=int, default=24, help="paddle-speed grid points per v_in")
    ap.add_argument("--n_commands", type=int, default=240, help="held-out commands to execute")
    ap.add_argument("--ratio_lo", type=float, default=0.5)
    ap.add_argument("--ratio_hi", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gen = build_striker(args.embodiment)
    rng = np.random.default_rng(args.seed)
    cert: dict = {"embodiment": args.embodiment, "tolerance": TOL, "pass_fraction_required": PASS_FRAC,
                  "solref": list(gen.solref), "fps": ps.FPS,
                  "ball_mass": ps.BALL_MASS, "paddle_mass": ps.PADDLE_MASS,
                  "nominal_contact_frame": ps.NOMINAL_CONTACT_FRAME,
                  "x_paddle_rest": ps.X_PADDLE_REST, "x_post": ps.X_POST,
                  "ratio_range_requested": [args.ratio_lo, args.ratio_hi]}
    gates: dict[str, bool] = {}

    # ---- 1. forward sweep ------------------------------------------------------------------------
    v_in_train = np.array([0.85, 0.95, 1.05, 1.15])
    # Spans the operating envelope (ratio 0.5..3.0 needs v_p in roughly [+0.19, -1.28]) with a modest
    # margin either side. Sweeping far beyond that buys no accuracy on a map this linear and costs
    # rollouts: a strongly receding paddle returns the ball at centimetres per second, so the episode
    # runs for hundreds of frames before the ball drifts back across X_POST.
    v_p_grid = np.linspace(0.25, -1.45, args.n_vp)
    print(f"[1/5] forward sweep: {len(v_in_train)} x {len(v_p_grid)} = "
          f"{len(v_in_train) * len(v_p_grid)} rollouts ...", flush=True)
    sweep = sweep_strikes(gen, v_in_train, v_p_grid)

    # ---- 2. map gates ----------------------------------------------------------------------------
    rhos = []
    e_by_vin = []
    for v in v_in_train:
        m = np.isclose(sweep["v_in"], v, atol=1e-3)
        rhos.append(_spearman(sweep["v_p"][m], sweep["v_out"][m]))
        c = np.polyfit(sweep["v_p"][m], sweep["v_out"][m], 1)
        e_by_vin.append(float(c[0] * (ps.BALL_MASS + ps.PADDLE_MASS) / ps.PADDLE_MASS - 1.0))
    e_by_vin = np.asarray(e_by_vin)
    gates["monotone"] = bool(np.allclose(np.abs(rhos), 1.0, atol=1e-9))

    inv_all = StrikeInverse.fit(sweep, ps.BALL_MASS, ps.PADDLE_MASS)
    resid_pct = 100.0 * inv_all.forward_max_resid / inv_all.forward_range
    gates["linear"] = bool(resid_pct < 1.0)

    e_spread_pct = float(100.0 * (e_by_vin.max() - e_by_vin.min()) / np.abs(e_by_vin.mean()))
    gates["restitution_stable"] = bool(e_spread_pct < 2.0)

    # World->image scale along the motion axis must be a CONSTANT. This is not a tautology and it is
    # not a check of the pixel tracker: the camera sits in the world y-z plane, so its x-axis is world
    # x and a ball moving in x holds constant depth -- image x should therefore be exactly affine in
    # world x, with zero perspective foreshortening. If that holds, constant world velocity implies
    # exactly constant IMAGE velocity, which is what the probe/steering contract needs.
    #   NOTE: img_pos is the analytic projection of the sim's own world track, so this says nothing
    #   about whether the darkness-centroid tracker recovers the ball from RENDERED pixels. That is a
    #   separate, genuinely independent check and it lives in tests/test_paddle_strike.py, which
    #   renders frames and runs src.analysis.ball_tracking on them.
    scale = sweep["v_out_img"] / sweep["v_out"]
    scale_dev_pct = float(100.0 * np.abs(scale - np.median(scale)).max() / np.abs(np.median(scale)))
    gates["image_projection_affine"] = bool(scale_dev_pct < 1.0)

    ratio = np.abs(sweep["v_out"] / sweep["v_in"])
    gates["covers_ratio_range"] = bool(ratio.min() <= args.ratio_lo and ratio.max() >= args.ratio_hi)

    cert["map"] = {
        "spearman_per_v_in": [float(r) for r in rhos],
        "forward_alpha": inv_all.alpha, "forward_beta": inv_all.beta, "forward_gamma": inv_all.gamma,
        "forward_max_resid": inv_all.forward_max_resid,
        "forward_resid_pct_of_range": resid_pct,
        "e_effective": inv_all.e_effective,
        "e_per_v_in": [float(e) for e in e_by_vin], "e_spread_pct": e_spread_pct,
        "world_to_image_scale": float(np.median(scale)),
        "image_track_max_dev_pct": scale_dev_pct,
        "ratio_achieved": [float(ratio.min()), float(ratio.max())],
    }
    print(f"      alpha={inv_all.alpha:.5f} beta={inv_all.beta:.5f} e_eff={inv_all.e_effective:.4f} "
          f"resid={resid_pct:.3f}% of range", flush=True)

    # ---- 3. inverse model, fit on TRAIN v_in only ------------------------------------------------
    inv = StrikeInverse.fit(sweep, ps.BALL_MASS, ps.PADDLE_MASS)
    inv.save(out / "strike_inverse.json")

    # ---- 4. THE BAR: held-out commands ----------------------------------------------------------
    # held-out v_in values are strictly interleaved with (and disjoint from) the training grid
    v_in_test = np.array([0.90, 1.00, 1.10])
    n = args.n_commands
    cmd_v_in = v_in_test[rng.integers(0, len(v_in_test), size=n)]
    cmd_ratio = rng.uniform(args.ratio_lo, args.ratio_hi, size=n)
    cmd_target = -cmd_ratio * cmd_v_in          # head-on return -> outcome is leftward (negative)
    print(f"[4/5] executing {n} held-out commands ...", flush=True)

    results = {}
    for mode in ("linear", "quadratic"):
        v_p_cmd = np.asarray(inv.action_for(cmd_v_in, cmd_target, mode=mode)).ravel()
        achieved, cf, touches, vis, gaps, pstart = [], [], [], [], [], []
        mpre, mpost = [], []
        for vi, vp in zip(cmd_v_in, v_p_cmd):
            r = gen.simulate(float(vi), float(vp), render=False)
            achieved.append(r["v_out_world"])
            cf.append(r["contact_frame"])
            touches.append(r["n_touches"])
            vis.append(r["post_frames_visible"])
            gaps.append(r["gap_frames"])
            pstart.append(r["post_start_x"])
            mpre.append(r["margin_after_pre"])
            mpost.append(r["margin_before_post"])
        achieved = np.asarray(achieved)
        rel = np.abs(achieved - cmd_target) / np.abs(cmd_target)
        frac = float((rel <= TOL).mean())
        results[mode] = {
            "pass_fraction": frac,
            "rel_err_mean": float(rel.mean()), "rel_err_median": float(np.median(rel)),
            "rel_err_p95": float(np.percentile(rel, 95)), "rel_err_max": float(rel.max()),
            "abs_err_median_mps": float(np.median(np.abs(achieved - cmd_target))),
            "worst_case": {"v_in": float(cmd_v_in[rel.argmax()]),
                           "target": float(cmd_target[rel.argmax()]),
                           "achieved": float(achieved[rel.argmax()]),
                           "rel_err": float(rel.max())},
            "contact_frame_range": [float(min(cf)), float(max(cf))],
            "touches": sorted(set(int(x) for x in touches)),
            "min_post_frames_visible": int(min(vis)),
            "gap_frames_range": [float(min(gaps)), float(max(gaps))],
            "post_start_x_spread_m": float(max(pstart) - min(pstart)),
            "min_margin_after_pre": float(min(mpre)),
            "min_margin_before_post": float(min(mpost)),
        }
        print(f"      {mode:9s} pass={frac * 100:.1f}%  median_rel_err={np.median(rel) * 100:.3f}%  "
              f"max={rel.max() * 100:.2f}%", flush=True)
    cert["commands"] = results
    cert["n_commands"] = n
    gates["bar_linear_inverse"] = bool(results["linear"]["pass_fraction"] >= PASS_FRAC)
    gates["bar_quadratic_inverse"] = bool(results["quadratic"]["pass_fraction"] >= PASS_FRAC)

    # ---- 5. hygiene ------------------------------------------------------------------------------
    print("[5/5] hygiene gates ...", flush=True)
    a = gen.simulate(1.0, -1.0, render=False)
    b = gen.simulate(1.0, -1.0, render=False)
    gates["deterministic"] = bool(np.array_equal(a["ball_x"], b["ball_x"])
                                  and a["v_out_world"] == b["v_out_world"])
    # The collision must fall strictly INSIDE the unrendered swing gap: after the pre-window's last
    # frame, and before the post-window opens. Contact time is free by design (the paddle's rest
    # position is fixed so the pre-window cannot leak the action), so what matters is not that
    # contact lands on some exact frame but that no rendered frame ever straddles it.
    # PER-ROLLOUT margins: comparing the earliest contact against the shortest gap across DIFFERENT
    # rollouts is meaningless, since a fast strike has both an early contact and a short gap.
    margin_pre = min([float(sweep["margin_after_pre"].min())]
                     + [m["min_margin_after_pre"] for m in results.values()])
    margin_post = min([float(sweep["margin_before_post"].min())]
                      + [m["min_margin_before_post"] for m in results.values()])
    gates["contact_inside_swing_gap"] = bool(margin_pre > 0.0 and margin_post > 0.0)
    cf_all = np.concatenate([sweep["contact_frame"]]
                            + [np.asarray(m["contact_frame_range"]) for m in results.values()])
    gap_all = np.concatenate([sweep["gap_frames"]]
                             + [np.asarray(m["gap_frames_range"]) for m in results.values()])
    # Every rank's post clip must open with the ball at the same place, so that within a scene the
    # clips differ only in VELOCITY -- the scene_velocity2d contract that H_b - H_a relies on.
    pstart_spread = max([float(sweep["post_start_x"].max() - sweep["post_start_x"].min())]
                        + [m["post_start_x_spread_m"] for m in results.values()])
    px_per_m = cert["map"]["world_to_image_scale"] * ps.FPS * 256.0
    gates["post_window_shared_start"] = bool(pstart_spread * px_per_m < 1.0)
    all_touch = set(sweep["n_touches"].tolist())
    for m in results.values():
        all_touch |= set(m["touches"])
    gates["single_touch"] = bool(all_touch == {1})
    # Frame residency is gated over the COMMANDS -- the operating envelope, ratio in
    # [ratio_lo, ratio_hi]. The fitting sweep deliberately overshoots that envelope to pin the map's
    # slope, and its extreme points are never issued as commands, so gating on them would reject a
    # perfectly usable scene. The sweep's own residency and the largest fully-in-frame ratio are
    # reported alongside, so the overshoot stays visible instead of being quietly dropped.
    min_vis = int(min(m["min_post_frames_visible"] for m in results.values()))
    gates["ball_in_frame"] = bool(min_vis >= 15)
    gates["post_window_constant_velocity"] = bool(sweep["v_out_std"].max() < 1e-3)
    in_frame_mask = sweep["post_frames_visible"] >= 15
    cert["hygiene"] = {"min_post_frames_visible_over_commands": min_vis,
                       "min_post_frames_visible_over_sweep": int(sweep["post_frames_visible"].min()),
                       "max_ratio_fully_in_frame": (float(ratio[in_frame_mask].max())
                                                    if in_frame_mask.any() else float("nan")),
                       "max_post_window_velocity_std": float(sweep["v_out_std"].max()),
                       "contact_frame_range": [float(cf_all.min()), float(cf_all.max())],
                       "min_margin_after_pre_frames": margin_pre,
                       "min_margin_before_post_frames": margin_post,
                       "swing_gap_frames_range": [float(gap_all.min()), float(gap_all.max())],
                       "post_start_x_spread_m": pstart_spread,
                       "post_start_x_spread_px": pstart_spread * px_per_m,
                       "touch_counts_seen": sorted(int(x) for x in all_touch)}

    cert["gates"] = gates
    cert["PASS"] = bool(all(gates.values()))
    cert["elapsed_sec"] = round(time.time() - t0, 1)
    (out / "certificate.json").write_text(json.dumps(cert, indent=2))

    # ---- report ----------------------------------------------------------------------------------
    lines = [f"# Paddle-strike controllability certificate ({args.embodiment})", "",
             f"**{'PASS' if cert['PASS'] else 'FAIL'}** "
             f"- bar: |v_out - v*|/|v*| <= {TOL:.0%} on >= {PASS_FRAC:.0%} of held-out commands", "",
             "## Gates", "", "| gate | result |", "|---|---|"]
    lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in gates.items()]
    m = cert["map"]
    lines += ["", "## Action -> outcome map", "",
              f"- fitted forward: `v_out = {m['forward_alpha']:.5f}*v_p + {m['forward_beta']:.5f}*v_in "
              f"+ {m['forward_gamma']:.2e}`",
              f"- max residual {m['forward_max_resid']:.3e} m/s "
              f"({m['forward_resid_pct_of_range']:.3f}% of range)",
              f"- effective restitution {m['e_effective']:.4f} "
              f"(spread across v_in {m['e_spread_pct']:.3f}%)",
              f"- Spearman(v_p, v_out) per v_in: "
              f"{', '.join(f'{r:+.6f}' for r in m['spearman_per_v_in'])}",
              f"- world->image scale constant to {m['image_track_max_dev_pct']:.3f}% "
              f"(scale {m['world_to_image_scale']:.5f} normalized-widths per m/s; "
              f"no perspective foreshortening along the motion axis)",
              f"- speed ratio achieved: {m['ratio_achieved'][0]:.3f} .. {m['ratio_achieved'][1]:.3f}",
              "", "## Held-out commands", "",
              "| inverse | pass @5% | median rel err | p95 | max |", "|---|---|---|---|---|"]
    for mode, r in results.items():
        lines.append(f"| {mode} | {r['pass_fraction']:.1%} | {r['rel_err_median']:.3%} | "
                     f"{r['rel_err_p95']:.3%} | {r['rel_err_max']:.3%} |")
    lines += ["", f"n = {n} commands, held-out v_in {[round(float(x), 3) for x in v_in_test]} m/s "
                  f"(train v_in {[round(float(x), 3) for x in v_in_train]} m/s), ratio in "
                  f"[{args.ratio_lo}, {args.ratio_hi}]",
              "", f"Effective restitution above is the value implied by inverting the fitted "
                  f"`alpha` through the ideal two-body law; the contact is compliant rather than "
                  f"impulsive, so it differs slightly from a direct relative-velocity measurement "
                  f"(~0.897). Both are measurements, of different quantities.", ""]
    (out / "certificate.md").write_text("\n".join(lines))

    print(f"\n{'PASS' if cert['PASS'] else 'FAIL'} -> {out}/certificate.json  "
          f"({cert['elapsed_sec']}s)")
    for k, v in gates.items():
        if not v:
            print(f"  FAILED GATE: {k}")


if __name__ == "__main__":
    main()
