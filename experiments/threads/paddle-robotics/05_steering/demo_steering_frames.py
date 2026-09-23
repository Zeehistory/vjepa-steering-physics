"""Three-frame still comparison: the same request, without steering and with it.

The mp4s from :mod:`scripts.paddle.demo_steering_compare` carry the result but they are a poor way to
show it -- a reviewer watching two panels for four seconds cannot hold the "before" in their head
while the "after" plays, and a talk slide cannot autoplay reliably anyway. This renders the same
counterfactual as SIX stills on one sheet: two rows (no steering / after steering) by three columns
(approach, contact, return).

WHY THESE THREE INSTANTS, AND WHY THEY ARE SHARED
-------------------------------------------------
Both rows are sampled at the SAME three times. That is what makes the sheet an honest comparison
rather than two flattering picks: the ball arrives on a fixed schedule and the swing runs on a fixed
schedule, so the same wall-clock instant means the same phase of the same event in both rows. The
instants are taken from the unsteered rollout's contact window:

* APPROACH -- a few frames before first contact, while nothing has happened yet. The two rows should
  look essentially identical here, and that is the control: any difference downstream is the
  controller, not the setup.
* CONTACT  -- the midpoint of the contact interval MuJoCo reports.
* RETURN   -- the last frame, after the ball has separated and is running out.

Only the RETURN column is supposed to differ, and the amount it differs by is the whole claim.

WHAT IS DRAWN VS WHAT IS MEASURED
---------------------------------
The speed annotated on each row comes from ``v_out_world`` -- world state, the same measurement the
certificate scores -- not from anything read off pixels. The pixels are an illustration of a number
computed elsewhere; they are never the evidence. Nothing here feeds training.

Usage::

    PYTHONPATH=. MUJOCO_GL=egl python experiments/threads/paddle-robotics/05_steering/demo_steering_frames.py \
        --output_dir /path/out --embodiment franka_dynamic --ratio 0.5 --wide
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS, build_striker

INK = "#1b1f24"
MUTED = "#7c8794"
OK = "#1b7f5a"
BAD = "#c43e2e"
RULE = "#d8dee6"


def _pick_instants(t: np.ndarray, c0: float, c1: float, lead: float) -> list[float]:
    """The three shared sample times: pre-contact, mid-contact, final."""
    return [max(float(t.min()), c0 - lead), 0.5 * (c0 + c1), float(t.max())]


def _nearest(t: np.ndarray, want: float) -> int:
    return int(np.abs(t - want).argmin())


def _crop(img: np.ndarray, keep: tuple[float, float]) -> np.ndarray:
    """Keep a vertical band of the frame. The rigs put the action in a horizontal strip and the rest
    is backdrop, which at 768 px is most of the sheet's area spent on nothing."""
    lo, hi = keep
    h = img.shape[0]
    return img[int(lo * h):int(hi * h)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="franka_dynamic")
    ap.add_argument("--v_in", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="requested outcome as a multiple of the incoming speed. 0.5 is the clearest "
                         "case: the unsteered rig is ~100%% wrong, so the two rows differ maximally.")
    ap.add_argument("--nominal_ratio", type=float, default=1.0,
                    help="the ONE swing the unsteered rig always runs, as an outcome ratio")
    # 2, not the demo video's 8: that setting exists to put several frames inside the ~0.5-frame
    # contact interval so the strike is visible in motion. Here only three stills survive, and at
    # 768 px the discarded frames are what makes this job need ~96G instead of running anywhere.
    # The contact instant is still resolved -- it is picked from the reported contact window, not
    # from whichever frame happens to land nearest.
    ap.add_argument("--gap_slowdown", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--lead_frames", type=float, default=3.0)
    ap.add_argument("--crop", default="0.10,0.92",
                    help="vertical band of each frame to keep, as lo,hi fractions")
    ap.add_argument("--wide", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = tuple(float(v) for v in args.crop.split(","))

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
    v_p_nom = action(-args.nominal_ratio * args.v_in)     # the one fixed swing, for every request
    v_p_steer = action(target)

    sims = {}
    for tag, v_p in (("unsteered", v_p_nom), ("steered", v_p_steer)):
        s = gen.simulate(args.v_in, v_p, render=True, render_all=True,
                         gap_slowdown=args.gap_slowdown)
        sims[tag] = {"frames": s["frames_continuous"], "t": s["t_continuous"],
                     "c0": float(s["contact_frame"]), "c1": float(s["contact_end_frame"]),
                     "v_out": float(s["v_out_world"]), "v_p": float(v_p)}

    # Shared instants, taken from the unsteered rollout. Both rigs swing on the same fixed schedule,
    # so these are the same phase of the same event in both rows -- see the module docstring.
    u = sims["unsteered"]
    instants = _pick_instants(u["t"], u["c0"], u["c1"], args.lead_frames)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panel = _crop(sims["unsteered"]["frames"][0], keep)
    ar = panel.shape[1] / panel.shape[0]
    cw = 3.15
    # Margins in INCHES, converted once. The row labels ("after steering" at 12.5 pt bold) live in
    # the left margin and the title block in the top one, so both are sized to the text rather than
    # guessed -- an under-sized left margin silently clips the labels off the canvas edge.
    m_left, m_top, m_bot, m_right = 1.80, 1.05, 0.62, 0.20
    fig_w = 3 * cw + m_left + m_right
    fig_h = 2 * (cw / ar) + m_top + m_bot + 0.30
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    left, top = m_left / fig_w, 1.0 - m_top / fig_h
    pw, ph = cw / fig_w, (cw / ar) / fig_h
    gx, gy = 0.055 / fig_w, 0.30 / fig_h

    col_titles = ["approach", "contact", "return"]
    rows = [("no steering", "unsteered", v_p_nom), ("after steering", "steered", v_p_steer)]
    # One shared arrow scale across both rows, so the two lengths are comparable by eye.
    v_ref = max(abs(sims["unsteered"]["v_out"]), abs(sims["steered"]["v_out"]), 1e-9)

    for ri, (row_label, tag, v_p) in enumerate(rows):
        s = sims[tag]
        err = abs(s["v_out"] - target) / abs(target)
        col = OK if err <= 0.05 else BAD
        for ci, want in enumerate(instants):
            x = left + ci * (pw + gx)
            y = top - (ri + 1) * ph - ri * gy
            ax = fig.add_axes([x, y, pw, ph])
            ax.imshow(_crop(s["frames"][_nearest(s["t"], want)], keep), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(RULE); sp.set_linewidth(0.8)
            if ri == 0:
                ax.set_title(col_titles[ci].upper(), fontsize=9.5, color=MUTED, pad=8,
                             fontfamily="DejaVu Sans", fontweight="bold")
            # The return panel is the only one carrying a claim, so it is the only one annotated.
            if ci == 2:
                ax.add_patch(Rectangle((0.035, 0.845), 0.295, 0.115, transform=ax.transAxes,
                                       facecolor="white", alpha=0.90, edgecolor=col, lw=1.0,
                                       zorder=3))
                ax.text(0.055, 0.903, f"{abs(s['v_out']):.2f} m/s", transform=ax.transAxes,
                        fontsize=11, color=col, va="center", ha="left", zorder=4,
                        fontweight="bold", fontfamily="DejaVu Sans")
                # Outgoing speed as a DRAWN arrow: same anchor and same scale in both rows, length
                # proportional to the measured |v_out|. The stills alone under-sell the result --
                # two balls a few frames after contact sit only a few pixels apart -- and this makes
                # the 2x speed difference directly legible without asserting anything the pixels do
                # not support. It is an annotation of a measured number, not a measurement.
                L = 0.52 * abs(s["v_out"]) / v_ref
                ax.annotate("", xy=(0.88 - L, 0.17), xytext=(0.88, 0.17),
                            xycoords="axes fraction", zorder=5,
                            arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                            color=col, lw=2.6, shrinkA=0, shrinkB=0))

        # row label + the measured outcome, in the left margin
        ymid = top - (ri + 1) * ph - ri * gy + ph / 2
        fig.text(left - 0.030, ymid + 0.055, row_label, fontsize=12.5, color=INK,
                 ha="right", va="center", fontweight="bold", fontfamily="DejaVu Sans")
        fig.text(left - 0.030, ymid - 0.005, f"blade {v_p:+.2f} m/s", fontsize=8.5, color=MUTED,
                 ha="right", va="center", fontfamily="DejaVu Sans")
        fig.text(left - 0.030, ymid - 0.058, f"{err:.0%} off" if err > 0.005 else "on target",
                 fontsize=9.5, color=col, ha="right", va="center", fontweight="bold",
                 fontfamily="DejaVu Sans")

    fig.text(m_left / fig_w - 0.030, 1.0 - 0.26 / fig_h,
             f"Request: return the ball at {abs(target):.2f} m/s",
             fontsize=15, color=INK, va="top", ha="left", fontweight="bold",
             fontfamily="DejaVu Sans")
    fig.text(m_left / fig_w - 0.030, 1.0 - 0.52 / fig_h,
             f"{args.embodiment} — the only difference between the rows is the striker "
             f"command; speeds are world state, not pixels",
             fontsize=9, color=MUTED, va="top", ha="left", fontfamily="DejaVu Sans")
    fig.text(m_left / fig_w - 0.030, 0.30 / fig_h,
             f"Without steering the rig runs one fixed {args.nominal_ratio:g}x swing whatever is "
             f"asked, and returns {abs(sims['unsteered']['v_out']):.2f} m/s. With steering it "
             f"returns {abs(sims['steered']['v_out']):.2f} m/s.",
             fontsize=9.5, color=INK, va="center", ha="left", fontfamily="DejaVu Sans")

    stem = f"steering_frames_{args.embodiment}_{args.ratio:g}x"
    fig.savefig(out / f"{stem}.png", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    plt.close(fig)

    rec = {"embodiment": args.embodiment, "ratio": args.ratio, "target": target,
           "instants": instants, "v_p_nominal": v_p_nom, "v_p_steered": v_p_steer,
           "achieved_unsteered": sims["unsteered"]["v_out"],
           "achieved_steered": sims["steered"]["v_out"]}
    (out / f"{stem}.json").write_text(json.dumps(rec, indent=2))
    print(f"requested {target:+.3f} | no-steer {sims['unsteered']['v_out']:+.3f} | "
          f"steered {sims['steered']['v_out']:+.3f}")
    print(f"wrote {out}/{stem}.png (+ .pdf, .json)")


if __name__ == "__main__":
    main()
