"""Side-by-side demo video: the SAME requested outcome, without steering and with it.

The goal is a video that shows what the controller buys you. The certificate says the closed loop
lands inside +-5% of any commanded ball speed; this renders the counterfactual next to it, so the
claim is visible rather than tabulated.

WHAT THE TWO PANELS ACTUALLY ARE
--------------------------------
Both panels are the same scene, the same incoming ball, the same renderer, the same frame grid. The
only thing that differs is the striker command.

* LEFT / "no steering" -- the striker runs ONE fixed nominal swing regardless of what was asked for.
  The nominal is the command that returns the ball at ratio 1.0 (mirror return), i.e. the sensible
  default an uncontrolled rig would sit at. So the left panel is identical in every cell of the
  comparison, and the ball comes off at the same speed no matter what the caption asked for. That is
  the point: with no steering, the request has no effect.
* RIGHT / "after steering" -- the command comes from the fitted strike inverse for THIS target, the
  same pathway the certificate scores. The ball comes off at the requested speed.

The nominal is deliberately not "zero command" / a stationary blade. A dead blade is a strawman: it
would fail every target trivially and by a huge margin, and it also changes the contact regime
(nearly-elastic bounce off a static wall) so the two panels would no longer differ by only the
control. Holding the swing at a fixed sensible default is the honest baseline and it is the one a
reviewer will accept.

WHAT IS MEASURED, NOT DRAWN
---------------------------
The caption strip under each pair reports commanded vs achieved for both panels, computed from
``v_out_world`` -- the same world-state measurement the certificate uses, not anything read off
pixels. The bar plot in ``steering_compare.png`` is that table.

Only ``--embodiment`` changes which rig; physics, labels and the certificate are untouched by
anything in this file. Every figure here is diagnostic: nothing feeds training.

Usage::

    PYTHONPATH=. MUJOCO_GL=egl python experiments/threads/paddle-robotics/05_steering/demo_steering_compare.py \
        --output_dir /path/out --embodiment paddle --ratios 0.5,1.0,2.0,3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS, build_striker

GUTTER = 245
PAD = 6
BAR_H = 6
MARK = (196, 62, 46)          # in-contact rule, same convention as demo_paddle_strike
OK = (54, 132, 82)
BAD = (196, 62, 46)


def _caption(width: int, lines: list[tuple[str, tuple[int, int, int]]], height: int = 74):
    """Render a caption band to a uint8 array with matplotlib.

    Matplotlib rather than PIL because the repo already depends on it for every other figure, and
    the default PIL bitmap font is unreadable next to a 768 px render.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(np.array([GUTTER] * 3) / 255.0)
    y = 0.80
    for text, rgb in lines:
        fig.text(0.012, y, text, fontsize=9.5, va="top", ha="left",
                 family="DejaVu Sans", color=np.array(rgb) / 255.0)
        y -= 0.34
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _pad_to(frames: np.ndarray, n: int) -> np.ndarray:
    """Hold the last frame so two rollouts of unequal length can play side by side."""
    if len(frames) >= n:
        return frames[:n]
    tail = np.repeat(frames[-1:], n - len(frames), axis=0)
    return np.concatenate([frames, tail], axis=0)


def _label_strip(width: int, text: str, rgb=(60, 60, 60)) -> np.ndarray:
    return _caption(width, [(text, rgb)], height=30)


def _side_by_side(fa: np.ndarray, fb: np.ndarray, contact_a: np.ndarray,
                  contact_b: np.ndarray) -> np.ndarray:
    """Two clips into one clip, each underlined red on the frames MuJoCo reports as in contact."""
    n = max(len(fa), len(fb))
    fa, fb = _pad_to(fa, n), _pad_to(fb, n)
    ca, cb = _pad_to(contact_a[:, None], n)[:, 0], _pad_to(contact_b[:, None], n)[:, 0]
    h, w, _ = fa[0].shape
    out = np.full((n, h + BAR_H, 2 * w + 3 * PAD, 3), GUTTER, np.uint8)
    out[:, :h, PAD:PAD + w] = fa
    out[:, :h, 2 * PAD + w:2 * PAD + 2 * w] = fb
    for i in range(n):
        if ca[i]:
            out[i, h + 1:, PAD:PAD + w] = MARK
        if cb[i]:
            out[i, h + 1:, 2 * PAD + w:2 * PAD + 2 * w] = MARK
    return out


def _summary(rows: list[dict], out_path: Path, embodiment: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt = np.array([r["target"] for r in rows])
    uns = np.array([r["achieved_unsteered"] for r in rows])
    ste = np.array([r["achieved_steered"] for r in rows])
    x = np.arange(len(rows))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax.plot(x, np.abs(tgt), "k--o", ms=5, lw=1.4, label="requested")
    ax.plot(x, np.abs(uns), "-s", ms=5, lw=1.6, color="#c43e2e", label="no steering (fixed swing)")
    ax.plot(x, np.abs(ste), "-o", ms=5, lw=1.6, color="#2f6fb0", label="after steering")
    ax.set_xticks(x, [f"{r['ratio']:g}x" for r in rows])
    ax.set_ylabel("|ball speed out|  (m/s)")
    ax.set_xlabel("requested outcome")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    eu = np.abs(uns - tgt) / np.abs(tgt) * 100
    es = np.abs(ste - tgt) / np.abs(tgt) * 100
    ax2.bar(x - 0.19, eu, 0.36, color="#c43e2e", label="no steering")
    ax2.bar(x + 0.19, es, 0.36, color="#2f6fb0", label="after steering")
    ax2.axhline(5.0, color="#444", lw=1.0, ls=":")
    ax2.text(len(rows) - 0.5, 5.4, "+-5% spec", fontsize=8, ha="right", color="#444")
    ax2.set_yscale("symlog", linthresh=1.0)
    ax2.set_xticks(x, [f"{r['ratio']:g}x" for r in rows])
    ax2.set_ylabel("relative error (%)")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.25, lw=0.5, axis="y")
    fig.suptitle(f"{embodiment}: same request, with and without the controller", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--v_in", type=float, default=1.0)
    ap.add_argument("--ratios", default="0.5,1.0,2.0,3.0")
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--nominal_ratio", type=float, default=1.0,
                    help="the ONE swing the unsteered rig always runs, as an outcome ratio")
    ap.add_argument("--gap_slowdown", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--wide", action="store_true",
                    help="wide demo camera (arm embodiments only) -- pixels only, moves no label")
    ap.add_argument("--look", action="store_true",
                    help="franka_dynamic only: shared paddle palette + blue blade drawn over the "
                         "mallet sphere -- pixels only, the sphere still does every contact")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = build_striker(args.embodiment, image_size=args.image_size)
    if args.look:
        if not hasattr(gen, "use_demo_look"):
            raise SystemExit(f"--look: embodiment '{args.embodiment}' has no demo look")
        gen.use_demo_look(True)
    if args.wide:
        if not hasattr(gen, "use_demo_camera"):
            raise SystemExit(f"--wide: embodiment '{args.embodiment}' has no wide demo camera")
        gen.use_demo_camera(True)

    inv = StrikeInverse.fit(sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15],
                                          np.linspace(0.25, -1.45, 20)),
                            BALL_MASS, PADDLE_MASS)

    def action(target: float) -> float:
        return float(np.asarray(inv.action_for(args.v_in, target, mode="quadratic")).ravel()[0])

    # The single fixed swing the unsteered rig runs for EVERY request.
    v_p_nom = action(-args.nominal_ratio * args.v_in)

    import imageio.v2 as iio

    ratios = [float(r) for r in args.ratios.split(",")]
    rows, lines, panels = [], [], []
    for r in ratios:
        target = -r * args.v_in
        v_p_steer = action(target)

        sims = {}
        for tag, v_p in (("unsteered", v_p_nom), ("steered", v_p_steer)):
            s = gen.simulate(args.v_in, v_p, render=True, render_all=True,
                             gap_slowdown=args.gap_slowdown)
            t = s["t_continuous"]
            sims[tag] = {
                "frames": s["frames_continuous"],
                "contact": (t >= s["contact_frame"]) & (t <= s["contact_end_frame"]),
                "v_out": float(s["v_out_world"]),
                "v_p": float(v_p),
            }

        clip = _side_by_side(sims["unsteered"]["frames"], sims["steered"]["frames"],
                             sims["unsteered"]["contact"], sims["steered"]["contact"])
        w = clip.shape[2]

        eu = abs(sims["unsteered"]["v_out"] - target) / abs(target)
        es = abs(sims["steered"]["v_out"] - target) / abs(target)
        head = _label_strip(w, f"REQUEST: return the ball at {abs(target):.2f} m/s  ({r:g}x in)")
        cap = _caption(w, [
            (f"no steering  -- fixed {args.nominal_ratio:g}x swing (blade {v_p_nom:+.3f} m/s):"
             f"   got {abs(sims['unsteered']['v_out']):.3f} m/s    error {eu:6.2%}"
             f"   {'PASS' if eu <= 0.05 else 'FAIL'} at +-5%", BAD if eu > 0.05 else OK),
            (f"after steering -- commanded blade {v_p_steer:+.3f} m/s:"
             f"   got {abs(sims['steered']['v_out']):.3f} m/s    error {es:6.2%}"
             f"   {'PASS' if es <= 0.05 else 'FAIL'} at +-5%", OK if es <= 0.05 else BAD),
        ])
        head = np.pad(head, ((0, 0), (0, w - head.shape[1]), (0, 0)), constant_values=GUTTER)
        cap = np.pad(cap, ((0, 0), (0, w - cap.shape[1]), (0, 0)), constant_values=GUTTER)
        framed = np.concatenate(
            [np.broadcast_to(head, (len(clip),) + head.shape), clip,
             np.broadcast_to(cap, (len(clip),) + cap.shape)], axis=1)

        iio.mimwrite(out / f"compare_ratio{r:g}.mp4", list(framed.astype(np.uint8)),
                     fps=args.fps, codec="libx264", quality=8)
        panels.append(framed)

        rows.append({"ratio": r, "target": target, "v_p_nominal": v_p_nom,
                     "v_p_steered": v_p_steer,
                     "achieved_unsteered": sims["unsteered"]["v_out"],
                     "achieved_steered": sims["steered"]["v_out"],
                     "rel_err_unsteered": eu, "rel_err_steered": es})
        lines.append(f"ratio {r:g}: requested {target:+.3f} | no-steer {sims['unsteered']['v_out']:+.3f} "
                     f"({eu:.2%}) | steered {sims['steered']['v_out']:+.3f} ({es:.2%})")
        print(lines[-1], flush=True)

    # One reel with every request back to back -- the thing to actually put in a talk.
    n = max(p.shape[1] for p in panels)
    wmax = max(p.shape[2] for p in panels)
    reel = [np.pad(f, ((0, n - p.shape[1]), (0, wmax - p.shape[2]), (0, 0)),
                   constant_values=GUTTER)
            for p in panels for f in p]
    iio.mimwrite(out / "compare_all.mp4", reel, fps=args.fps, codec="libx264", quality=8)

    _summary(rows, out / "steering_compare.png", args.embodiment)
    import json
    (out / "steering_compare.json").write_text(json.dumps(
        {"embodiment": args.embodiment, "v_in": args.v_in,
         "nominal_ratio": args.nominal_ratio, "rows": rows}, indent=2))
    (out / "compare.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}/compare_all.mp4 + per-ratio mp4s + steering_compare.png")


if __name__ == "__main__":
    main()
