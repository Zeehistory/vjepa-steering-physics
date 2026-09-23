"""Render paddle-strike episodes for eyeballing: filmstrips + a continuous mp4.

The certificate proves the map is controllable in numbers; this is the "does it look like what we
think it is" check.

It renders the CONTINUOUS episode, including the swing gap that the dataset itself never renders.
That matters more than it sounds. The dataset's own frames are pre-window + post-window with the
collision cut out between them, so a filmstrip built from ``sim["frames"]`` shows the ball closing on
the paddle, then a jump cut, then the ball already heading back -- the strike itself is simply
missing, and it reads as the ball reversing on its own without ever being touched. That is an
artefact of the window layout, not the physics: this script used to do exactly that, because it
described itself as rendering the continuous episode but never passed ``render_all``.

The gap is also rendered at ``--gap_slowdown`` times the frame rate, because contact lasts about half
a frame and at 1x whether any frame lands inside it is down to luck.

WHY THE STRIKE STILL LOOKED LIKE A MISS, AND WHAT FIXED IT
---------------------------------------------------------
Rendering the gap was necessary but not sufficient. Measured over the rendered frames, the ball and
the blade genuinely INTERPENETRATE at contact -- the contact is soft, so the ball dents into the face
by 2.1 mm (ratio 0.5) to 5.7 mm (ratio 3.0). The problem is the scale that lands on screen: at the
dataset's 256 px the local image scale is 200 px/m, so the deepest overlap is 0.43 to 1.13 PIXELS.
Anti-aliasing spreads a sub-pixel overlap across the ball's rim and the result reads as "nearly
touching, then the ball leaves" -- exactly the artefact the window layout used to cause, arriving now
by a completely different route.

The ratio that matters is overlap over ball radius, which is 4-10% and is SCALE FREE: no camera move
or table resize changes it. So the fix is on-screen ball size, not timing and not geometry:

* ``--image_size`` renders the demo larger than the dataset (default 768). Physics, labels and the
  certificate are untouched -- ``image_size`` only sizes the renderer, so the rollout is bit-identical.
* ``contact_closeup.png`` crops a ``--closeup_m`` window around the projected contact point and
  upscales it with nearest-neighbour, so the dent is tens of pixels and readable at figure size. This
  used to be produced by hand outside the repo, which is why it could not be regenerated or trusted.
* ``contact_trace.png`` plots the ball's leading edge against the blade's striking face over the
  strike. Two curves meeting and crossing, with the contact interval shaded, is the part a reviewer
  can check without squinting at pixels.

Every figure here is diagnostic, not data: nothing in this script feeds training.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/08_figures/demo_paddle_strike.py \
        --output_dir outputs/paddle_strike/demo --ratios 0.5,1.0,2.0,3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import (BALL_MASS, BALL_R, PADDLE_HALF, PADDLE_MASS, TABLE_H,
                                    build_striker)

GUTTER = 245                # light gutter between filmstrip cells
PAD = 4
MARK = (196, 62, 46)        # contact marker: a rule under every frame inside the collision
# Half-extent of the STRIKER along the motion axis: the striking face is at pad_x - half_x. This is
# not a constant across embodiments and treating it as one is a silent 32 mm error -- the abstract
# paddle is an 18 mm-thin box, the dynamically-actuated arm swings a 50 mm-radius sphere. Read it off
# the generator so a new striker cannot quietly inherit the wrong face.
DEFAULT_HX = PADDLE_HALF[0]


def _half_x(gen) -> float:
    return float(getattr(gen, "striker_half_x", DEFAULT_HX))


def _px_per_m(gen, sim) -> float:
    """Local image-x scale near the contact point, in pixels per metre at ``gen.image_size``.

    Measured by projecting two points a known distance apart rather than derived from fovy, so it
    stays correct if the camera is ever retuned.
    """
    probe = np.array([[0.40, 0.0, TABLE_H + BALL_R], [0.50, 0.0, TABLE_H + BALL_R]])
    u = gen.project(probe, sim)[:, 0]
    return abs(u[1] - u[0]) / 0.10 * gen.image_size


def _gap_m(sim: dict, half_x: float) -> np.ndarray:
    """Surface-to-surface clearance per rendered continuous frame. Negative == interpenetrating."""
    return (sim["pad_x_continuous"] - half_x) - (sim["ball_x_continuous"] + BALL_R)


def _closeup(sim: dict, gen, span_m: float, upscale: int) -> np.ndarray:
    """Contact-centred crop of the frames in (and adjacent to) the collision, upscaled.

    Centred on the projected contact POINT -- the blade face at the moment of first touch -- not on
    the frame centre, so the dent is dead centre in every cell at every ratio.
    """
    f, t = sim["frames_continuous"], sim["t_continuous"]
    in_contact = (t >= sim["contact_frame"]) & (t <= sim["contact_end_frame"])
    hit = np.flatnonzero(in_contact)
    lo, hi = int(hit[0]), int(hit[-1])
    idx = list(range(max(0, lo - 2), min(len(f), hi + 3)))

    n = gen.image_size
    scale = _px_per_m(gen, gen._lazy_sim())
    face_x = float(sim["pad_x_continuous"][lo]) - _half_x(gen)
    ball_z = TABLE_H + BALL_R
    u, v = gen.project(np.array([[face_x, 0.0, ball_z]]), gen._lazy_sim())[0]
    cu, cv = int(round(u * n)), int(round(v * n))
    half = int(round(span_m * scale / 2))

    cells = []
    for i in idx:
        # Pad first, then slice, so a crop window running off the frame edge is impossible.
        padded = np.pad(f[i], ((half, half), (half, half), (0, 0)), mode="edge")
        crop = padded[cv:cv + 2 * half, cu:cu + 2 * half]
        big = np.repeat(np.repeat(crop, upscale, axis=0), upscale, axis=1)
        h, w, _ = big.shape
        cell = np.full((h + 6, w, 3), GUTTER, np.uint8)
        cell[:h] = big
        if in_contact[i]:
            cell[h + 1:] = MARK
        cells.append(cell)
    gut = np.full((cells[0].shape[0], PAD, 3), GUTTER, np.uint8)
    out = [gut]
    for c in cells:
        out += [c, gut]
    return np.concatenate(out, axis=1)


def _strip(frames: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Lay frames side by side on a light gutter, underlining the ones that are in contact."""
    h, w, _ = frames[0].shape
    bar = 5
    gut = np.full((h + bar, PAD, 3), GUTTER, np.uint8)
    cells = [gut]
    for f, hit in zip(frames, contact):
        cell = np.full((h + bar, w, 3), GUTTER, np.uint8)
        cell[:h] = f
        if hit:
            cell[h + 1:] = MARK
        cells += [cell, gut]
    return np.concatenate(cells, axis=1)


def _trace(sims: dict, out_path: Path, half_x: float) -> None:
    """Ball leading edge vs blade striking face, over the strike. The touch, as two curves.

    Plotted in millimetres of clearance rather than absolute x: the interesting quantity is the gap,
    and at these ratios the two absolute positions differ by less than the line width.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sims), figsize=(3.1 * len(sims), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (r, sim) in zip(axes, sims.items()):
        t, gap = sim["t_continuous"], _gap_m(sim, half_x) * 1e3
        keep = (t > sim["contact_frame"] - 3) & (t < sim["contact_end_frame"] + 3)
        ax.axhspan(gap[keep].min(), 0, color="#c43e2e", alpha=0.10, lw=0)
        ax.axhline(0.0, color="#444", lw=1.0, zorder=3)
        ax.plot(t[keep], gap[keep], "-o", ms=3.2, lw=1.4, color="#2f6fb0", zorder=4)
        ax.axvspan(sim["contact_frame"], sim["contact_end_frame"],
                   color="#c43e2e", alpha=0.16, lw=0, zorder=1)
        ax.set_title(f"ratio {r:g}   min {gap.min():.2f} mm", fontsize=9)
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("ball edge $-$ blade face  (mm)\n$<0$ = interpenetrating")
    fig.suptitle("Shaded band = MuJoCo reports the ball/blade contact pair active", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--v_in", type=float, default=1.0)
    ap.add_argument("--ratios", default="0.5,1.0,2.0,3.0")
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--gap_slowdown", type=int, default=8,
                    help="render the swing gap at N x the frame rate so contact lands on-screen")
    ap.add_argument("--image_size", type=int, default=768,
                    help="render size for the DEMO only; the dataset is 256. Physics is unaffected.")
    ap.add_argument("--closeup_m", type=float, default=0.26,
                    help="width of the contact close-up window, in metres of world space")
    ap.add_argument("--closeup_upscale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20, help="playback rate of the episode mp4s")
    ap.add_argument("--wide", action="store_true",
                    help="render from the wide demo camera instead of the dataset camera. Only the "
                         "dynamically-actuated arm has one, and it is the only way to actually SEE "
                         "the robot: the dataset camera is framed tight on the ball's travel and "
                         "crops the arm to a wrist stub. Pixels only -- every measured quantity is "
                         "computed from world state or the fixed camera's projection, so this "
                         "cannot move a label. The close-up and the on-screen overlap figure are "
                         "skipped under it, since both are calibrated in the FIXED camera's pixels "
                         "per metre and would be quietly wrong read off another view.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = build_striker(args.embodiment, image_size=args.image_size)

    # command each ratio through the fitted inverse, exactly as the certificate does
    inv = StrikeInverse.fit(sweep_strikes(gen, [0.85, 0.95, 1.05, 1.15],
                                          np.linspace(0.25, -1.45, 20)),
                            BALL_MASS, PADDLE_MASS)

    ratios = [float(r) for r in args.ratios.split(",")]
    rows, lines, sims, closeups = [], [], {}, []
    scale = _px_per_m(gen, gen._lazy_sim())
    if args.wide:
        if not hasattr(gen, "use_demo_camera"):
            raise SystemExit(f"--wide: embodiment '{args.embodiment}' has no wide demo camera")
        gen.use_demo_camera(True)
    for r in ratios:
        target = -r * args.v_in
        v_p = float(np.asarray(inv.action_for(args.v_in, target, mode="quadratic")).ravel()[0])
        sim = gen.simulate(args.v_in, v_p, render=True, render_all=True,
                           gap_slowdown=args.gap_slowdown)

        f = sim["frames_continuous"]                 # pre + swing gap + post, in capture order
        t = sim["t_continuous"]                      # capture time of each, in frame units
        in_contact = (t >= sim["contact_frame"]) & (t <= sim["contact_end_frame"])

        # Filmstrip: sample the approach coarsely, then every frame in or adjacent to the collision,
        # so the cell where the blade actually meets the ball is always one of the ones shown.
        hit = np.flatnonzero(in_contact)
        if not hit.size:
            raise SystemExit(f"ratio {r}: no rendered frame inside the contact interval "
                             f"-- raise --gap_slowdown above {args.gap_slowdown}")
        lo, hi = int(hit[0]), int(hit[-1])
        idx = sorted(set([0, int(lo * 0.45), int(lo * 0.75)]
                         + list(range(max(0, lo - 2), min(len(f), hi + 3)))
                         + [min(len(f) - 1, hi + 6), min(len(f) - 1, hi + 14), len(f) - 1]))
        rows.append(_strip(f[idx], in_contact[idx]))

        # Keep only what contact_trace needs. Retaining the full sim keeps every rendered frame of
        # every ratio alive to the end of the run, which is the other half of the memory problem.
        sims[r] = {k: sim[k] for k in ("t_continuous", "pad_x_continuous", "ball_x_continuous",
                                       "contact_frame", "contact_end_frame")}
        gap = _gap_m(sim, _half_x(gen))
        achieved = sim["v_out_world"]
        head = (f"ratio {r:.2f}: commanded {target:+.3f} m/s -> achieved {achieved:+.3f} m/s "
                f"({abs(achieved - target) / abs(target):.2%} err), paddle {v_p:+.3f} m/s, "
                f"contact frames {sim['contact_frame']:.2f}..{sim['contact_end_frame']:.2f} "
                f"({int(in_contact.sum())} rendered frames in contact)")
        if args.wide:
            # deepest overlap in mm is a world quantity and still true; the px figure is not, so it
            # is omitted rather than restated in the wrong camera's units
            lines.append(f"{head}, deepest overlap {-gap.min() * 1e3:.2f} mm "
                         f"({int((gap <= 0).sum())} frames overlapping)")
        else:
            closeups.append(_closeup(sim, gen, args.closeup_m, args.closeup_upscale))
            # The overlap actually on screen. Reported in px as well as mm because px is the number
            # that decides whether a human can see it, and at 256 it was sub-pixel for every ratio.
            lines.append(f"{head}, deepest overlap {-gap.min() * 1e3:.2f} mm = "
                         f"{-gap.min() * scale:.1f} px at {gen.image_size} "
                         f"({int((gap <= 0).sum())} frames overlapping)")
        print(lines[-1], flush=True)

        # Written straight from uint8. Going via a float tensor and save_video round-trips through
        # a float32 copy of the whole clip -- 4x the frames, ~1.8 GB at 768 with the gap rendered at
        # 8x -- and save_video's first act is to convert it back to uint8 anyway. This script runs
        # inside an 8 GB allocation and that spike was enough to get it OOM-killed.
        import imageio.v2 as iio_w
        iio_w.mimwrite(out / f"episode_ratio{r:g}.mp4", list(f), fps=args.fps,
                       codec="libx264", quality=8)

    import imageio.v2 as iio

    def _stack(name: str, panels: list[np.ndarray]) -> None:
        width = max(p.shape[1] for p in panels)
        iio.imwrite(out / name, np.concatenate(
            [np.pad(p, ((0, PAD), (0, width - p.shape[1]), (0, 0)), constant_values=GUTTER)
             for p in panels], axis=0))

    _stack("filmstrips.png", rows)
    if closeups:
        _stack("contact_closeup.png", closeups)
    _trace(sims, out / "contact_trace.png", _half_x(gen))
    (out / "demo.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}/filmstrips.png, "
          f"{'contact_closeup.png, ' if closeups else ''}contact_trace.png  "
          f"(rows = {ratios}; red rule = frame is in contact)")


if __name__ == "__main__":
    main()
