#!/usr/bin/env python
"""Still sheet demonstrating VECTOR-valued steering: aim the struck ball, don't just set its speed.

``demo_steering_frames.py`` shows the scalar result -- one commanded outgoing SPEED, two rows (fixed
swing vs steered swing). This is its 2-D sibling: three rows, each a different commanded outgoing
DIRECTION, all reaching the same commanded speed. It is the picture behind the `vector_steer` numbers
(franka_dynamic held-out |v_out_y| residual 0.018, closed-loop pass@5% 82.5%), which until now existed
only as a scatter plot of targets against achievements.

The controller is the same one `vector_steer.py` certifies, imported rather than reimplemented so the
sheet cannot drift from the result it illustrates: sweep the ``(v_p, y_b0)`` action grid, pick a model
by HELD-OUT error with the same margin rule (a richer model must beat linear by 10% to earn the swap),
then invert for each commanded ``(v_x, v_y)``. On ``franka_dynamic`` the winner is the ``full`` model
(``y^2`` plus the ``v_p*y`` cross term -- the lateral throw is a PRODUCT of impact speed and offset, so
a model quadratic in ``y`` alone cannot express it at any order); on the flat paddle nothing beats
linear and the linear inverse is used, which is what keeps the negative control honest.

Honest-comparison rules, inherited from the scalar sheet for the same reasons:

* **All three rows are sampled at the SAME three instants**, taken from the centre row's contact
  window. The ball and the swing run on fixed schedules, so identical wall-clock is identical phase.
  Picking each row's own most flattering frames would make this an advert rather than a comparison.
* **The third column is the END of the post-contact flight**, not a few frames after it. That is the
  difference between a demonstration and a caption: at ``c1 + 3`` frames a +-0.45 m/s lateral command
  moves the ball a couple of pixels, so the first version of this sheet showed three visually
  identical balls and the whole claim lived in the annotation.
* **The velocity arrow is drawn at one shared scale across rows**, anchored at the ball. It stays
  even now that the pixels separate, because the arrow carries the SPEED (identical across rows by
  construction) as well as the direction, and speed is genuinely not readable from a still.
* The arrow annotates a MEASURED outcome (``v_out_world``, ``v_out_lateral_signed``). It does not
  assert that the pixels alone prove the number.
* The centre row (``v_y* = 0``) is the control: the same machinery commanded to do nothing laterally.
  On the flat paddle every row collapses onto it, which is the negative control the vector result
  rests on (``cond(A) ~ 7.5e28``, lateral range +-0.000).

    MUJOCO_GL=egl PYTHONPATH=. python experiments/threads/paddle-robotics/08_figures/demo_vector_frames.py \
        --output_dir $SCRATCH/paddle_strike/vector_frames --embodiment franka_dynamic
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
import importlib.util
import json
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from src.data.paddle_strike import build_striker

OK, BAD, MUTED, RULE = "#1a7f37", "#b3261e", "#57606a", "#d0d7de"


def _load_vector_steer():
    """Import ``vector_steer.py`` as a module so its fitted controller is reused, not re-derived."""
    path = _REPO_ROOT / "experiments/threads/paddle-robotics/05_steering/vector_steer.py"
    spec = importlib.util.spec_from_file_location("_vector_steer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _pick_instants(t: np.ndarray, c0: float, c1: float, lead: float) -> list[float]:
    """approach / contact / LAST rendered frame.

    The third column is the last available instant rather than ``c1 + lead``. At a few frames past
    contact a +-0.45 m/s lateral difference is a couple of pixels, so the first version of this sheet
    showed three balls in visually identical places and the entire claim rested on the annotation
    arrow. Sampling the end of the post window lets the commanded directions actually separate ON THE
    TABLE, which is what makes this a demonstration rather than a caption. Still one shared instant
    across rows -- taken from the centre row, as before -- so it remains a like-for-like comparison.
    """
    return [c0 - lead, 0.5 * (c0 + c1), float(np.asarray(t).max())]


def _nearest(t: np.ndarray, want: float) -> int:
    return int(np.abs(np.asarray(t) - want).argmin())


def _crop(img: np.ndarray, keep: tuple[float, float]) -> np.ndarray:
    h = img.shape[0]
    a, b = int(keep[0] * h), int(keep[1] * h)
    return img[a:b]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="franka_dynamic")
    ap.add_argument("--v_in", type=float, default=1.0)
    ap.add_argument("--speed_ratio", type=float, default=2.2,
                    help="commanded |v_out_x| as a multiple of v_in, shared by all three rows")
    ap.add_argument("--lat_targets", default="-0.45,0.0,0.45",
                    help="commanded v_out_y (m/s) per row")
    ap.add_argument("--n_vp", type=int, default=7)
    ap.add_argument("--n_y", type=int, default=7)
    ap.add_argument("--vp_lo", type=float, default=-1.45)
    ap.add_argument("--vp_hi", type=float, default=0.25)
    ap.add_argument("--y_max", type=float, default=0.030)
    ap.add_argument("--image_size", type=int, default=768)
    ap.add_argument("--gap_slowdown", type=int, default=2)
    ap.add_argument("--lead_frames", type=float, default=3.0)
    ap.add_argument("--crop", default="0.10,0.92")
    ap.add_argument("--wide", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    keep = tuple(float(v) for v in args.crop.split(","))
    vs = _load_vector_steer()

    gen = build_striker(args.embodiment, image_size=args.image_size)
    if args.wide and hasattr(gen, "use_demo_camera"):
        gen.use_demo_camera(True)

    # -- fit the controller on a rendered-off sweep (cheap; no EGL needed for this part) ----------
    A_rows, U_rows, fallback = [], [], False
    for vp in np.linspace(args.vp_lo, args.vp_hi, args.n_vp):
        for y in np.linspace(-args.y_max, args.y_max, args.n_y):
            s = gen.simulate(args.v_in, float(vp), render=False, y_b0=float(y))
            vx, vy, fb = vs._outcome(s, float(y))
            fallback = fallback or fb
            A_rows.append([vp, y]); U_rows.append([vx, vy])
    if fallback:
        # Same hard gate as vector_steer: a fabricated second axis must never reach a figure looking
        # like a measurement. sign(y)*|lateral| is monotone in y and zero at y=0 by construction, so
        # the usual sanity checks cannot catch it.
        raise SystemExit("FATAL: no signed lateral key -- refusing to draw a fabricated direction.")
    A_in, U = np.array(A_rows), np.array(U_rows)

    # ``_fit`` returns (coef, predict_fn, resid_frac); only the coefficients are needed here.
    # MODEL SELECTION, with vector_steer's margin rule rather than always taking the richest model.
    # Without it the flat-paddle negative control is misrepresented: its lateral axis is exactly
    # degenerate (range 0.000 m/s, cond(A) ~ 7.5e28), the cubic elimination inside ``_invert`` then
    # returns numerical garbage, and the sheet shows the controller missing the SPEED target too --
    # which is a property of an ill-posed inverse, not of the rig. The certified controller falls back
    # to the linear inverse there and still hits the achievable axis, so the demo must do the same.
    train = (np.arange(len(A_in)) % 2 == 0)
    ho = {name: vs._heldout_resid(A_in, U, tms, train) for name, tms in vs.MODELS.items()}
    cheapest = min(vs.MODELS, key=lambda m: ho[m].sum())
    best = cheapest if ho[cheapest].sum() < 0.9 * ho["linear"].sum() else "linear"
    terms = vs.MODELS[best]
    print(f"[vecdemo] held-out model: {best}"
          f"{'' if best == cheapest else f' (ranked {cheapest}, margin too small to swap)'}", flush=True)
    W, _pred, resid = vs._fit(A_in, U, terms)
    W_lin, _pl, resid_lin = vs._fit(A_in, U, ())
    # The linear SOLUTION for each command, built the way vector_steer builds it: the inverse picks
    # the real root nearest this one, because a far-side contact can throw the ball the same way and
    # that is not the branch the linear controller is being compared against.
    A_mat = W_lin[:2, :].T          # d(outcome)/d(action)
    b_vec = W_lin[-1, :]
    lat_range = float(U[:, 1].max() - U[:, 1].min())
    print(f"[vecdemo] in-sample resid (full) = {np.round(resid, 4).tolist()}, "
          f"(linear) = {np.round(resid_lin, 4).tolist()}", flush=True)
    print(f"[vecdemo] swept {len(A_in)} actions; v_out_y range = {lat_range:.4f} m/s", flush=True)

    targets = [float(x) for x in args.lat_targets.split(",")]
    vx_star = -args.speed_ratio * args.v_in       # outgoing x velocity is negative (ball returns)

    sims = []
    for vy_star in targets:
        u_star = np.array([vx_star, vy_star])
        a_lin = vs._invert_linear(A_mat, b_vec, u_star)
        # On a degenerate axis the exact elimination is ill-posed while pinv is fine, so the linear
        # solution is used outright when no richer model earned the swap.
        a = a_lin if best == "linear" else vs._invert(W, terms, u_star, args.y_max, a_lin)
        # Same clipping to the swept action box that the certified closed loop applies, so the sheet
        # demonstrates the controller as evaluated rather than a version allowed to extrapolate.
        a = np.array([np.clip(a[0], args.vp_lo, args.vp_hi),
                      np.clip(a[1], -args.y_max, args.y_max)])
        v_p, y_b0 = float(a[0]), float(a[1])
        s = gen.simulate(args.v_in, v_p, render=True, render_all=True, y_b0=y_b0,
                         gap_slowdown=args.gap_slowdown)
        vx, vy, _ = vs._outcome(s, y_b0)
        sims.append({"target": u_star.tolist(), "action": [v_p, y_b0],
                     "achieved": [vx, vy], "frames": s["frames_continuous"],
                     "t": s["t_continuous"], "c0": float(s["contact_frame"]),
                     "c1": float(s["contact_end_frame"])})
        print(f"[vecdemo] target vy={vy_star:+.3f} -> action (v_p={v_p:+.3f}, y={y_b0:+.4f}) "
              f"-> achieved ({vx:+.3f}, {vy:+.3f})", flush=True)

    # Shared instants from the CENTRE row (the v_y*=0 control), so all rows show the same phase.
    mid = sims[len(sims) // 2]
    instants = _pick_instants(mid["t"], mid["c0"], mid["c1"], args.lead_frames)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panel = _crop(sims[0]["frames"][0], keep)
    ar = panel.shape[1] / panel.shape[0]
    cw = 3.15
    m_left, m_top, m_bot, m_right = 2.05, 1.10, 0.62, 0.20
    nrow = len(sims)
    fig_w = 3 * cw + m_left + m_right
    fig_h = nrow * (cw / ar) + m_top + m_bot + 0.30 * (nrow - 1)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    left, top = m_left / fig_w, 1.0 - m_top / fig_h
    pw, ph = cw / fig_w, (cw / ar) / fig_h
    gx, gy = 0.055 / fig_w, 0.30 / fig_h

    col_titles = ["approach", "contact", "end of flight"]
    # ONE arrow scale across every row, so lengths and angles are comparable by eye.
    v_ref = max(max(abs(s["achieved"][0]), abs(s["achieved"][1])) for s in sims)

    for ri, s in enumerate(sims):
        err_y = abs(s["achieved"][1] - s["target"][1])
        rel = err_y / max(abs(s["target"][1]), 1e-9) if abs(s["target"][1]) > 1e-6 else err_y / max(lat_range, 1e-9)
        col = OK if rel <= 0.05 else BAD
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
            if ci == 0:
                ax.text(-0.045, 0.5, f"aim  $v_y^*$ = {s['target'][1]:+.2f}",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=12.5, fontweight="bold", color="#24292f")
            if ci == 2:
                # Only the return panel carries a claim, so it is the only one annotated.
                ax.add_patch(Rectangle((0.035, 0.845), 0.40, 0.115, transform=ax.transAxes,
                                       facecolor="white", alpha=0.82, edgecolor="none", zorder=3))
                ax.text(0.05, 0.90, f"achieved ({s['achieved'][0]:+.2f}, {s['achieved'][1]:+.2f}) m/s",
                        transform=ax.transAxes, fontsize=8.4, color=col, zorder=4, va="center")
                # measured-velocity arrow, shared scale, anchored at frame centre
                dx, dy = s["achieved"][0] / v_ref, s["achieved"][1] / v_ref
                ax.annotate("", xy=(0.5 + 0.32 * dx, 0.5 - 0.32 * dy), xytext=(0.5, 0.5),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col), zorder=5)

    fig.text(m_left / fig_w, 1.0 - 0.42 / fig_h,
             f"Vector-valued steering: one controller, three commanded directions ({args.embodiment})",
             fontsize=13.5, fontweight="bold", color="#24292f", va="center")
    fig.text(m_left / fig_w, 1.0 - 0.74 / fig_h,
             "same commanded speed in every row; arrows are MEASURED outgoing velocity at one shared scale",
             fontsize=9.5, color=MUTED, va="center")

    for ext in ("png", "pdf"):
        fig.savefig(out / f"vector_frames_{args.embodiment}.{ext}", facecolor="white")
    (out / f"vector_frames_{args.embodiment}.json").write_text(json.dumps(
        {"embodiment": args.embodiment, "v_in": args.v_in, "vx_star": vx_star,
         "lateral_range": lat_range, "model_selected": best, "model_terms": list(terms),
         "cond_A": float(np.linalg.cond(W_lin[:2, :].T)),
         "rows": [{k: s[k] for k in ("target", "action", "achieved")} for s in sims]}, indent=2))
    print(f"[vecdemo] wrote {out}/vector_frames_{args.embodiment}.png", flush=True)


if __name__ == "__main__":
    main()
