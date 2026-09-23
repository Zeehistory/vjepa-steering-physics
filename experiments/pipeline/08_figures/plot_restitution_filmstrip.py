#!/usr/bin/env python
"""The restitution proof-by-eye: one scene, four rows, the frames around the wall contact.

Rows, top to bottom:
    GT video of the ANCHOR clip (low e)          -- what the ball actually did
    decode(H_a)                                  -- unedited latent through the frozen decoder
    decode(H_a + edit)                           -- command-only steer toward e_b (no H_b used)
    GT video of the TARGET clip (high e)         -- what a ball with e_b actually does

The two GT rows are pixel-identical up to the contact frame by construction (the generator integrates
the same trajectory and only the rebound differs), so any visible difference in the top two rows before
contact would be a decoder artifact, and any difference in the steered row before contact would be a
specificity failure. The rebound height after contact is the whole claim.

Below the strip: the tracked wall-normal (y) trajectory for all four, which is what the numbers in the
main figure are computed from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e3e2dd"


def track_y(clip: np.ndarray) -> np.ndarray:
    """Dark-pixel centroid y per frame, in [0,1] with y increasing downward (image convention)."""
    g = clip.mean(1) if clip.ndim == 4 else clip
    dark = np.clip(1.0 - g, 0, None)
    dark = np.where(dark > 0.5, dark, 0.0)
    ys = np.linspace(0, 1, dark.shape[1])[None, :, None]
    m = dark.sum((1, 2))
    return np.where(m > 1e-6, (dark * ys).sum((1, 2)) / np.maximum(m, 1e-6), np.nan)


def show(ax, frame):
    img = np.transpose(np.clip(frame.astype(np.float32), 0, 1), (1, 2, 0))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="viz_sceneXXXXX_rN.npz written by decopt3")
    ap.add_argument("--out", default="paper/figures/restitution_filmstrip.png")
    ap.add_argument("--window", type=int, default=4, help="frames shown either side of contact")
    ap.add_argument("--shape", default="outputs/restitution/audit/shape_audit.json",
                    help="shape audit JSON: its numbers caption the visible render distortion")
    a = ap.parse_args()

    z = np.load(a.npz)
    bf = int(z["bounce_frame"]); e_a = float(z["e_a"]); e_b = float(z["e_b"])
    rows = [("GT video, $e_a$ = %.2f" % e_a, z["gt_anchor"], MUTED),
            ("decode($H_a$)  unedited", z["unsteered"], MUTED),
            ("decode($H_a$ + edit)  commanded $e_b$ = %.2f" % e_b, z["steered"], BLUE),
            ("GT video, $e_b$ = %.2f" % e_b, z["gt_target"], VIOLET)]
    T = min(r[1].shape[0] for r in rows)
    lo, hi = max(0, bf - a.window), min(T, bf + a.window + 2)
    cols = list(range(lo, hi))

    fig = plt.figure(figsize=(1.05 * len(cols) + 2.6, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(5, len(cols), height_ratios=[1, 1, 1, 1, 1.35], hspace=0.10, wspace=0.04,
                          left=0.155, right=0.995, top=0.925, bottom=0.075)
    for i, (label, clip, col) in enumerate(rows):
        for j, t in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            show(ax, clip[t])
            if i == 0:
                ax.set_title(f"t={t}" + ("  contact" if t == bf else ""), fontsize=7.5,
                             color=(ORANGE if t == bf else INK2), pad=3)
            if t == bf:
                for s in ax.spines.values():
                    s.set_color(ORANGE); s.set_linewidth(1.6)
            if j == 0:
                ax.set_ylabel(label, fontsize=8, color=col, rotation=0, ha="right", va="center",
                              labelpad=8)

    ax = fig.add_subplot(gs[4, :])
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    styles = [("GT $e_a$", MUTED, ":"), ("decode($H_a$)", MUTED, "-"),
              ("steered (ours)", BLUE, "-"), ("GT $e_b$", VIOLET, "--")]
    for (label, clip, _), (lab, col, ls) in zip(rows, styles):
        y = track_y(np.asarray(clip[:T], dtype=np.float32))
        ax.plot(np.arange(T), y, ls, color=col, lw=2.0 if lab.startswith("steered") else 1.4,
                label=lab, zorder=4 if lab.startswith("steered") else 3)
    ax.axvline(bf, color=ORANGE, lw=1.2, ls="--", zorder=2)
    ax.text(bf + 0.12, 0.02, "wall contact", fontsize=7.5, color=ORANGE, transform=
            ax.get_xaxis_transform(), va="bottom")
    ax.invert_yaxis()                                  # image y grows downward; show the bounce as a bounce
    ax.set_xlabel("frame", fontsize=9, color=INK)
    ax.set_ylabel("ball height (tracked)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, labelcolor=INK2, loc="upper center")

    fig.suptitle("The edit changes the rebound, not the approach",
                 fontsize=11.5, color=INK, fontweight="bold")
    # Be explicit about the visible cost: the decoded ball is already smeared without any edit, and the
    # edit adds to it. Numbers are the measured post-contact blob statistics, not an impression.
    cap = ("the decoder's own render distorts the ball before any edit; the edit adds to it")
    try:
        import json as _json
        sh = _json.load(open(a.shape))["post_contact_means"]
        cap = ("post-contact blob elongation (sd$_y$/sd$_x$):  GT video %.2f  ·  unedited decode %.2f  ·  "
               "render ceiling $decode(H_b)$ %.2f  ·  steered %.2f"
               % (sh["gt_video"]["elong"], sh["unsteered"]["elong"], sh["render_ceiling"]["elong"],
                  sh["steered"]["elong"]))
    except Exception:
        pass
    fig.text(0.5, 0.005, cap, ha="center", fontsize=7.5, color=INK2)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
