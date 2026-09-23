#!/usr/bin/env python3
"""Slide figure: latent gate lift by encoder size, all 6 quantities, mid depth.

Numbers are the sim battery (jobs 20030994-20031011, 2026-07-27), read off
$SWEEP/_logs/pilot_p_*.out -- the "[a-cmd] L<mid> ... lift=" line per run.

Two panels rather than one, because the translational quantities (~0.3-0.44) and
the angular/gravity ones (~0.001-0.07) differ by an order of magnitude; a single
linear axis would flatten the angular panel into invisible stubs. Separate scales
are labelled as such on each panel.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("figures")

# dataviz reference palette, categorical slots 1-3 (light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8983"
GRID = "#e3e2dd"

MODELS = ["ViT-L (300M)", "ViT-H (600M)", "ViT-g (1B)"]

# quantity -> (L, H, g) gate lift at mid depth
PANELS = [
    ("Translational", 0.50, [
        ("velocity2d",       [0.389, 0.442, 0.410]),
        ("velocity2d_mixed", [0.366, 0.417, 0.393]),
        ("accel2d",          [0.316, 0.358, 0.332]),
    ]),
    ("Angular + gravity", 0.080, [
        ("angvel2d",   [0.052, 0.050, 0.066]),
        ("angaccel2d", [0.022, 0.032, 0.029]),
        ("gravity",    [0.001, 0.001, 0.001]),
    ]),
]


def bar_aspect(ax):
    """mutation_aspect that makes a data-unit rounding_size render circular.

    FancyBboxPatch applies rounding_size in x-data units and in y-data units
    scaled by mutation_aspect, so the corner is only visually round if the
    aspect cancels the axes' data-per-pixel ratio on each axis.
    """
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return ((y1 - y0) / (x1 - x0)) * (bbox.width / bbox.height)


def rounded_bar(ax, x, w, h, color, radius, aspect):
    """A bar with a rounded data-end, anchored square to the baseline.

    FancyBboxPatch rounds all four corners, so the rect is extended below zero
    and the axis floor at 0 clips the bottom rounding away.
    """
    drop = radius * aspect
    ax.add_patch(FancyBboxPatch(
        (x, -drop), w, h + drop,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, mutation_aspect=aspect, zorder=3,
    ))


fig, axes = plt.subplots(
    1, 2, figsize=(13.0, 5.9), dpi=220, facecolor=SURFACE,
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.16},
)

for ax, (panel_title, ymax, rows) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    n = len(rows)
    ax.set_xlim(-0.62, n - 0.38)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(n))
    ax.set_xticklabels([r[0] for r in rows], fontsize=11, color=INK)
    ax.set_ylabel("gate lift  (real − shuffled command)", fontsize=10.5, color=INK2)
    ax.set_title(panel_title, fontsize=12.5, color=INK, pad=26, loc="left", fontweight="bold")

    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9.5)

# The reversal is the point of the figure, but an arrow into the bar group
# collides with the value labels -- so it sits in the panel's empty upper right.
axes[0].text(2.44, 0.483, "ViT-g below ViT-H in all three\n(−0.032, −0.024, −0.026)",
             fontsize=10.5, color=INK2, ha="right", va="top", linespacing=1.5)
axes[1].text(2.0, 0.014, "n/a — constant per scene\n(probe R² = 1.00)",
             ha="center", va="bottom", fontsize=9.5, color=INK3, style="italic")

handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                      color=c, label=m) for c, m in zip(SERIES, MODELS)]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.062, 0.955),
           ncol=3, frameon=False, fontsize=11, handletextpad=0.5,
           columnspacing=1.8, labelcolor=INK)

fig.text(0.062, 0.975, "Bigger is not better: the 1B encoder loses to the 600M one",
         fontsize=16.5, color=INK, fontweight="bold", va="top")
fig.text(0.062, 0.135,
         "Latent gate lift = cos(predicted ΔH, true ΔH) on 50 held-out scenes, minus a shuffled-command derangement control (control ≈ 0.000).\n"
         "V-JEPA2, mid depth, decoder-free; 18 runs, sim battery 2026-07-27. The two panels use different y-scales.\n"
         "Across every run the operator recovers direction but not magnitude (relative error 0.96–1.00) — the ceiling is the linear command→edit operator, not the encoder.",
         fontsize=9.5, color=INK3, va="top", linespacing=1.6)

fig.subplots_adjust(left=0.062, right=0.985, top=0.80, bottom=0.28)

# aspect depends on the final axes geometry, so bars are drawn after layout
fig.canvas.draw()
for ax, (_, ymax, rows) in zip(axes, PANELS):
    aspect = bar_aspect(ax)
    radius = 0.016          # x-data units; rendered circular via `aspect`
    group_w, bar_w = 0.74, 0.74 / 3
    inset = 0.012           # ~2px of surface between adjacent bars
    for gi, (_label, vals) in enumerate(rows):
        for si, v in enumerate(vals):
            x = gi - group_w / 2 + si * bar_w + inset
            rounded_bar(ax, x, bar_w - 2 * inset, v, SERIES[si], radius, aspect)
            ax.text(x + (bar_w - 2 * inset) / 2, v + ymax * 0.03,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=INK2)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    p = OUT / f"model_size_gate_lift.{ext}"
    fig.savefig(p, facecolor=SURFACE)
    print("wrote", p)
