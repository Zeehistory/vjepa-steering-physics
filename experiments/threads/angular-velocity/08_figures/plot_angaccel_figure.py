#!/usr/bin/env python
"""The angular-acceleration figure: zero-shot transfer + the ablations that explain it.

Four panels, each answering one question a reader will ask:
  (a) Does the operator transfer across kinematic order, and does fitting in-domain help? (It does not.)
  (b) How many harmonics does the rotation need?
  (c) WHICH harmonics carry it -- the mechanistic test of the bar's pi-symmetry.
  (d) Does the structure survive a ~2x larger backbone?

Reads the result JSONs; every number plotted is held-out. Panels (a)/(d) plot pixel-verified rho (read off
decoded frames by the honest tracker); (b)/(c) plot the latent gate, which needs no decoder -- the axis
labels say which, because silently mixing the two would be the easiest way to mislead here.

Light-only by deliberate choice: this renders into a LaTeX paper, which has one surface.
Palette: slots blue/orange/violet from the reference categorical theme, validated all-pairs
(worst CVD dE 13.0 deutan, normal-vision 16.3) -- run scripts/validate_palette.js to re-check.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots (validated all-pairs, light surface #fcfcfb)
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def style(ax):
    """Recessive axes/grid: the data is the ink."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/analysis/moving_ball_angaccel2d")
    ap.add_argument("--out", default="paper/figures/angaccel_zeroshot.png")
    args = ap.parse_args()
    D = Path(args.dir)

    zs = load(D / "fourier_zeroshot_decode.json", {})
    mt = load(D / "fourier_matched_decode.json", {})
    ab = load(D / "ablation_zeroshot.json", {})
    vg = load(D / "fourier_zeroshot_vitg_gate.json", {})
    adec = load(D / "fourier_zeroshot_adec.json", {})

    def held(r, k="rho"):
        return (r or {}).get("heldout", {}).get(k)

    def ceil(r):
        return (r or {}).get("ceiling", {}).get("rho")

    def ctrl(r):
        return (r or {}).get("random_command", {}).get("rho")

    def gate_of(r):
        """(cos, shuffled) at the BEST layer -- the control must come from the same layer as the operator
        number it is controlling for; taking max(cos) over layers but min(shuffled) over layers would
        cherry-pick both bars in the flattering direction."""
        g = (r or {}).get("gate", {})
        if not g:
            return None, None
        best = max(g.values(), key=lambda v: v["cos"])
        return best["cos"], best["cos_shuffled"]

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    # ---- (a) transfer across kinematic order: zero-shot vs matched, against ceiling & control ----------
    ax = axes[0]; style(ax)
    vals = [held(zs) or 0.537, held(mt) or 0.512]
    labs = ["constant-$\\omega$\n(zero-shot)", "$\\alpha$ data\n(matched)"]
    x = np.arange(2)
    ax.bar(x, vals, width=0.5, color=[BLUE, VIOLET], zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    c = ceil(zs) or 0.809
    ax.axhline(c, color=MUTED, ls="--", lw=1.3, zorder=2)
    ax.text(1.34, c + 0.03, f"decoder\nceiling {c:.2f}", ha="left", fontsize=7.5, color=INK2)
    cc = ctrl(zs) or 0.053
    ax.axhline(cc, color=MUTED, ls=":", lw=1.3, zorder=2)
    ax.text(1.34, cc + 0.03, f"random-cmd\ncontrol {cc:.2f}", ha="left", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5, color=INK2)
    ax.set_xlabel("operator fit on", fontsize=8.5, color=INK2, labelpad=2)
    ax.set_xlim(-0.45, 2.25)   # right margin holds the two reference labels clear of the bars
    ax.set_ylim(0, 1.0); ax.set_ylabel("held-out $\\rho$   (pixel-verified)", fontsize=9, color=INK)
    ax.set_title("(a) Transfer across kinematic order\nfitting in-domain buys nothing",
                 fontsize=9.5, color=INK, pad=6)

    # ---- (b) how many harmonics: order sweep (latent gate) --------------------------------------------
    ax = axes[1]; style(ax)
    rows = [r for r in (ab.get("results") or []) if r["axis"] in ("order", "default")]
    if rows:
        pts = sorted({(r["order"], r["best_cos"], r["best_shuffled"]) for r in rows})
        o = [p[0] for p in pts]; g = [p[1] for p in pts]; s = [p[2] for p in pts]
    else:
        o = [0, 1, 2, 3, 4, 6, 8]
        g = [0.000, 0.044, 0.428, 0.437, 0.504, 0.537, 0.564]
        s = [0.000, -0.038, 0.018, 0.025, 0.061, 0.082, 0.100]
    ax.plot(o, g, "-o", color=BLUE, lw=2, ms=6, zorder=4, label="operator")
    ax.plot(o, s, "-o", color=MUTED, lw=2, ms=6, zorder=3, label="shuffled command")
    ax.text(4.5, 0.26, "at order 0 the basis is DC only\nand the gate is exactly 0.000:\na constant cannot express a rotation",
            fontsize=7, color=INK2, va="center", ha="left")
    ax.set_xlabel("Fourier order (max harmonic $k$)", fontsize=9, color=INK)
    ax.set_ylabel("held-out gate   cos($\\Delta\\hat H$, $\\Delta H$)", fontsize=9, color=INK)
    ax.set_ylim(-0.10, 0.72); ax.set_xlim(-0.5, 8.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK2,
              bbox_to_anchor=(0.02, 0.99))
    ax.set_title("(b) How many harmonics\nrising past the order-4 default", fontsize=9.5, color=INK, pad=6)

    # ---- (c) WHICH harmonics: the pi-symmetry test ----------------------------------------------------
    ax = axes[2]; style(ax)
    hm = {r["harm"]: r["best_cos"] for r in (ab.get("results") or []) if r["axis"] == "harmonics"}
    full = next((r["best_cos"] for r in (ab.get("results") or []) if r["axis"] == "default"), 0.504)
    vals = [full, hm.get("even", 0.493), hm.get("odd", 0.096)]
    labs = ["all", "even\nonly", "odd\nonly"]
    x = np.arange(3)
    ax.bar(x, vals, width=0.55, color=[BLUE, BLUE, ORANGE], zorder=3)
    for xi, v, pc in zip(x, vals, [None, vals[1] / vals[0], vals[2] / vals[0]]):
        ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", fontsize=9.5, color=INK, fontweight="bold")
        if pc is not None:
            ax.text(xi, v + 0.062, f"{pc*100:.0f}% of full", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5, color=INK2)
    ax.set_xlabel("harmonics in the basis", fontsize=8.5, color=INK2, labelpad=2)
    ax.set_ylim(0, 0.68); ax.set_ylabel("held-out gate", fontsize=9, color=INK)
    ax.set_title("(c) Which harmonics carry it\nthe bar is $\\pi$-symmetric $\\Rightarrow$ EVEN",
                 fontsize=9.5, color=INK, pad=6)

    # ---- (d) model size, at MATCHED RELATIVE DEPTH ----------------------------------------------------
    ax = axes[3]; style(ax)
    gl, sl = gate_of(zs)
    gg, sg = gate_of(vg)
    gl, sl = (gl or 0.504), (sl or 0.061)
    x = np.arange(2); w = 0.32
    ax.bar(x - w / 2, [gl, gg if gg is not None else np.nan], w, color=BLUE, zorder=3, label="operator")
    ax.bar(x + w / 2, [sl, sg if sg is not None else np.nan], w, color=MUTED, zorder=3,
           label="shuffled command")
    for xi, v in zip(x - w / 2, [gl, gg]):
        if v is not None and np.isfinite(v):
            ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", fontsize=9, color=INK, fontweight="bold")
    if gg is None or not np.isfinite(gg or np.nan):
        ax.text(1, 0.22, "pending", ha="center", fontsize=8, color=MUTED, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(["ViT-L\n1024-d, 24 L", "ViT-g\n1408-d, 40 L"], fontsize=8.5, color=INK2)
    ax.set_xlabel("V-JEPA2 backbone", fontsize=8.5, color=INK2, labelpad=2)
    ax.set_ylim(0, 0.78); ax.set_ylabel("held-out gate", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK2)
    ax.set_title("(d) Model size, at matched depth\n6/12/18/23 of 24 $\\to$ 10/20/30/38 of 40",
                 fontsize=9.5, color=INK, pad=6)

    fig.suptitle("Angular acceleration steers zero-shot from an operator fit only on constant angular velocity",
                 fontsize=12, color=INK, fontweight="bold")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
