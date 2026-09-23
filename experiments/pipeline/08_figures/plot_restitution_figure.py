#!/usr/bin/env python
"""The restitution figure: steering a CONTACT-ONLY property, and the audit that bounds the claim.

Restitution is not a continuous state like velocity. Within a scene every clip has an identical
incoming trajectory and the coefficient e only exists after wall contact -- we verified from the
dataset's own pixels that the frames before contact are bit-identical across all 8 values of e, so e
is not a causal function of anything visible pre-contact. Linear command operators hit a hard
magnitude/ordering trade-off on it (gain 1 keeps the ordering and under-bounces, gain 1.5 nails the
magnitude and the ordering collapses). Optimizing the latent edit through the frozen decoder breaks
that trade-off.

Top row = the claim. Bottom row = the audit, including the part it does not fully pass:
  (a) commanded e vs the bounce actually rendered, 400 held-out (scene, target) pairs
  (b) against the linear operators and the H_b oracle
  (c) it holds steering DOWN, on target ranks the first run never used, and at a smaller budget
  (d) pixel distance to the dataset's own GT video of the target clip -- evidence no tracker is
      involved in at all
  (e) a SECOND tracker, never optimized against, calibrated on GT video: the ordering survives, the
      magnitude partly does not -- the steered rebound overshoots where the render ceiling does less
  (f) the same edit rendered by a decoder it was never optimized through

Every number is held-out: the edit sees only H_a and the commanded scalar e_b, never H_b.
Light-only by deliberate choice: this renders into a LaTeX paper, which has one surface.
Palette: blue/orange/violet from the same validated categorical theme as the angular-acceleration
figure, so the paper's figures read as one system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, GRID = "#fcfcfb", "#e3e2dd"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def bars(ax, labels, vals, colors, fmt="{:.3f}", fs=8.5):
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.55, color=colors, zorder=3)
    top = max([v for v in vals if np.isfinite(v)] + [1e-9])
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v + top * 0.03, fmt.format(v), ha="center", fontsize=fs, color=INK,
                    fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, color=INK2)
    ax.set_ylim(0, top * 1.30)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="outputs/restitution/audit/audit_report.json")
    ap.add_argument("--xdec", default="outputs/restitution/xdec_step5000/xdec_summary.json")
    ap.add_argument("--out", default="paper/figures/restitution_decopt.png")
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    up, down, hp = rep["steer_up"], rep["steer_down"], rep.get("hparam_variant", {})
    pix = rep["pixel_evidence_up"]
    xd = json.loads(Path(a.xdec).read_text())["results"] if Path(a.xdec).exists() else {}

    rows = []
    for d in sorted(Path(a.report).parent.parent.glob("decopt3_*")):
        f = d / "decopt_summary.json"
        if not f.exists() or d.name.endswith("smoke"):
            continue
        s = json.loads(f.read_text())
        if int(s.get("steps", 0)) != 400 or abs(float(s.get("lr", 0)) - 0.1) > 1e-9:
            continue
        if s.get("anchor_rank", 0) != 0:
            continue
        rows.extend(s.get("audit", []))

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    # ---- (a) commanded vs rendered -------------------------------------------------------------
    ax = axes[0, 0]; style(ax)
    e = np.array([r["e_b"] for r in rows]); m = np.array([r["ratio_opt"] for r in rows])
    nl = np.array([r["ratio_null"] for r in rows])
    ax.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.scatter(e, nl, s=9, color=MUTED, alpha=0.5, zorder=3, label="unedited anchor (null)")
    ax.scatter(e, m, s=11, color=BLUE, alpha=0.8, zorder=4, label="decoder-in-the-loop edit")
    t, b = up["TTO_honest_tracker"], up["_bootstrap_TTO_honest"]
    ax.text(0.32, 1.03, f"n = {t['n']} pairs, {rep['n_scenes']['up']} held-out scenes\n"
                        f"MAE {t['mae']:.3f}  CI95 [{b['mae_ci95'][0]:.3f}, {b['mae_ci95'][1]:.3f}]\n"
                        f"$\\rho$ {t['rho']:.3f}   null MAE {up['null_anchor']['mae']:.3f}",
            fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel("commanded restitution $e_b$", fontsize=9, color=INK)
    ax.set_ylabel("rendered bounce  $|v_y^{post}|/|v_y^{pre}|$", fontsize=9, color=INK)
    ax.set_xlim(0.3, 1.02); ax.set_ylim(0.25, 1.08)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK2)
    ax.set_title("(a) Command-only bounce steering\nthe edit never sees $H_b$", fontsize=9.5,
                 color=INK, pad=6)

    # ---- (b) against the linear operators ------------------------------------------------------
    ax = axes[0, 1]; style(ax)
    names = ["cmd $U_{16}$\ns=1", "cmd $U_{16}$\ns=1.5", "$\\Delta H$ oracle\n(needs $H_b$)",
             "ours"]
    maes = [0.101, 0.038, up["oracle_decode_Hb_honest"]["mae"], t["mae"]]
    rhos = [0.897, 0.329, up["oracle_decode_Hb_honest"]["rho"], t["rho"]]
    x = np.arange(4); w = 0.36
    cols = [MUTED, MUTED, VIOLET, BLUE]
    ax.bar(x - w / 2, maes, w, color=cols, zorder=3)
    ax.bar(x + w / 2, rhos, w, color=cols, alpha=0.40, zorder=3)
    for xi, v in zip(x - w / 2, maes):
        ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=8, color=INK, fontweight="bold")
    for xi, v in zip(x + w / 2, rhos):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("solid: MAE (lower better)\nfaded: $\\rho$ (higher better)", fontsize=8.5, color=INK)
    ax.set_title("(b) The linear magnitude/ordering wall\nlinear ops must trade one for the other",
                 fontsize=9.5, color=INK, pad=6)

    # ---- (c) robustness axes --------------------------------------------------------------------
    ax = axes[0, 2]; style(ax)
    trio = [("up\nranks 1,3,5,7", up["TTO_honest_tracker"], BLUE),
            ("DOWN\nranks 0,2,4,6", down.get("TTO_honest_tracker", {}), ORANGE),
            ("150 steps\nlr 0.05", hp.get("TTO_honest_tracker", {}), VIOLET)]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, [b[1].get("mae", np.nan) for b in trio], w, color=[b[2] for b in trio], zorder=3)
    ax.bar(x + w / 2, [b[1].get("rho", np.nan) for b in trio], w, color=[b[2] for b in trio],
           alpha=0.40, zorder=3)
    for xi, bb in zip(x, trio):
        if np.isfinite(bb[1].get("mae", np.nan)):
            ax.text(xi - w / 2, bb[1]["mae"] + 0.025, f"{bb[1]['mae']:.3f}", ha="center", fontsize=8,
                    color=INK, fontweight="bold")
            ax.text(xi + w / 2, bb[1]["rho"] + 0.025, f"{bb[1]['rho']:.3f}", ha="center", fontsize=8,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[0]}\nn={b[1].get('n', 0)}" for b in trio], fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1.22); ax.set_ylabel("solid: MAE · faded: $\\rho$", fontsize=8.5, color=INK)
    ax.set_title("(c) Direction, unseen ranks, budget\nthe anchor is the HARDEST bounce in 'DOWN'",
                 fontsize=9.5, color=INK, pad=6)

    # ---- (d) pixels, no tracker involved --------------------------------------------------------
    ax = axes[1, 0]; style(ax)
    mm = pix["mean_mse_to_GTb"]
    x = bars(ax, ["unsteered\n$decode(H_a)$", "steered\n(ours)", "render ceiling\n$decode(H_b)$"],
             [mm["unsteered"], mm["steered"], mm["oracle_decode"]], [MUTED, BLUE, VIOLET],
             fmt="{:.4f}")
    d = pix["steered_minus_oracle"]
    ax.text(0.03, 0.97, f"steered $-$ ceiling  {d['mean_diff']:+.4f}\n"
                        f"CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  (n={d['n']})\n"
                        f"pre-bounce drift {pix['specificity']['prebounce_drift_steered_vs_unsteered']:.4f}"
                        f"  (frame widths)",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_ylabel("pixel MSE to the GT target video", fontsize=8.5, color=INK)
    ax.set_title("(d) Toward the target video, not just the metric\nno tracker used in this panel",
                 fontsize=9.5, color=INK, pad=6)

    # ---- (e) the second tracker: where the claim is bounded ------------------------------------
    ax = axes[1, 1]; style(ax)
    grp = [("optimized\ntracker", up["TTO_honest_tracker"]["mae"], up["oracle_decode_Hb_honest"]["mae"],
            up["GTvideo_honest"]["mae"]),
           ("2nd tracker\n(independent)", up["TTO_alt_tracker"]["mae"],
            up["oracle_decode_Hb_alt"]["mae"], up["GTvideo_alt"]["mae"])]
    x = np.arange(2); w = 0.26
    ax.bar(x - w, [g[1] for g in grp], w, color=BLUE, zorder=3, label="steered (ours)")
    ax.bar(x, [g[2] for g in grp], w, color=VIOLET, zorder=3, label="render ceiling $decode(H_b)$")
    ax.bar(x + w, [g[3] for g in grp], w, color=MUTED, zorder=3, label="tracker on GT video")
    for xi, g in zip(x, grp):
        for dx, v in ((-w, g[1]), (0, g[2]), (w, g[3])):
            ax.text(xi + dx, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in grp], fontsize=8, color=INK2)
    ax.set_ylim(0, max(up["TTO_alt_tracker"]["mae"], up["oracle_decode_Hb_alt"]["mae"]) * 1.45)
    ax.set_ylabel("MAE vs commanded $e_b$", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelcolor=INK2)
    ax.text(0.5, 0.62, f"ordering survives: $\\rho$ {up['TTO_alt_tracker']['rho']:.2f}\n"
                       f"magnitude does not: the steered\nrebound reads {up['TTO_alt_tracker']['bias']:+.2f} here,\n"
                       f"the ceiling {up['oracle_decode_Hb_alt']['bias']:+.2f}",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_title("(e) What the claim does NOT survive\nthe exact magnitude is metric-specific",
                 fontsize=9.5, color=INK, pad=6)

    # ---- (f) cross-decoder ----------------------------------------------------------------------
    ax = axes[1, 2]; style(ax)
    if xd:
        labs = ["unedited\n(null)", "steered\n(ours)", "oracle $decode(H_b)$"]
        vals = [xd["B_null"]["ratio_mae"], xd["B_steered"]["ratio_mae"],
                xd["B_oracle_decode_Hb"]["ratio_mae"]]
        rh = [xd["B_null"]["rho"], xd["B_steered"]["rho"], xd["B_oracle_decode_Hb"]["rho"]]
        x = bars(ax, labs, vals, [MUTED, BLUE, VIOLET], fmt="{:.3f}")
        for xi, v, r in zip(x, vals, rh):
            ax.text(xi, v + max(vals) * 0.11, f"$\\rho$ {r:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.text(0.34, 0.97, f"edit optimized through decoder A,\nrendered by decoder B (n={xd['B_steered']['n']}).\n"
                            "B is an earlier checkpoint of the\nsame run: this bounds parameter-\n"
                            "dependence, not model-independence.",
                transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_ylabel("MAE vs commanded $e_b$, rendered by B", fontsize=8.5, color=INK)
    ax.set_title("(f) A decoder the edit never touched\nstill renders the commanded bounce",
                 fontsize=9.5, color=INK, pad=6)

    fig.suptitle("Restitution: a contact-only property is steerable command-only by optimizing the "
                 "latent edit through the frozen decoder -- and what that does not prove",
                 fontsize=12.5, color=INK, fontweight="bold")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
