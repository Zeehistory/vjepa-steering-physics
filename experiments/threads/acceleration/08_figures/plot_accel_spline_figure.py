#!/usr/bin/env python
"""Figures + a web payload for spline-in-time acceleration steering.

Reads whatever of the three artifact sets exists and renders the panels it can, so it is safe to run
before the decode finishes (geometry-only) and again afterwards (full):

  spline_geom/spline_geometry.json, traj_pca.json   latent geometry (CPU, no decoder)
  spline/spline_operator_meta.json                  held-out latent gate per knot count
  steer_spline/steer2d_summary.json, trajectories.json   decoded pixel results

Panels
  fig1_why_constant_fails.png   per-token edit magnitude + the K-knot representational ceiling
  fig2_knot_ablation.png        decoded angle error vs knot count, against the standing baselines
  fig3_decoded_paths.png        decoded ball trajectories: anchor vs steered vs ground-truth target
  fig4_family_loop.png          the scene's acceleration family as a loop in latent PCA space

Also writes ``web_payload.json``: everything the interactive page needs, already reduced to plot-ready
arrays so the page carries no analysis logic of its own.

    python experiments/threads/acceleration/08_figures/plot_accel_spline_figure.py \
        --analysis_dir outputs/analysis/moving_ball_accel2d_mixed --output_dir .../spline_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Standing bar on this dataset (100 held-out test scenes, decoded-vs-target accel angle error).
# Sources: outputs/analysis/moving_ball_accel2d_mixed/steer_*/steer2d_summary.json and
# steer_decopt_free (test-time optimization, n=30).
BASELINES = {
    "no-op floor (random subspace)": 48.14,
    "plain linear operator (ridge_global)": 47.60,
    "best prior linear operator (cmd_U8)": 13.36,
    "per-pair oracle (full_delta, needs H_b)": 10.64,
    "decoder-in-the-loop TTO (n=30)": 5.07,
}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fig_why_constant_fails(geom, meta, out: Path) -> dict:
    """Left: the edit's magnitude is not constant over the clip. Right: what K knots can express."""
    if not geom:
        return {}
    L = str(geom["layers"][0])
    g = geom["per_layer"][L]
    norms = np.asarray(g["edit_norm_per_t_mean"], dtype=float)
    ceil = {int(k): v for k, v in g["knot_cos_ceiling"].items()}
    Ks = sorted(ceil)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(np.arange(len(norms)), norms, "o-", lw=2, color="#2b6cb0")
    ax.axhline(norms.mean(), ls="--", color="#a0aec0",
               label=f"what a constant-in-t edit assumes ({norms.mean():.1f})")
    ax.set_xlabel("temporal token t  (each spans 2 video frames)")
    ax.set_ylabel(r"$\|\Delta H_t\|$")
    ax.set_title(f"The acceleration edit grows over the clip\n"
                 f"L{L}: {g['edit_norm_growth_last_over_first']}x from t=0 to t=7")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ks, [ceil[k] for k in Ks], "o-", lw=2, color="#805ad5")
    ax.axhline(1.0, ls=":", color="#a0aec0")
    ax.scatter([1], [ceil[1]], s=110, zorder=5, color="#e53e3e")
    ax.annotate(f"K=1 is the classical\nglobal steering vector\ncos={ceil[1]:.3f}",
                xy=(1, ceil[1]), xytext=(1.9, min(ceil.values()) + 0.03), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#e53e3e"))
    ax.set_xlabel("K = spline control points over time")
    ax.set_ylabel(r"cos(K-knot fit, true $\Delta R$)")
    ax.set_title("How much of the edit a K-knot edit can\nEXPRESS at all (representational ceiling)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig1_why_constant_fails.png", dpi=140); plt.close(fig)
    return {"edit_norm_per_t": norms.tolist(), "knot_ceiling": {str(k): ceil[k] for k in Ks},
            "layer": L, "growth": g["edit_norm_growth_last_over_first"]}


def fig_knot_ablation(steer, out: Path, calib=None) -> dict:
    """Decoded angle error as a function of temporal degrees of freedom, against the standing bar.

    Prefers the LEAKAGE-FREE numbers when a calibration is available: gain chosen on a validation half,
    error reported on the disjoint half. Choosing the gain on the same scenes being reported would bias
    every arm, and unevenly — the constant-in-time arm peaks at a much larger gain than the shaped ones.
    """
    if not steer:
        return {}
    res = steer["results"]
    Ks = sorted(steer["knots"])
    gains = steer["gains"]
    best, leakage_free = {}, False
    cal = (calib or {}).get("calibrated") or {}
    if cal:
        leakage_free = True
        for K in Ks:
            r = cal.get(f"spline_K{K}")
            if r and np.isfinite(r.get("test_angle_err_deg", np.nan)):
                best[K] = (r["test_angle_err_deg"], r["chosen_gain"])
    if not best:
        for K in Ks:
            cand = [(res[f"spline_K{K}_s{g:g}"]["angle_err_deg"], g) for g in gains
                    if f"spline_K{K}_s{g:g}" in res
                    and np.isfinite(res[f"spline_K{K}_s{g:g}"]["angle_err_deg"])]
            if cand:
                best[K] = min(cand)
    if not best:
        return {}
    ung = (calib or {}).get("ungained") or {}
    proj = {K: (ung.get(f"proj_K{K}", {}).get("test_angle_err_deg")
                if leakage_free and f"proj_K{K}" in ung else
                res[f"proj_K{K}"]["angle_err_deg"])
            for K in Ks if f"proj_K{K}" in res}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ks = sorted(best)
    ax.plot(ks, [best[k][0] for k in ks], "o-", lw=2.2, color="#2b6cb0",
            label="spline operator (command-only, no $H_b$)")
    if proj:
        pk = sorted(proj)
        ax.plot(pk, [proj[k] for k in pk], "s--", lw=1.6, color="#805ad5",
                label=r"K-knot fit of the TRUE edit (uses $H_b$)")
    for name, val in BASELINES.items():
        ax.axhline(val, ls=":", lw=1.2, color="#718096")
        ax.text(ks[-1], val, f" {name} ({val}°)", va="center", fontsize=7, color="#4a5568")
    ax.set_xlabel("K = temporal degrees of freedom  (K=1 is constant in time)")
    ax.set_ylabel("decoded-vs-target acceleration angle error (deg)")
    ax.set_title("Does giving the edit a shape in TIME help?"
                 + ("\n(gain chosen on a held-out half)" if leakage_free else ""))
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig2_knot_ablation.png", dpi=140); plt.close(fig)
    return {"best_per_knot": {str(k): {"angle_err_deg": best[k][0], "gain": best[k][1]} for k in ks},
            "proj_per_knot": {str(k): proj[k] for k in proj if proj[k] is not None},
            "leakage_free": leakage_free, "baselines": BASELINES}


def _best_arm(steer, prefix: str) -> str | None:
    """Name of the lowest-angle-error arm whose method name starts with ``prefix``."""
    if not steer:
        return None
    res = steer.get("results", {})
    cand = [(v["angle_err_deg"], m) for m, v in res.items()
            if m.startswith(prefix) and np.isfinite(v.get("angle_err_deg", np.nan))]
    return min(cand)[1] if cand else None


def fig_decoded_paths(traj, out: Path, steer=None) -> dict:
    """The actual ask: how does the ball's PATH change when you steer acceleration?"""
    if not traj:
        return {}
    scenes = sorted(traj)[:4]
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.5 * len(scenes), 3.8), squeeze=False)
    payload = {}
    for j, s in enumerate(scenes):
        d = traj[s]
        ax = axes[0][j]
        series = [("gt_anchor", "#a0aec0", "anchor clip (GT)"),
                  ("gt_target", "#111827", "target clip (GT)"),
                  ("noop", "#e53e3e", "unsteered"),
                  ("full_delta", "#38a169", "oracle edit")]
        # pick each family's BEST gain arm, so the comparison is best-vs-best rather than arbitrary
        const_key = _best_arm(steer, "spline_K1_") or next(
            (k for k in d if k.startswith("spline_K1_")), None)
        spline_key = None
        if steer:
            cand = [(v["angle_err_deg"], m) for m, v in steer.get("results", {}).items()
                    if m.startswith("spline_K") and not m.startswith("spline_K1_")
                    and m in d and np.isfinite(v.get("angle_err_deg", np.nan))]
            spline_key = min(cand)[1] if cand else None
        if spline_key is None:
            spline_key = next((k for k in d if k.startswith("spline_K")
                               and not k.startswith("spline_K1_")), None)
        if spline_key:
            series.append((spline_key, "#2b6cb0", f"spline steer ({spline_key})"))
        if const_key and const_key in d:
            series.append((const_key, "#dd6b20", f"constant-in-t ({const_key})"))
        for key, color, label in series:
            arr = d.get(key)
            if not arr:
                continue
            a = np.asarray([[np.nan if v is None else v for v in row] for row in arr], dtype=float)
            ax.plot(a[:, 0], a[:, 1], "-o", ms=2.5, lw=1.5, color=color, label=label, alpha=0.9)
            payload.setdefault(s, {})[key] = np.where(np.isnan(a), None, a).tolist()
        ax.invert_yaxis()          # image coords: y grows downward
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y" if j == 0 else "")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=6.5, loc="best")
    fig.suptitle("Decoded ball trajectories — curvature IS the acceleration", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig3_decoded_paths.png", dpi=140); plt.close(fig)
    return payload


def fig_family_loop(pca, geom, out: Path) -> dict:
    """The scene's 8 clips sweep acceleration direction through a full turn — a loop, not a ramp."""
    if not pca:
        return {}
    scenes = sorted(pca)[:3]
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.7 * len(scenes), 3.8), squeeze=False)
    payload = {}
    for j, s in enumerate(scenes):
        d = pca[s]
        coords = np.asarray(d["traj_pca"], dtype=float)      # (M, T, 2)
        ax = axes[0][j]
        cmap = plt.get_cmap("twilight")
        for i in range(coords.shape[0]):
            c = cmap(i / coords.shape[0])
            ax.plot(coords[i, :, 0], coords[i, :, 1], "-", lw=1.4, color=c, alpha=0.9)
            ax.scatter(coords[i, 0, 0], coords[i, 0, 1], s=14, color=c)
        ends = coords[:, -1, :]
        ax.plot(np.append(ends[:, 0], ends[0, 0]), np.append(ends[:, 1], ends[0, 1]),
                "k--", lw=1.0, alpha=0.5)
        ax.set_title(f"{s}\n8 clips, accel direction swept", fontsize=8.5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2" if j == 0 else "")
        ax.grid(alpha=0.25)
        payload[s] = {"traj_pca": coords.round(4).tolist(),
                      "accel_angle_deg": d.get("accel_angle_deg"),
                      "accel_mag": d.get("accel_mag")}
    if geom:
        L = str(geom["layers"][0])
        g = geom["per_layer"][L]
        fig.suptitle(f"Latent temporal trajectories per clip (each line = one clip's 8 tokens).  "
                     f"Family turning angle {g['family_turn_angle_deg_mean']}° · "
                     f"leave-one-out: spline {g['loo_rel_err_spline']} vs line {g['loo_rel_err_lerp']}",
                     fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig4_family_loop.png", dpi=140); plt.close(fig)
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--extra_steer", default="",
                   help="another steer2d_summary.json (e.g. an extended gain sweep) whose results are "
                        "merged in, so every K is compared at its OWN optimum rather than at the edge "
                        "of a truncated sweep")
    args = p.parse_args()

    ana = Path(args.analysis_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    geom = _load(ana / "spline_geom" / "spline_geometry.json")
    pca = _load(ana / "spline_geom" / "traj_pca.json")
    meta = _load(ana / "spline" / "spline_operator_meta.json")
    steer = _load(ana / "steer_spline" / "steer2d_summary.json")
    if args.extra_steer and steer:
        extra = _load(Path(args.extra_steer))
        if extra:
            n_before = len(steer["results"])
            steer["results"].update(extra.get("results", {}))
            steer["gains"] = sorted(set(steer.get("gains", [])) | set(extra.get("gains", [])))
            for k, v in (extra.get("per_scene") or {}).items():
                steer.setdefault("per_scene", {}).setdefault(k, {}).update(v)
            print(f"[fig] merged {len(steer['results']) - n_before} extra arms from {args.extra_steer}")
    traj = _load(ana / "steer_spline" / "trajectories.json")
    calib = _load(ana / "steer_spline" / "calib_spline_gain.json")
    print(f"[fig] geom={bool(geom)} pca={bool(pca)} meta={bool(meta)} steer={bool(steer)} traj={bool(traj)}")

    payload = {
        "geometry": fig_why_constant_fails(geom, meta, out),
        "ablation": fig_knot_ablation(steer, out, calib),
        "paths": fig_decoded_paths(traj, out, steer),
        "family": fig_family_loop(pca, geom, out),
        "latent_gate": (meta or {}).get("per_knot", {}),
        "family_geometry": ((geom or {}).get("per_layer", {}) or {}).get(
            str((geom or {}).get("layers", [12])[0]), {}),
        "calibration": calib,
        "steer_results": (steer or {}).get("results", {}),
        "family_results": (steer or {}).get("family", {}),
        "n_scenes": (steer or {}).get("n_scenes"),
        "baselines": BASELINES,
    }
    (out / "web_payload.json").write_text(json.dumps(payload, indent=1))
    print(f"[fig] wrote figures + web_payload.json -> {out}")


if __name__ == "__main__":
    main()
