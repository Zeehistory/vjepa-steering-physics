

#!/usr/bin/env python
"""Top-3 PCA of the latent space, before and after each steering method (the Sonia figure).

Follows the conventions of the V-JEPA 2 PCA explorer (https://vid-rep-pca.netlify.app/): points are
per-temporal-token activations, a clip's tokens are joined chronologically by a polyline, and colour
encodes a physical parameter. Two views, because they answer different questions:

**Atlas view (the one that speaks to the crosstalk question).** A single PCA is fitted across the REAL
held-out clips, so every clip and every steered latent lives in one shared basis. Colouring the same
atlas twice -- once by spin, once by speed -- shows whether the two quantities occupy distinguishable
directions at all. On top of that, each steering method is drawn as an ARROW from the base clip to its
steered latent, against the arrow from the base clip to the real target clip. That is the crosstalk
result made visual: a selective velocity edit should travel along the velocity axis and leave the spin
coordinate where it was, and a leaking one should visibly drift along the spin axis.

**Per-clip view.** An independent PCA of one clip's 8 temporal tokens, which (as the explorer's author
notes) foregrounds the local geometry of a single trajectory. Here it shows what an edit does to the
*shape* of the temporal path rather than to its location in the atlas.

Latent-only: no decoder is needed, so this runs as soon as the operators are fitted. Note that this
figure shows what the edit does in LATENT space; whether those latent displacements actually render as
the intended physics is what ``crosstalk_eval.py`` measures in pixels. Both are needed -- a latent that
moves in the right direction but decodes to nothing would look perfect here.

    PYTHONPATH=. python experiments/threads/restitution-spin/11_run/pca_viz.py \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --operators_dir .../analysis/spin_ball3d/operators \
        --output_dir .../analysis/spin_ball3d/pca --layer 18
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
import json
from pathlib import Path

import sys
from pathlib import Path as _P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import spin_ops as so
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

METHOD_COLORS = {"V": "#1f77b4", "S": "#d62728", "V+S": "#9467bd", "rand_V": "#7f7f7f"}


def _clip_embed(sample, layer: int) -> np.ndarray:
    """Clip-level embedding: the per-temporal-token activations, averaged over space -> ``(n_t, D)``.

    Spatial averaging is deliberate. The raw token grid is dominated by WHERE the ball is, which within
    a scene is a shared constant and across scenes is the largest source of variance by far -- a PCA of
    it would return a map of the tabletop, not of the physics. Averaging over space leaves the temporal
    structure, which is where velocity and spin live.
    """
    arr = np.asarray(sample["layers"][layer], dtype=np.float64)   # (n_tok, D)
    n_t = int(sample["grid"][0])
    D = arr.shape[-1]
    return arr.reshape(n_t, -1, D).mean(axis=1)                   # (n_t, D)


def _pca_fit(X: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-``k`` PCA of rows of ``X``. Returns ``(mean, components (k,D), explained_ratio (k,))``."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # Gram trick: n rows (few thousand) << D (1024 here, but this keeps it right if D grows).
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(len(X) - 1, 1)
    return mu, Vt[:k], (var[:k] / var.sum())


def _proj(X: np.ndarray, mu: np.ndarray, comp: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(X) - mu) @ comp.T


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--operators_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--num_scenes", type=int, default=40)
    p.add_argument("--num_arrows", type=int, default=24)
    p.add_argument("--n_spin", type=int, default=4)
    p.add_argument("--n_vel", type=int, default=4)
    p.add_argument("--max_cached_shards", type=int, default=2)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    L = args.layer
    # Bounded shard cache, for the same reason the operator fit needs one: the default is unbounded
    # and each cached payload carries the stored FRAMES as well as the latents, so streaming 64
    # scenes accumulated ~20 GB and was OOM-killed at 32 GB.
    ds = LatentDataset(args.test_dir, layers=[L], max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]

    ops = {}
    for kind in ("vel", "spin"):
        z = np.load(Path(args.operators_dir) / f"operator_{kind}.npz")
        ops[kind] = z[f"B_{L}"].astype(np.float64)
    print(f"[pca] {len(sids)} scenes, layer {L}", flush=True)

    # -- atlas: every real clip, embedded and labelled --------------------------------------------
    rows, embeds = [], []
    cell_index: dict[int, dict[tuple[int, int], int]] = {}
    for s in sids:
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        cell_index[s] = cells
        for (vi, si), idx in cells.items():
            sam = ds[idx]
            e = _clip_embed(sam, L)
            embeds.append(e)
            rows.append({"scene": s, "vi": vi, "si": si, "idx": idx,
                         "omega": so.clip_spin(sam), "v": vo.clip_velocity(sam),
                         "n_t": e.shape[0]})
    E_raw = np.concatenate(embeds, axis=0)                 # (n_clips*n_t, D)
    n_t = rows[0]["n_t"]

    # SCENE-CENTRE before the PCA. Without this the figure is a map of the tabletop: every clip in a
    # scene shares one start position, so the leading components separate SCENES, and the physics
    # colours come out mixed inside each blob. That is a true fact about the latent space and it is
    # reported (`scene_variance_fraction` below), but it is not the question -- the operators are fitted
    # on WITHIN-scene differences, so the geometry they act in is the within-scene geometry. Subtracting
    # each scene's mean removes the start-position offset the operators never see and leaves exactly the
    # velocity x spin variation they do.
    Eb = E_raw.reshape(len(rows), n_t, -1)
    scene_of = np.array([r["scene"] for r in rows])
    E_cent = np.empty_like(Eb)
    scene_mean: dict[int, np.ndarray] = {}
    for s in np.unique(scene_of):
        m = scene_of == s
        scene_mean[int(s)] = Eb[m].mean(axis=0)            # (n_t, D)
        E_cent[m] = Eb[m] - scene_mean[int(s)][None]
    between = float(np.var(Eb.mean(axis=1), axis=0).sum())
    within = float(np.var(E_cent.mean(axis=1), axis=0).sum())
    scene_var_frac = (between - within) / max(between, 1e-12)

    E = E_cent.reshape(len(rows) * n_t, -1)
    mu, comp, evr = _pca_fit(E, k=3)
    P = _proj(E, mu, comp)
    Pc = P.reshape(len(rows), n_t, 3)
    print(f"[pca] scene identity accounts for {scene_var_frac*100:.1f}% of clip-level variance "
          f"(removed before this PCA)", flush=True)
    print(f"[pca] atlas: {len(rows)} clips x {n_t} tokens; explained {np.round(evr, 3).tolist()}",
          flush=True)

    omega = np.array([r["omega"] for r in rows])
    speed = np.array([np.linalg.norm(r["v"]) for r in rows])
    vang = np.array([np.arctan2(r["v"][1], r["v"][0]) for r in rows])

    # -- steering displacements in the SAME basis ---------------------------------------------------
    arrows: list[dict] = []
    rng = np.random.default_rng(0)
    for s in sids[: args.num_arrows]:
        if s not in cell_index:
            continue
        cells = cell_index[s]
        vi_a, si_a = 0, 0
        vi_b, si_b = args.n_vel - 1, args.n_spin - 1
        sq = so.commutation_square(cells, vi_a, si_a, vi_b, si_b)
        sam = {k: ds[i] for k, i in sq.items()}
        va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
        wa, wb = so.clip_spin(sam["base"]), so.clip_spin(sam["spin_only"])
        fv = vo.command_features_pos(va, vb, vo.clip_start_pos(sam["base"]))
        fs = so.spin_command_features(wa, wb, so.clip_phi0(sam["base"]))

        base_flat = np.asarray(sam["base"]["layers"][L], dtype=np.float64)
        n_tok, D = base_flat.shape
        eV = (fv @ ops["vel"]).reshape(n_tok, D)
        eS = (fs @ ops["spin"]).reshape(n_tok, D)
        r = rng.standard_normal((n_tok, D))
        eR = r * (np.linalg.norm(eV) / (np.linalg.norm(r) + 1e-12))

        # Same scene-centring as the atlas, so steered latents land in the same coordinates as the
        # real clips they are being compared against.
        sm = scene_mean[int(s)]

        def emb_from(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(n_t, -1, D).mean(axis=1) - sm

        entry = {"scene": s, "base": _proj(emb_from(base_flat), mu, comp)}
        for name, ed in (("V", eV), ("S", eS), ("V+S", eV + eS), ("rand_V", eR)):
            entry[name] = _proj(emb_from(base_flat + ed), mu, comp)
        for name, key in (("gt_vel", "vel_only"), ("gt_spin", "spin_only"), ("gt_both", "both")):
            entry[name] = _proj(emb_from(np.asarray(sam[key]["layers"][L], dtype=np.float64)),
                                mu, comp)
        arrows.append(entry)

    # -- figure -------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.24)

    def atlas_panel(ax, colour, label, cmap):
        for i in range(len(rows)):
            ax.plot(Pc[i, :, 0], Pc[i, :, 1], "-", color="0.85", lw=0.4, zorder=1)
        sc = ax.scatter(Pc[:, :, 0].ravel(), Pc[:, :, 1].ravel(),
                        c=np.repeat(colour, n_t), s=5, cmap=cmap, zorder=2)
        plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.02)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")

    ax = fig.add_subplot(gs[0, 0])
    atlas_panel(ax, omega, "spin $\\omega$ (rad/frame)", "RdYlBu_r")
    ax.set_title("Atlas coloured by SPIN\n(real held-out clips, tokens joined chronologically)")

    ax = fig.add_subplot(gs[0, 1])
    atlas_panel(ax, speed, "image speed |v|", "viridis")
    ax.set_title("Same atlas coloured by SPEED")

    ax = fig.add_subplot(gs[0, 2])
    atlas_panel(ax, vang, "velocity heading (rad)", "twilight")
    ax.set_title("Same atlas coloured by HEADING")

    # steering arrows, PC1-PC2 and PC1-PC3
    for col, (a, b, nm) in enumerate(((0, 1, "PC1-PC2"), (0, 2, "PC1-PC3"))):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(P[:, a], P[:, b], c="0.88", s=4, zorder=1)
        for e in arrows:
            o = e["base"].mean(axis=0)
            for name in ("V", "S", "V+S", "rand_V"):
                d = e[name].mean(axis=0)
                ax.annotate("", xy=(d[a], d[b]), xytext=(o[a], o[b]),
                            arrowprops=dict(arrowstyle="->", color=METHOD_COLORS[name],
                                            lw=1.3, alpha=0.85), zorder=3)
            for name, c in (("gt_vel", "#1f77b4"), ("gt_spin", "#d62728")):
                d = e[name].mean(axis=0)
                ax.annotate("", xy=(d[a], d[b]), xytext=(o[a], o[b]),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.1,
                                            alpha=0.55, linestyle=":"), zorder=2)
        ax.set_xlabel(f"PC{a+1}"); ax.set_ylabel(f"PC{b+1}")
        ax.set_title(f"Steering displacements ({nm})\nsolid = fitted operator, dotted = real target clip")
        if col == 0:
            handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in METHOD_COLORS.items()]
            ax.legend(handles=handles, fontsize=8, loc="best")

    # per-clip local geometry
    ax = fig.add_subplot(gs[1, 2])
    e = arrows[0]
    local = np.concatenate([e["base"]] + [e[k] for k in ("V", "S", "V+S")], axis=0)
    lmu, lcomp, levr = _pca_fit(local, k=2)
    for name, c in (("base", "k"), ("V", METHOD_COLORS["V"]), ("S", METHOD_COLORS["S"]),
                    ("V+S", METHOD_COLORS["V+S"])):
        q = _proj(e[name], lmu, lcomp)
        ax.plot(q[:, 0], q[:, 1], "-o", color=c, ms=3, lw=1.2, label=name)
    ax.set_title(f"Per-clip temporal trajectory (scene {e['scene']})\nindependent PCA of this clip's tokens")
    ax.set_xlabel("local PC1"); ax.set_ylabel("local PC2"); ax.legend(fontsize=8)

    fig.suptitle(f"Latent geometry of velocity vs spin steering — layer {L}, {len(rows)} held-out clips "
                 f"(scene-centred: start position accounted for {scene_var_frac*100:.0f}% of raw variance)",
                 fontsize=13)
    fig.savefig(out / "pca_steering.png", dpi=140, bbox_inches="tight")
    print(f"[pca] wrote {out/'pca_steering.png'}", flush=True)

    # -- quantitative companion: how much does each edit move ALONG each quantity's axis? ------------
    # The picture is suggestive; these numbers are what can be quoted. Axes are the atlas directions
    # that best predict omega and |v| (a 1-D linear probe in PC space), so "along the spin axis" has a
    # definition rather than being read off by eye.
    Pm = Pc.mean(axis=1)                                   # (n_clips, 3) clip-level PC coords
    def axis_for(y):
        A = np.concatenate([Pm, np.ones((len(Pm), 1))], axis=1)
        w = np.linalg.lstsq(A, y, rcond=None)[0][:3]
        return w / (np.linalg.norm(w) + 1e-12)
    ax_w, ax_v = axis_for(omega), axis_for(speed)

    # VELOCITY IS A 2-VECTOR, and a scalar "speed axis" cannot score a velocity edit. The commanded
    # change alters heading as well as magnitude, and heading is a different latent direction -- which
    # is why most of BOTH the operator's and the real clip's displacement lands OFF the (speed, spin)
    # plane. Scoring a vector edit by its projection onto |v| alone gave a NEGATIVE "velocity gain"
    # that was measuring the basis, not the operator. So velocity gets a 2-D basis, one axis per
    # component, and is scored against the real velocity change in it.
    ax_vx = axis_for(np.array([r["v"][0] for r in rows]))
    ax_vy = axis_for(np.array([r["v"][1] for r in rows]))
    M3 = np.stack([ax_vx, ax_vy, ax_w], axis=1)            # (3, 3): [vx, vy, spin]

    def _decomp3(name: str) -> np.ndarray:
        d = np.array([e[name].mean(axis=0) - e["base"].mean(axis=0) for e in arrows])
        return np.linalg.lstsq(M3, d.T, rcond=None)[0]     # (3, n_arrows)

    ref_vel = np.median(_decomp3("gt_vel")[:2], axis=1)    # the real velocity change, in (vx, vy)
    ref_spin = float(np.median(_decomp3("gt_spin")[2]))    # the real spin change, on the spin axis
    vector_stats = {}
    for _name in ("V", "S", "V+S", "rand_V", "gt_vel", "gt_spin"):
        c = _decomp3(_name)
        vv = np.median(c[:2], axis=1)
        vector_stats[_name] = {
            # projection onto the REAL velocity change, so motion in a wrong direction cannot score
            "vel_gain": float(vv @ ref_vel / (ref_vel @ ref_vel + 1e-12)),
            "vel_leak_vs_gt": float(np.linalg.norm(vv) / (np.linalg.norm(ref_vel) + 1e-12)),
            "spin_gain": float(np.median(c[2]) / (ref_spin + 1e-12)),
        }

    # The two SCALAR probe axes are NOT orthogonal (they typically sit ~70 deg apart), so a bare dot product
    # onto each would charge part of a pure speed displacement to the spin axis and vice versa --
    # manufacturing apparent crosstalk out of the basis. Decompose jointly instead: least-squares
    # coefficients (a, b) in ``d ~= a * speed_axis + b * spin_axis``, which is the oblique projection
    # and reduces to the dot products only when the axes happen to be orthogonal.
    M = np.stack([ax_v, ax_w], axis=1)                     # (3, 2)
    stats = {"explained_ratio": evr.tolist(), "scene_variance_fraction": float(scene_var_frac),
             "spin_axis": ax_w.tolist(), "speed_axis": ax_v.tolist(),
             "cos_spin_speed_axes": float(ax_w @ ax_v),
             "axes_angle_deg": float(np.degrees(np.arccos(np.clip(ax_w @ ax_v, -1, 1)))),
             "n_clips": len(rows)}
    for name in ("V", "S", "V+S", "rand_V", "gt_vel", "gt_spin"):
        d = np.array([e[name].mean(axis=0) - e["base"].mean(axis=0) for e in arrows])
        coef = np.linalg.lstsq(M, d.T, rcond=None)[0]      # (2, n_arrows): [speed, spin]
        resid = d.T - M @ coef
        stats[name] = {"along_speed_axis": float(np.median(coef[0])),
                       "along_spin_axis": float(np.median(coef[1])),
                       "off_axis_norm": float(np.median(np.linalg.norm(resid, axis=0))),
                       "norm": float(np.median(np.linalg.norm(d, axis=1)))}
    # Selectivity, stated the way the question was asked: what fraction of the spin displacement that a
    # REAL spin change produces does the velocity edit produce as a side effect, and vice versa?
    stats["vector_basis"] = vector_stats
    # The headline selectivity numbers, in the well-posed basis: velocity scored as a 2-vector against
    # the real velocity change, spin scored against the real spin change.
    stats["latent_crosstalk_vector"] = {
        "V_vel_gain": vector_stats["V"]["vel_gain"],
        "V_spin_leak": vector_stats["V"]["spin_gain"],
        "S_spin_gain": vector_stats["S"]["spin_gain"],
        "S_vel_leak": vector_stats["S"]["vel_leak_vs_gt"],
        "rand_vel_leak": vector_stats["rand_V"]["vel_leak_vs_gt"],
        "rand_spin_leak": vector_stats["rand_V"]["spin_gain"],
    }
    stats["latent_crosstalk_scalar_speed_axis_DEPRECATED"] = {
        "V_spin_leak_vs_gt_spin": float(stats["V"]["along_spin_axis"]
                                        / (stats["gt_spin"]["along_spin_axis"] + 1e-12)),
        "S_speed_leak_vs_gt_vel": float(stats["S"]["along_speed_axis"]
                                        / (stats["gt_vel"]["along_speed_axis"] + 1e-12)),
        "V_speed_gain": float(stats["V"]["along_speed_axis"]
                              / (stats["gt_vel"]["along_speed_axis"] + 1e-12)),
        "S_spin_gain": float(stats["S"]["along_spin_axis"]
                             / (stats["gt_spin"]["along_spin_axis"] + 1e-12)),
    }
    (out / "pca_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2)[:1200], flush=True)


if __name__ == "__main__":
    main()
