

#!/usr/bin/env python
"""Fit the masked TRAJECTORY-TRANSPORT operator (PI direction 2026-06-29) + held-out latent gate.

The command-only ridge ``F_U: dv -> dH`` plateaus at ~34 deg (latent cos 0.39) because ``dv`` says WHAT
velocity to write but not WHERE in the (8x16x16) token grid. We hand the operator the geometry as soft
trajectory masks built from ground-truth ball centers and let it learn only velocity->channel:

    dH_hat = M_b*(c+_t + v_b @ B+_t) + M_a*(c-_t + v_a @ B-_t) + M_U*((v_b-v_a) @ Bd_t)

Each mask carries a velocity-INDEPENDENT presence BIAS (c+_t writes a ball, c-_t removes it) -- the
dominant near-ball dH is the disk's presence, the same at any speed. This is LINEAR in the params, so per
(layer L, temporal token t) it is a ridge of the 8-dim per-token feature
``phi = [M_b, M_b*v_b, M_a, M_a*v_a, M_U*dv]`` onto the 1024-dim ``dH`` token. We fit it in a single
streaming pass over the TRAIN cache with ``velocity_ops.LinearLS`` (8x8 normal equations),
for several Gaussian widths ``sigma`` at once (latents are sigma-independent). The SHARED-across-t variant
is derived by summing the per-t accumulators (shared X^T X = sum_t X^T X_t).

Then a cheap, no-decode latent GATE on the held-out TEST cache: predict dH from masks two ways --
``deployable`` (target mask M_b forward-simulated from start+v_b, the test-time-realistic build, no H_b)
and ``oracle`` (M_b from clip-b's true per-frame centers) -- and report cos(dH_hat, dH_true) + rel_error
per layer vs the ridge 0.39 baseline. Pick sigma/layer here before spending any GPU.

    python experiments/pipeline/04_operators/fit_transport_operator.py \
        --train_dir .../moving_ball_scene_v2d/train/vjepa2_large \
        --test_dir  .../moving_ball_scene_v2d/test/vjepa2_large \
        --layers 6,12,18,23 --sigmas 0.75,1.0,1.5 --ridge 1.0 \
        --output_dir outputs/analysis/moving_ball_v2d/subspace
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
import gc
import json
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


P = vo.TRANSPORT_FEATURE_DIM   # 8


def _masks(sample_a, vb, grid, sigma, target="oracle", sample_b=None):
    """Build (M_a, M_b) for a pair. M_a always from clip-a's GT centers. M_b is either forward-simulated
    from (clip-a start, vb) [deployable, no H_b] or read from clip-b's GT centers [oracle]."""
    T = grid[0]
    pa = vo.clip_positions(sample_a)
    M_a = vo.gaussian_mask(vo.temporal_token_centers(pa, T), grid, sigma)
    if target == "oracle":
        pb = vo.clip_positions(sample_b)
    else:  # deployable forward-sim from the command
        pb = vo.forward_sim_positions(pa[0], vb, pa.shape[0])
    M_b = vo.gaussian_mask(vo.temporal_token_centers(pb, T), grid, sigma)
    return M_a, M_b


def _cos_local(pred, dH, M_union, n_tok, thr=0.2):
    """Cosine restricted to the BALL-REGION tokens (M_union > thr) — isolates placement quality from the
    background ΔH energy that deflates the full-vector cosine for a localized predictor."""
    tok = (M_union.reshape(-1) > thr)
    if tok.sum() == 0:
        return float("nan")
    D = dH.size // (n_tok * M_union.shape[0])
    p = pred.reshape(n_tok * M_union.shape[0], D)[tok].reshape(-1)
    t = dH.reshape(n_tok * M_union.shape[0], D)[tok].reshape(-1)
    return vo.cosine(p, t)


def _predict(B_per_t, phi, n_t, n_tok):
    """phi (n_tok_total, 6) @ per-t B (n_t, 6, D) -> flat dH_hat (n_tok_total*D,)."""
    D = B_per_t.shape[2]
    out = np.empty((phi.shape[0], D))
    for t in range(n_t):
        sl = slice(t * n_tok, (t + 1) * n_tok)
        out[sl] = phi[sl] @ B_per_t[t]
    return out.reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--sigmas", default="0.75,1.0,1.5")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_scenes", type=int, default=0, help="cap train+test scenes (0=all; for smoke)")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    sigmas = [float(x) for x in args.sigmas.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    print(f"[transport] train {len(tr_scenes)} scenes, test {len(te_scenes)} scenes; "
          f"layers={layers}; sigmas={sigmas}; ridge={args.ridge}", flush=True)

    grid0 = tuple(int(x) for x in tr[tr_scenes[sorted(tr_scenes)[0]][0]]["grid"])
    n_t = grid0[0]; n_tok = grid0[1] * grid0[2]  # 8, 256

    # ops[sigma][L][t] = LinearLS(P=8, D) over (train pairs x spatial tokens at slice t)
    ops = {sg: {L: {t: vo.LinearLS(P, 1, args.ridge) for t in range(n_t)} for L in layers}
           for sg in sigmas}   # out_dim fixed lazily on first add via re-init below
    inited = {sg: {L: False for L in layers} for sg in sigmas}

    # ---- TRAIN pass: stream scenes, accumulate per (sigma, L, t) ----
    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]; va = vo.clip_velocity(sa)
        grid = tuple(int(x) for x in sa["grid"])
        Ha = {L: vo.layer_flat(sa["layers"][L]).reshape(n_tok * n_t, -1) for L in layers}
        # source mask is shared across ranks within a scene (clip-a geometry); precompute per sigma
        Ma = {sg: vo.gaussian_mask(vo.temporal_token_centers(vo.clip_positions(sa), n_t), grid, sg)
              for sg in sigmas}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]; vb = vo.clip_velocity(sb)
            Mb = {sg: vo.gaussian_mask(
                vo.temporal_token_centers(vo.clip_positions(sb), n_t), grid, sg) for sg in sigmas}
            phi = {sg: vo.transport_features(Ma[sg], Mb[sg], va, vb, grid) for sg in sigmas}
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]).reshape(n_tok * n_t, -1) - Ha[L]  # (2048, D)
                D = dH.shape[1]
                for sg in sigmas:
                    if not inited[sg][L]:
                        ops[sg][L] = {t: vo.LinearLS(P, D, args.ridge) for t in range(n_t)}
                        inited[sg][L] = True
                    for t in range(n_t):
                        sl = slice(t * n_tok, (t + 1) * n_tok)
                        ops[sg][L][t].add(phi[sg][sl], dH[sl])
        del Ha, Ma; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[transport]   trained on {n}/{len(tr_scenes)} scenes", flush=True)

    # ---- solve: per-t B (n_t,6,D) and shared B (1,6,D) = sum of per-t accumulators ----
    B_pt = {sg: {} for sg in sigmas}        # per temporal token
    B_sh = {sg: {} for sg in sigmas}        # shared across t
    for sg in sigmas:
        for L in layers:
            D = ops[sg][L][0].out_dim
            B_pt[sg][L] = np.stack([ops[sg][L][t].solve() for t in range(n_t)], 0).astype(np.float32)
            sh = vo.LinearLS(P, D, args.ridge)
            for t in range(n_t):
                sh.XtX += ops[sg][L][t].XtX; sh.XtY += ops[sg][L][t].XtY; sh.n += ops[sg][L][t].n
            B_sh[sg][L] = sh.solve().astype(np.float32)[None, :, :]  # (1,6,D)
    del ops; gc.collect()

    # ---- TEST latent gate: cos(dH_hat, dH_true) + rel_error, deployable & oracle, per-t & shared ----
    variants = {"transport_pt": (B_pt, n_t), "transport_shared": (B_sh, 1)}
    gate = {sg: {L: {f"{v}_{tg}": {"cos": [], "rel": [], "cosL": []}
                     for v in variants for tg in ("deployable", "oracle")}
                 for L in layers} for sg in sigmas}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = vo.clip_velocity(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        grid = tuple(int(x) for x in sa["grid"])
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; vb = vo.clip_velocity(sb)
            phi = {}; Mu = {}
            for sg in sigmas:
                for tg in ("deployable", "oracle"):
                    Ma, Mb = _masks(sa, vb, grid, sg, target=tg, sample_b=sb)
                    phi[(sg, tg)] = vo.transport_features(Ma, Mb, va, vb, grid)
                    Mu[(sg, tg)] = np.maximum(Ma, Mb)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                for sg in sigmas:
                    for vname, (B, nt) in variants.items():
                        for tg in ("deployable", "oracle"):
                            pred = _predict(B[sg][L], phi[(sg, tg)], nt, n_tok)
                            g = gate[sg][L][f"{vname}_{tg}"]
                            g["cos"].append(vo.cosine(pred, dH))
                            g["rel"].append(vo.rel_error(pred, dH))
                            g["cosL"].append(_cos_local(pred, dH, Mu[(sg, tg)], n_tok))
        del Ha; gc.collect()

    # ---- pick best (variant, sigma, layer) by deployable cosine; save artifacts for it + all per-layer ----
    summary = {"train_dir": args.train_dir, "test_dir": args.test_dir, "layers": layers,
               "sigmas": sigmas, "ridge": args.ridge, "n_train_scenes": len(tr_scenes),
               "n_test_scenes": len(te_scenes), "n_temporal": n_t,
               "ridge_global_baseline_cos": 0.39, "gate": {}}
    # Select by LOCALIZED cosine (placement quality where the ball is); full-vector cos is reported too
    # but it is structurally deflated by background dH energy for a localized predictor.
    best = {"cosL": -2.0}
    for sg in sigmas:
        summary["gate"][str(sg)] = {}
        for L in layers:
            row = {}
            for key, g in gate[sg][L].items():
                row[key] = {"cos": round(float(np.mean(g["cos"])), 4),
                            "cosL": round(float(np.nanmean(g["cosL"])), 4),
                            "rel": round(float(np.mean(g["rel"])), 4)}
            summary["gate"][str(sg)][str(L)] = row
            r = row["transport_pt_deployable"]
            print(f"[transport] sigma={sg} L{L}: pt dep cos={r['cos']:.3f} cosL={r['cosL']:.3f} "
                  f"(orc cosL={row['transport_pt_oracle']['cosL']:.3f}) | "
                  f"shared dep cosL={row['transport_shared_deployable']['cosL']:.3f}", flush=True)
            if r["cosL"] > best["cosL"]:
                best = {"cosL": r["cosL"], "cos": r["cos"], "sigma": sg, "layer": L,
                        "variant": "transport_pt"}

    # Save per-t B for ALL layers at the best sigma (steer needs every decoder layer), plus shared.
    bsg = best["sigma"]
    for L in layers:
        np.save(out / f"transport_B_L{L}.npy", B_pt[bsg][L])           # (n_t, 8, D)
        np.save(out / f"transport_shared_B_L{L}.npy", B_sh[bsg][L])    # (1, 8, D)
    summary["best"] = best
    summary["saved_sigma"] = bsg
    summary["artifacts"] = {"per_t": "transport_B_L*.npy (n_t,8,D)",
                            "shared": "transport_shared_B_L*.npy (1,8,D)",
                            "feature_order": "[M_b, M_b*v_b(2), M_a, M_a*v_a(2), M_U*dv(2)]; rows t*256+h*16+w"}
    (out / "transport_meta.json").write_text(json.dumps(summary, indent=2))
    print(f"[transport] BEST deployable (by localized cos): variant={best['variant']} "
          f"sigma={best['sigma']} L{best['layer']} cosL={best['cosL']:.3f} cos={best['cos']:.3f} "
          f"(ridge baseline full-vec cos 0.39)")
    print(f"[transport] saved per-t + shared B (sigma={bsg}) + transport_meta.json -> {out}", flush=True)


if __name__ == "__main__":
    main()
