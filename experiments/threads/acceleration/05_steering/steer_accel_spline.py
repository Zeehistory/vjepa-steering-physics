

#!/usr/bin/env python
"""Pixel-level test of SPLINE-IN-TIME acceleration steering (Step 2 "Bravo", spline arm).

Two protocols, both decoded and re-tracked so the verdict is what the pixels do.

PROTOCOL A -- command-only steering (same contract as every prior accel number: anchor = rank 0, target
= rank 7, 100 held-out TEST scenes, metric = decoded-vs-target acceleration angle error):

  noop              H_a unchanged                            the stay-put floor
  full_delta        H_a + Delta H                            per-pair oracle ceiling (needs H_b)
  prof_full         H_a + broadcast(Delta R)                 profile-only oracle: how much of the edit
                                                             survives dropping spatial placement
  proj_K{K}         H_a + broadcast(K-knot fit of Delta R)   representational ceiling at K knots
  spline_K{K}_s{g}  H_a + g * broadcast(spline(W_K . phi))   THE METHOD: command-only, no H_b
  shufT_K{K}_s{g}   same profile, temporal order PERMUTED    the control that isolates time SHAPE

K = 1 is a constant-in-t edit and is therefore the classical global-vector operator; K = 8 is the
unconstrained per-token profile. Reading spline_K1 -> spline_K8 across a fixed gain is a clean ablation
of temporal smoothness with everything else held equal. shufT keeps the exact per-token magnitudes and
destroys only their ORDER: if it matches spline_K{K}, the win was magnitude, not timing.

PROTOCOL B -- latent-family interpolation (the direct port of the spline-steering paper): a scene's 8
clips are a curve of latents indexed by acceleration. Hold out one middle rank, build the curve from the
other 7, and ask it for the held-out acceleration by Catmull-Rom vs straight-line interpolation. This is
the paper's question -- is the latent family curved enough that spline interpolation beats linear? --
asked of acceleration. INTERPOLATION only: Catmull-Rom's reflected end tangent flattens outside the
control points, so extrapolation is a known loss (pinned in tests/test_spline_ops.py).

    python experiments/threads/acceleration/05_steering/steer_accel_spline.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../moving_ball_scene_accel2d_mixed/test/vjepa2_large \
        --spline_dir outputs/analysis/moving_ball_accel2d_mixed/spline \
        --checkpoint .../moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt \
        --output_dir outputs/analysis/moving_ball_accel2d_mixed/steer_spline --device cuda
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
import sys
from pathlib import Path

import numpy as np
import torch

from src.analysis import manifold_ops as mo
from src.analysis import spline_ops as sp
from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import ball_centroids, measured_acceleration
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _apply_edit(Ha, edit_flat, device):
    """H_a (dict L->(1,Ltok,D)) + per-layer flat edit -> edited latent dict."""
    out = {}
    for L, t in Ha.items():
        e = torch.from_numpy(edit_flat[L].reshape(t.shape[1], t.shape[2]).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


def _replace_latent(Ha, new_flat, device):
    """Replace the latent outright (Protocol B interpolates absolute latents, it does not edit)."""
    return {L: torch.from_numpy(new_flat[L].reshape(1, t.shape[1], t.shape[2]).astype(np.float32)).to(device)
            for L, t in Ha.items()}


@torch.no_grad()
def _decode(decoder, latents, grid, want_frames=False):
    out = decoder(latents, grid)
    if out.frames is None:
        return {"acc_x": float("nan"), "acc_y": float("nan")}, None, None
    fr = out.frames[0].cpu()
    return measured_acceleration(fr), ball_centroids(fr), (fr if want_frames else None)


def _agg(decoded, target):
    d = np.asarray(decoded, dtype=float); t = np.asarray(target, dtype=float)
    ok = np.isfinite(d).all(1)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "rho_ax": float("nan"), "rho_ay": float("nan"),
                "angle_err_deg": float("nan"), "mag_ratio": float("nan")}
    dot = (d * t).sum(1)
    cos = dot / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    mr = np.linalg.norm(d, axis=1) / (np.linalg.norm(t, axis=1) + 1e-12)
    return {"n": int(len(d)),
            "rho_ax": round(float(np.corrcoef(t[:, 0], d[:, 0])[0, 1]), 4),
            "rho_ay": round(float(np.corrcoef(t[:, 1], d[:, 1])[0, 1]), 4),
            "angle_err_deg": round(float(np.nanmean(ang)), 2),
            "mag_ratio": round(float(np.nanmedian(mr)), 3)}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--spline_dir", required=True, help="dir holding spline_W_K*_L*.npy + meta")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--knots", default="", help="override the K list (default: all fitted)")
    p.add_argument("--gains", default="1.0,1.5,2.0,2.5,3.0")
    p.add_argument("--oracle_gains", default="",
                   help="gain grid for the ORACLE arms full_delta / prof_full / proj_K* (default: empty, "
                        "i.e. gain 1 only -- which is how they were reported through 2026-07-27 while "
                        "every FITTED arm got its own sweep. That asymmetry is what makes prof_full look "
                        "like a 31.3deg 'ceiling' sitting ABOVE fitted members of its own function class)")
    p.add_argument("--rich_dir", default="",
                   help="dir holding rich_{tag}_W_K*_L*.npy + rich_{tag}_operator_meta.json "
                        "(experiments/threads/acceleration/04_operators/fit_accel_spline_rich.py)")
    p.add_argument("--rich_features", default="appc,bilinear",
                   help="comma list of rich operator tags to decode from --rich_dir")
    p.add_argument("--appc_dir", default="",
                   help="dir holding appc_mean_L*.npy / appc_basis_L*.npy, for the rich feature map")
    p.add_argument("--layer_subsets", default="",
                   help="';'-separated layer subsets to steer, e.g. '23;18,23'. Emits "
                        "spline_K{K}L{tag}_s{g} arms that zero the edit outside the subset. The layers "
                        "are far from interchangeable -- plan_d2/d2_support.json reports that zeroing "
                        "the optimized edit at L23 costs 43.34deg while zeroing L6 costs 11.16deg "
                        "against 11.09deg intact -- yet one scalar gain has always scaled all four")
    p.add_argument("--subset_knot", type=int, default=3,
                   help="K the --layer_subsets arms are built from (one K, to bound the decode count)")
    p.add_argument("--oracle_arms", default="full_delta,prof_full,proj_K8",
                   help="comma list of oracle arms that --oracle_gains applies to (the rest stay at "
                        "gain 1). Each swept arm costs len(oracle_gains) extra decodes per pair")
    p.add_argument("--manifold_dir", default="",
                   help="dir holding man_L*.npz / manlin_L*.npz / mandense_L*.npz "
                        "(experiments/threads/acceleration/04_operators/fit_accel_manifold.py). Enables the CONCEPT-SPACE manifold arms: "
                        "edit = s(a_b) - s(a_a) where s is a thin-plate-spline surface through the "
                        "acceleration centroids (arXiv:2605.05115). Command-only -- never reads H_b")
    p.add_argument("--manifold_variants", default="man,manlin,mandense",
                   help="which fitted surfaces to decode. 'manlin' is the zero-curvature control: same "
                        "estimator, same data, same ridge, RBF block removed. man - manlin IS the "
                        "measurement of whether concept-space curvature does any work")
    p.add_argument("--manifold_knots", default="3",
                   help="comma list of K for manK{K} arms: the manifold edit projected onto the "
                        "K-knot TEMPORAL spline basis, i.e. both geometries composed. Empty to skip")
    p.add_argument("--manifold_blend", default="8",
                   help="comma list of spline K to average the manifold edit with (arm mansp{K}); the "
                        "two operators are fit on different geometries, so their mean is only useful "
                        "if their errors are partly independent. Empty to skip")
    p.add_argument("--bigU_dir", default="",
                   help="dir holding global_basis_L*.npy (the KU-dim PCA basis of the TRUE per-pair "
                        "delta H, experiments/threads/acceleration/04_operators/accel_subspace.py --save_k). Enables the ORACLE-DENOISE arms "
                        "oprojU{k}: the true Delta H projected onto its own top-k subspace. This is the "
                        "one class of arm that can beat full_delta, because full_delta at gain 1 IS the "
                        "decode of the real target latent -- i.e. the decoder/tracker noise floor, not "
                        "a steering ceiling. Beating it means the synthesized latent reads CLEANER than "
                        "reality, which is exactly what on-manifold projection is supposed to do")
    p.add_argument("--bigU_ks", default="8,32,128", help="comma list of retained dims for oprojU arms")
    p.add_argument("--manifold_mask_sigmas", default="",
                   help="comma list of Gaussian sigmas (in token cells) for the PLACED manifold arms "
                        "manmask{sigma}: the same manifold profile edit, deposited along the TARGET "
                        "trajectory instead of spatially uniform, energy-matched so the difference is "
                        "placement and not magnitude. The trajectory is rolled out from the anchor's own "
                        "pos0/v0 under the target acceleration, so it stays command-only. Empty = skip. "
                        "Placement is the one axis on which every command-only arm has been mute while "
                        "the oracle is not: prof_full (true profile, no placement) decodes at 64deg "
                        "ungained against full_delta's 11.24")
    p.add_argument("--oracle_hybrid_srcs", default="",
                   help="comma list of profile sources for the ORACLE-HYBRID family ohyb{src}_s{g} = "
                        "the true delta's SPATIAL RESIDUAL (dH - broadcast(dR)) plus g x broadcast of "
                        "that source's temporal profile. Sources: any loaded manifold variant, "
                        "'spline{K}', 'mansp{K}', and 'true' (the true profile itself -- the SHRINKAGE "
                        "control, whose g=1 member is exactly full_delta). This family exists because "
                        "'oman' beat the oracle at n=700 and the win has to be attributed: if 'ohybtrue' "
                        "matches it at some g < 1, the mechanism is plain attenuation of the true "
                        "profile and the manifold contributed nothing")
    p.add_argument("--hybrid_gains", default="",
                   help="gain grid for the oracle-hybrid family (default: --oracle_gains). Wants a "
                        "FINER, LOWER grid than the fitted arms -- the win is at g~1, not g~2.5")
    p.add_argument("--hybrid_blend_srcs", default="",
                   help="comma list of profile sources for the SHRINKAGE-BLEND family oblend{src}_s{lam} "
                        "= true spatial residual + broadcast(lam * true profile + (1-lam) * predicted). "
                        "lam is swept as if it were a gain, so calibrate_spline_gain.py selects it on "
                        "the val half like any other hyperparameter. lam=1 IS full_delta and lam=0 IS "
                        "the ohyb arm, so the family brackets both endpoints. The point: if replacing "
                        "the true profile wins because the true profile is NOISY, the optimum is "
                        "INTERIOR -- a James-Stein shrink toward the prediction should beat both ends "
                        "and by a bigger margin than either")
    p.add_argument("--hybrid_blend_lams", default="0,0.2,0.35,0.5,0.65,0.8,1.0")
    p.add_argument("--mask_control", action="store_true",
                   help="also emit manmaskX{sigma}: the same mask taken from a DIFFERENT pair's "
                        "trajectory. If it matches manmask{sigma}, placement is inert and the mask is "
                        "only reshaping magnitude")
    p.add_argument("--pairs", choices=["extreme", "all"], default="extreme",
                   help="'extreme' = the single rank0->rank_last pair per scene (the protocol through "
                        "2026-07-27, n=100); 'all' = every rank0->rank_r pair, r=1..7 (n=700). The fit "
                        "already trains on all 7 pairs; scoring on one throws away 7x the power, and at "
                        "n=100 the whole 13.4-14.5deg operator band is statistically unresolvable")
    p.add_argument("--num_scenes", type=int, default=100)
    p.add_argument("--scene_start", type=int, default=0,
                   help="slice sorted scenes [start : start+num_scenes] -- shard a long sweep across "
                        "jobs; calibrate_spline_gain.py merges the summaries")
    p.add_argument("--shuffle_knot", type=int, default=0,
                   help="K whose profile gets the temporal-shuffle control (0 = the largest fitted K)")
    p.add_argument("--family_scenes", type=int, default=40, help="Protocol B scenes (0 = skip)")
    p.add_argument("--family_ranks", default="2,4,6", help="held-out middle ranks for Protocol B")
    p.add_argument("--traj_scenes", type=int, default=6, help="scenes whose decoded trajectories are dumped")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.spline_dir)
    meta = json.loads((art / "spline_operator_meta.json").read_text())
    device = args.device

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[args.scene_start: args.scene_start + args.num_scenes]

    Ks = [int(x) for x in args.knots.split(",")] if args.knots else list(meta["knots"])
    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    ogains = [float(x) for x in args.oracle_gains.split(",") if x.strip()]
    grid = tuple(int(x) for x in ds[scenes[scene_ids[0]][0]]["grid"])
    T, D = grid[0], int(ds.records[0]["hidden_dim"])
    B = {K: sp.spline_basis(T, K, meta.get("degree", 3)) for K in Ks}
    W = {K: {L: np.load(art / f"spline_W_K{K}_L{L}.npy").astype(np.float64) for L in layers} for K in Ks}

    # Richer-conditioning operators (fit_accel_spline_rich.py). Same spline basis, wider feature map --
    # the K-sweep showed the basis is exact at K=8 while pred_cos stalls at ~0.62, so the gap is
    # synthesis. These arms read the ANCHOR latent H_a and never H_b, so they stay command-only.
    rich_tags = [t.strip() for t in args.rich_features.split(",") if t.strip()] if args.rich_dir else []
    rich_W, rich_K, rich_fm = {}, {}, {}
    if rich_tags:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fit_accel_spline_rich import FeatureMap  # noqa: E402
        rdir = Path(args.rich_dir)
        for tag in rich_tags:
            rmeta = json.loads((rdir / f"rich_{tag}_operator_meta.json").read_text())
            rich_K[tag] = [K for K in rmeta["knots"] if not args.knots or K in Ks]
            rich_fm[tag] = FeatureMap(rmeta["features"], Path(args.appc_dir) if args.appc_dir else None,
                                      layers, int(rmeta["ka"]))
            rich_W[tag] = {K: {L: np.load(rdir / f"rich_{tag}_W_K{K}_L{L}.npy").astype(np.float64)
                               for L in layers} for K in rich_K[tag]}
            for K in rich_K[tag]:                      # a rich fit may carry a K the base sweep lacks
                B.setdefault(K, sp.spline_basis(T, K, rmeta.get("degree", 3)))
            print(f"[spline-steer] loaded rich operator '{tag}' K={rich_K[tag]}", flush=True)

    # ---------------------------------------------------------------- concept-space manifold (the paper)
    # s : acceleration -> pooled temporal profile, a thin-plate-spline surface through the per-concept
    # centroids fitted in a PCA subspace. The steering edit is the difference of two points ON that
    # surface, so the intervention stays on the manifold instead of cutting a chord through activation
    # space. 'manlin' is the same estimator with the curvature removed, and is the arm that decides
    # whether the paper's mechanism is doing anything here.
    man_tags = [t.strip() for t in args.manifold_variants.split(",") if t.strip()] if args.manifold_dir else []
    manifolds = {}
    for tag in man_tags:
        mdir = Path(args.manifold_dir)
        manifolds[tag] = {L: mo.load_manifold(mdir / f"{tag}_L{L}") for L in layers}
        print(f"[spline-steer] loaded manifold '{tag}' "
              f"(k_pca={manifolds[tag][layers[0]].basis.shape[0]}, "
              f"knots={manifolds[tag][layers[0]].centers.shape[0]})", flush=True)
    if any(t.startswith("manU") for t in man_tags) and not args.bigU_dir:
        raise SystemExit("a manU* variant needs --bigU_dir (it is fitted in that basis's coordinates)")
    mask_sigmas = ([float(x) for x in args.manifold_mask_sigmas.split(",") if x.strip()]
                   if man_tags else [])
    hyb_srcs = [x.strip() for x in args.oracle_hybrid_srcs.split(",") if x.strip()]
    blend_srcs = [x.strip() for x in args.hybrid_blend_srcs.split(",") if x.strip()]
    blend_lams = [float(x) for x in args.hybrid_blend_lams.split(",") if x.strip()]
    hyb_gains = ([float(x) for x in args.hybrid_gains.split(",") if x.strip()]
                 if args.hybrid_gains else list(ogains) or [1.0])
    for src in set(hyb_srcs) | set(blend_srcs):
        if src.startswith("spline") or src.startswith("mansp"):
            K = int(src.replace("spline", "").replace("mansp", ""))
            if K not in W:
                raise SystemExit(f"--oracle_hybrid_srcs {src!r} needs spline K={K} among {sorted(W)}")
            B.setdefault(K, sp.spline_basis(T, K, meta.get("degree", 3)))
        if src.startswith("mansp") and not man_tags:
            raise SystemExit(f"--oracle_hybrid_srcs {src!r} blends a manifold profile: needs "
                             "--manifold_dir / --manifold_variants")
        if not src.startswith(("spline", "mansp")) and src != "true" and src not in man_tags:
            raise SystemExit(f"--oracle_hybrid_srcs {src!r}: not 'true', not a spline/mansp arm, and "
                             f"not among --manifold_variants {man_tags}")
    man_Ks = [int(x) for x in args.manifold_knots.split(",") if x.strip()] if man_tags else []
    man_blend = [int(x) for x in args.manifold_blend.split(",") if x.strip()] if man_tags else []
    for K in man_Ks:
        B.setdefault(K, sp.spline_basis(T, K, meta.get("degree", 3)))
    missing_blend = [K for K in man_blend if K not in W]
    if missing_blend:
        raise SystemExit(f"--manifold_blend {missing_blend} not among the fitted spline K={sorted(W)}")

    # ---------------------------------------------------------------- oracle-denoise subspace
    bigU = {}
    bigU_ks = []
    if args.bigU_dir:
        bigU_ks = sorted({int(x) for x in args.bigU_ks.split(",") if x.strip()})
        for L in layers:
            # keep every saved row: the oprojU arms slice it, and a token-space manifold edit lifts
            # through however many dims it was FITTED with, which need not be in --bigU_ks
            bigU[L] = np.asarray(np.load(Path(args.bigU_dir) / f"global_basis_L{L}.npy"),
                                 dtype=np.float32)                      # (KU, T*H*W*D)
        print(f"[spline-steer] loaded oracle-denoise basis {bigU[layers[0]].shape} ks={bigU_ks}",
              flush=True)
    # dims used to LIFT a token-space manifold edit -- must match --bigU_k at fit time
    bigU_lift = manifolds[[t for t in man_tags if t.startswith("manU")][0]][layers[0]].basis.shape[1] \
        if any(t.startswith("manU") for t in man_tags) else 0

    subsets = []
    for spec in (x for x in args.layer_subsets.split(";") if x.strip()):
        sel = [int(v) for v in spec.split(",") if v.strip()]
        missing = [L for L in sel if L not in layers]
        if missing:
            raise SystemExit(f"--layer_subsets {spec!r}: layers {missing} are not in the latents {layers}")
        subsets.append(("".join(str(L) for L in sel), sel))
    K_sub = args.subset_knot if subsets else None
    if K_sub is not None and K_sub not in Ks:
        raise SystemExit(f"--subset_knot {K_sub} is not among the fitted/selected K={Ks}")

    K_shuf = args.shuffle_knot or max(Ks)
    rng = np.random.default_rng(0)
    # one fixed derangement of the temporal axis, shared across scenes so the control is a property of
    # the METHOD and not of a per-scene random draw
    perm = np.arange(T)
    while (perm == np.arange(T)).any():
        perm = rng.permutation(T)

    print(f"[spline-steer] {len(scenes)} test scenes, steering {len(scene_ids)}; layers={layers}; "
          f"grid={grid}; K={Ks}; gains={gains}; shuffle control on K={K_shuf} perm={perm.tolist()}",
          flush=True)

    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    # Oracle arms get a gain sweep only when --oracle_gains is given. Gain 1 is always kept so the old
    # unswept names stay present and the new run remains readable against the 2026-07-27 table.
    oracle_base = ["full_delta", "prof_full"] + [f"proj_K{K}" for K in Ks]
    swept_oracles = [m for m in oracle_base
                     if m in {x.strip() for x in args.oracle_arms.split(",") if x.strip()}]
    methods = (["noop"] + oracle_base
               + [f"{m}_s{g:g}" for m in swept_oracles for g in ogains]
               + [f"spline_K{K}_s{g:g}" for K in Ks for g in gains]
               + [f"rich{tag}_K{K}_s{g:g}" for tag in rich_tags for K in rich_K[tag] for g in gains]
               + [f"spline_K{K_sub}L{tag}_s{g:g}" for tag, _ in subsets for g in gains]
               + [f"{tag}_s{g:g}" for tag in man_tags for g in gains]
               + [f"manK{K}_s{g:g}" for K in man_Ks for g in gains]
               + [f"mansp{K}_s{g:g}" for K in man_blend for g in gains]
               + [f"oprojU{k}_s{g:g}" for k in bigU_ks for g in ogains or [1.0]]
               + ([f"oman_s{g:g}" for g in ogains or [1.0]] if man_tags else [])
               + [f"ohyb{src}_s{g:g}" for src in hyb_srcs for g in hyb_gains]
               + [f"oblend{src}_s{lam:g}" for src in blend_srcs for lam in blend_lams]
               + [f"manmask{sg:g}_s{g:g}" for sg in mask_sigmas for g in gains]
               + ([f"manmaskX{sg:g}_s{g:g}" for sg in mask_sigmas for g in gains]
                  if args.mask_control else [])
               + [f"shufT_K{K_shuf}_s{g:g}" for g in gains])
    decoded = {m: [] for m in methods}
    targets, per_scene, trajectories = [], {}, {}

    # ---------------------------------------------------------------- Protocol A: command-only steering
    n = 0
    for s in scene_ids:
        ranks = sorted(scenes[s])
        sa = ds[scenes[s][ranks[0]]]
        aa = vo.clip_acceleration(sa)
        Ha = _to_dev(sa, layers, device)
        tgt_ranks = ranks[1:] if args.pairs == "all" else [ranks[-1]]

        for b in tgt_ranks:
            sb = ds[scenes[s][b]]
            ab = vo.clip_acceleration(sb)
            phi = vo.command_features(aa, ab)

            dH = {L: vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]) for L in layers}
            dR = {L: sp.temporal_profile(dH[L], grid) for L in layers}
            zero = {L: np.zeros_like(dH[L]) for L in layers}

            base_edits = {"full_delta": dH,
                          "prof_full": {L: sp.broadcast_profile(dR[L], grid) for L in layers}}
            edits = {"noop": zero}
            for K in Ks:
                base_edits[f"proj_K{K}"] = {
                    L: sp.broadcast_profile(sp.smooth_profile(dR[L], B[K]), grid) for L in layers}
            for m, e in base_edits.items():
                edits[m] = e
                if m in swept_oracles:
                    for g in ogains:
                        edits[f"{m}_s{g:g}"] = {L: g * e[L] for L in layers}
            for K in Ks:
                pred = {L: sp.reconstruct_profile((phi @ W[K][L]).reshape(K, D), B[K]) for L in layers}
                for g in gains:
                    edits[f"spline_K{K}_s{g:g}"] = {L: sp.broadcast_profile(g * pred[L], grid)
                                                    for L in layers}
                if K == K_shuf:
                    for g in gains:
                        edits[f"shufT_K{K}_s{g:g}"] = {
                            L: sp.broadcast_profile(g * pred[L][perm], grid) for L in layers}
                if K == K_sub:
                    for tag, sel in subsets:
                        for g in gains:
                            edits[f"spline_K{K}L{tag}_s{g:g}"] = {
                                L: (sp.broadcast_profile(g * pred[L], grid) if L in sel else zero[L])
                                for L in layers}
            # ---- concept-space manifold arms (command-only: labels a_a, a_b only, never H_b)
            # A 'manU*' variant was fitted in TOKEN-SPACE coordinates (the saved global delta basis),
            # so its edit lifts back through that basis and keeps spatial placement. Every other
            # variant lives on the pooled profile and is broadcast spatially uniform, i.e. in the same
            # function class as the spline arms it is being compared to.
            man_edit, prof_tags = {}, []
            for tag in man_tags:
                if tag.startswith("manU"):
                    flat = {L: manifolds[tag][L].edit(aa, ab) @ bigU[L][: bigU_lift] for L in layers}
                    for g in gains:
                        edits[f"{tag}_s{g:g}"] = {L: g * flat[L] for L in layers}
                else:
                    mp = {L: manifolds[tag][L].edit(aa, ab).reshape(T, D) for L in layers}
                    man_edit[tag] = mp
                    prof_tags.append(tag)
                    for g in gains:
                        edits[f"{tag}_s{g:g}"] = {L: sp.broadcast_profile(g * mp[L], grid)
                                                  for L in layers}
            if prof_tags:
                base_mp = man_edit[prof_tags[0]]
                for K in man_Ks:
                    for g in gains:
                        edits[f"manK{K}_s{g:g}"] = {
                            L: sp.broadcast_profile(g * sp.smooth_profile(base_mp[L], B[K]), grid)
                            for L in layers}
                for K in man_blend:
                    sppred = {L: sp.reconstruct_profile((phi @ W[K][L]).reshape(K, D), B[K])
                              for L in layers}
                    for g in gains:
                        edits[f"mansp{K}_s{g:g}"] = {
                            L: sp.broadcast_profile(0.5 * g * (base_mp[L] + sppred[L]), grid)
                            for L in layers}
                # PLACED manifold edit: same profile, deposited along the target trajectory
                if mask_sigmas:
                    F = int(np.asarray(sa["state"]).shape[0])
                    skeys = list(sa["state_keys"]); sst = np.asarray(sa["state"])
                    pos0 = np.array([float(sst[0, skeys.index("obj0_pos_x")]),
                                     float(sst[0, skeys.index("obj0_pos_y")])])
                    v0 = np.array([float(sst[0, skeys.index("obj0_vel_x")]),
                                   float(sst[0, skeys.index("obj0_vel_y")])])
                    cen = vo.temporal_token_centers(mo.accel_trajectory(pos0, v0, ab, F), T)
                    cenX = vo.temporal_token_centers(
                        mo.accel_trajectory(1.0 - pos0, -v0, -ab, F), T)   # a decoy path, same shape
                    for sg in mask_sigmas:
                        mk = vo.gaussian_mask(cen, grid, sg)
                        placed = {L: mo.place_profile(base_mp[L], mk, grid) for L in layers}
                        for g in gains:
                            edits[f"manmask{sg:g}_s{g:g}"] = {L: g * placed[L] for L in layers}
                        if args.mask_control:
                            mkX = vo.gaussian_mask(cenX, grid, sg)
                            placedX = {L: mo.place_profile(base_mp[L], mkX, grid) for L in layers}
                            for g in gains:
                                edits[f"manmaskX{sg:g}_s{g:g}"] = {L: g * placedX[L] for L in layers}
                # ORACLE HYBRID: keep the true delta's SPATIAL placement, replace its temporal profile
                # with the on-manifold one. prof_full (true profile, no placement) decodes at 64deg
                # ungained while full_delta decodes at 11.24, so placement carries most of the oracle's
                # advantage; this asks whether the profile half of the oracle can be improved on.
                for g in ogains or [1.0]:
                    edits[f"oman_s{g:g}"] = {
                        L: dH[L] - sp.broadcast_profile(dR[L], grid)
                           + sp.broadcast_profile(g * base_mp[L], grid) for L in layers}
            # ---- ORACLE-HYBRID family: true spatial residual + a chosen temporal profile
            if hyb_srcs or blend_srcs:
                resid = {L: dH[L] - sp.broadcast_profile(dR[L], grid) for L in layers}

                def _profile(src):
                    if src == "true":
                        return dR
                    if src.startswith("mansp"):
                        K = int(src.replace("mansp", ""))
                        return {L: 0.5 * (man_edit[prof_tags[0]][L]
                                          + sp.reconstruct_profile((phi @ W[K][L]).reshape(K, D), B[K]))
                                for L in layers}
                    if src.startswith("spline"):
                        K = int(src.replace("spline", ""))
                        return {L: sp.reconstruct_profile((phi @ W[K][L]).reshape(K, D), B[K])
                                for L in layers}
                    return man_edit[src]

                for src in hyb_srcs:
                    P = _profile(src)
                    for g in hyb_gains:
                        edits[f"ohyb{src}_s{g:g}"] = {
                            L: resid[L] + sp.broadcast_profile(g * P[L], grid) for L in layers}
                for src in blend_srcs:
                    P = _profile(src)
                    for lam in blend_lams:
                        edits[f"oblend{src}_s{lam:g}"] = {
                            L: resid[L] + sp.broadcast_profile(lam * dR[L] + (1.0 - lam) * P[L], grid)
                            for L in layers}
                del resid
            # ---- oracle-denoise arms: the TRUE delta projected onto its own global subspace
            for k in bigU_ks:
                proj = {}
                for L in layers:
                    Uk = bigU[L][:k]
                    proj[L] = (Uk.T @ (Uk @ dH[L].astype(np.float32))).astype(np.float64)
                for g in ogains or [1.0]:
                    edits[f"oprojU{k}_s{g:g}"] = {L: g * proj[L] for L in layers}
            for tag in rich_tags:
                for K in rich_K[tag]:
                    rp = {L: sp.reconstruct_profile(
                        (rich_fm[tag](aa, ab, sa["layers"][L], L) @ rich_W[tag][K][L]).reshape(K, D),
                        B[K]) for L in layers}
                    for g in gains:
                        edits[f"rich{tag}_K{K}_s{g:g}"] = {
                            L: sp.broadcast_profile(g * rp[L], grid) for L in layers}

            targets.append(ab)
            # key carries the scene id so calibrate_spline_gain.py can split by SCENE, never by pair --
            # the 7 pairs of a scene share an anchor clip and are not independent samples
            key = f"scene{s:05d}_r{b}" if args.pairs == "all" else f"scene{s:05d}"
            row = {"v_b": ab.tolist(), "a_a": aa.tolist(), "scene": int(s), "rank": int(b)}
            keep_traj = n < args.traj_scenes
            traj = {"target_a": ab.tolist(), "anchor_a": aa.tolist()} if keep_traj else None
            for m in methods:
                meas, cent, _ = _decode(decoder, _apply_edit(Ha, edits[m], device), grid)
                decoded[m].append([meas["acc_x"], meas["acc_y"]])
                row[m] = [round(meas["acc_x"], 6), round(meas["acc_y"], 6)]
                if keep_traj and cent is not None:
                    traj[m] = np.where(np.isnan(cent), None, cent.round(5)).tolist()
            if keep_traj:
                # ground truth: the real target clip's own pixels, tracked identically
                traj["gt_target"] = ball_centroids(sb["frames"]).round(5).tolist()
                traj["gt_anchor"] = ball_centroids(sa["frames"]).round(5).tolist()
                trajectories[key] = traj
            per_scene[key] = row
            if n % 25 == 0:
                print(f"  [A] {key} (pair {n+1}) a_b=({ab[0]:.4f},{ab[1]:.4f}) "
                      f"full={tuple(round(x,4) for x in decoded['full_delta'][-1])}", flush=True)
            n += 1
            del sb, dH, dR, zero, edits, base_edits
        del sa, Ha

    results = {m: _agg(decoded[m], targets) for m in methods}

    # ------------------------------------------------- Protocol B: latent-family spline vs linear interp
    family = {}
    if args.family_scenes > 0:
        fam_ranks = [int(x) for x in args.family_ranks.split(",") if x.strip()]
        fam_dec = {"family_spline": [], "family_lerp": [], "family_true": []}
        fam_targets = []
        for n, s in enumerate(sorted(scenes)[: args.family_scenes]):
            ranks = sorted(scenes[s])
            samples = {r: ds[scenes[s][r]] for r in ranks}
            accs = {r: vo.clip_acceleration(samples[r]) for r in ranks}
            for held in fam_ranks:
                if held not in ranks:
                    continue
                keep = [r for r in ranks if r != held]
                # parametrize the family by SIGNED acceleration magnitude along the target direction,
                # which is monotone in rank by construction of the dataset
                mags = np.array([float(np.linalg.norm(accs[r])) for r in keep])
                order = np.argsort(mags)
                keep_sorted = [keep[i] for i in order]
                mags_sorted = mags[order]
                q = sp.interp_index(mags_sorted, float(np.linalg.norm(accs[held])))
                Hq = {"family_spline": {}, "family_lerp": {}}
                for L in layers:
                    P_ctrl = np.stack([vo.layer_flat(samples[r]["layers"][L]) for r in keep_sorted], 0)
                    Hq["family_spline"][L] = sp.catmull_rom(P_ctrl, q)
                    Hq["family_lerp"][L] = sp.linear_eval(P_ctrl, q)
                    del P_ctrl
                base = _to_dev(samples[held], layers, device)
                for m in ("family_spline", "family_lerp"):
                    meas, _, _ = _decode(decoder, _replace_latent(base, Hq[m], device), grid)
                    fam_dec[m].append([meas["acc_x"], meas["acc_y"]])
                meas_true, _, _ = _decode(decoder, base, grid)
                fam_dec["family_true"].append([meas_true["acc_x"], meas_true["acc_y"]])
                fam_targets.append(accs[held])
            del samples
            if n % 10 == 0:
                print(f"  [B] scene{s:05d} ({n+1}/{args.family_scenes})", flush=True)
        family = {m: _agg(fam_dec[m], fam_targets) for m in fam_dec}
        family["note"] = ("leave-one-rank-out INTERPOLATION of the scene's latent family; "
                          "family_true = decoding the real held-out latent (this protocol's ceiling)")
        family["held_out_ranks"] = fam_ranks

    summary = {
        "protocol_A": ("command-only: anchor rank0 -> target rank_r, decoded-vs-target accel angle "
                       f"error; pairs={args.pairs} ({len(per_scene)} evaluated pairs)"),
        "test_dir": args.test_dir, "checkpoint": args.checkpoint, "spline_dir": str(art),
        "layers": layers, "grid": list(grid), "knots": Ks, "gains": gains, "oracle_gains": ogains,
        "pairs": args.pairs, "n_scenes": len(scene_ids), "n_pairs": len(per_scene),
        "scene_ids": [int(s) for s in scene_ids],
        "shuffle_knot": K_shuf, "shuffle_perm": perm.tolist(),
        "target": "a_b (decoded vs target acceleration)",
        # every knob, so a number is never orphaned from the run that produced it (the 5.07deg TTO
        # result is not reproducible from its own summary because --anchor was never recorded)
        "args": {k: v for k, v in vars(args).items() if k != "overrides"},
        "results": results, "family": family, "per_scene": per_scene,
    }
    (out / "steer2d_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "trajectories.json").write_text(json.dumps(trajectories, indent=1))

    print("\n[spline-steer] PROTOCOL A — decoded-vs-target acceleration:")
    for m in methods:
        r = results[m]
        print(f"  {m:20s} angle_err={r['angle_err_deg']!s:>6}deg mag={r['mag_ratio']!s:>6} "
              f"rho=({r['rho_ax']},{r['rho_ay']}) n={r['n']}")
    if family:
        print("\n[spline-steer] PROTOCOL B — latent-family interpolation (spline vs linear):")
        for m in ("family_true", "family_spline", "family_lerp"):
            r = family[m]
            print(f"  {m:20s} angle_err={r['angle_err_deg']!s:>6}deg mag={r['mag_ratio']!s:>6} n={r['n']}")
    print(f"[spline-steer] -> {out}/steer2d_summary.json")


if __name__ == "__main__":
    main()
