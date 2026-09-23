

#!/usr/bin/env python
"""Pixel-level proof for the 2D-ACCELERATION subspace / command operator (Step 2 "Bravo").

The acceleration analog of ``experiments/threads/velocity/05_steering/steer_velocity2d.py``. Loads the held-out TEST scene cache + the
faithful decoder trained on accel latents + the artifacts fit on TRAIN by ``accel_subspace.py`` /
``fit_command_operators_accel.py``. For each test scene it forms the anchor->extreme pair
(a_a -> a_b, Delta a = a_b - a_a, Delta H = H_b - H_a) and steers H_a by several methods, then DECODES
and re-tracks the ball's 2D ACCELERATION (parabola fit to the decoded centroid, a = 2*c2):

  full_delta      H_a + Delta H                     per-pair on-manifold ceiling
  subspace_U[k]   H_a + P_U(Delta H)                project true edit onto top-k global PCA basis
  random[k]       H_a + P_R(Delta H)                random same-rank subspace control (should fail)
  ridge_global    H_a + B . Delta a                 steer straight from the accel command (bare ridge)
  cmd_U8_s{g}     H_a + g * (phi(a_a,a_b) Wu) U     COMMAND-ONLY synthesis in U (gain swept) -- the method
  ridge_rich      H_a + phi . B_rich                richer-feature global ridge

Reports decoded-vs-target acceleration angle error (deg) + magnitude ratio, aggregated. The per-scene
target acceleration is stored under the key ``v_b`` so ``experiments/pipeline/04_operators/calibrate_cmd_gain.py`` (quantity-
agnostic; it just reads ``v_b`` as "the target vector") does the leakage-free gain calibration unchanged.

    python experiments/threads/acceleration/05_steering/steer_accel2d.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../moving_ball_scene_accel2d/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_accel2d/subspace \
        --checkpoint .../moving_ball_scene_accel2d_decoder_fp/checkpoints/last.pt \
        --output_dir outputs/analysis/moving_ball_accel2d/steer --ks 2,4,8,16 --num_scenes 100 --device cuda
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

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_acceleration
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    from src.encoders.feature_extractor import latent_collate
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _temporal_pool(flat: np.ndarray, grid) -> np.ndarray:
    """(T*H*W*D,) -> (T, D) spatial mean per frame (matches accel_operator_search / probe)."""
    T, H, W = grid
    D = flat.size // (T * H * W)
    return flat.reshape(T, H, W, D).mean(axis=(1, 2))


def _broadcast_profile(profile_td: np.ndarray, grid) -> np.ndarray:
    """(T, D) temporal profile -> flat (T*H*W*D,) constant across spatial tokens (spatial-mean = profile)."""
    T, H, W = grid
    D = profile_td.shape[1]
    return np.broadcast_to(profile_td[:, None, None, :], (T, H, W, D)).reshape(-1).copy()


def _apply_edit(Ha, edit_flat, device):
    out = {}
    for L, t in Ha.items():
        Ltok, Dd = t.shape[1], t.shape[2]
        e = torch.from_numpy(edit_flat[L].reshape(Ltok, Dd).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


@torch.no_grad()
def _decode_accel(decoder, latents, grid):
    out = decoder(latents, grid)
    if out.frames is None:
        return {"acc_x": float("nan"), "acc_y": float("nan")}
    return measured_acceleration(out.frames[0].cpu())


@torch.no_grad()
def _decode_frames(decoder, latents, grid):
    out = decoder(latents, grid)
    return None if out.frames is None else out.frames[0].cpu().numpy()  # (T, C, H, W)


def _save_viz(path, rows, titles, max_frames=8):
    """Filmstrip grid: one row per condition, evenly-spaced frames across columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = rows[0].shape[0]
    idx = np.linspace(0, T - 1, min(max_frames, T)).round().astype(int)
    nr, nc = len(rows), len(idx)
    fig, axes = plt.subplots(nr, nc, figsize=(1.5 * nc, 1.7 * nr), squeeze=False)
    for r, (frames, title) in enumerate(zip(rows, titles)):
        for c, fi in enumerate(idx):
            img = np.transpose(frames[fi], (1, 2, 0)).clip(0, 1)
            ax = axes[r][c]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
            if r == 0:
                ax.set_title(f"t={fi}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _agg(decoded, target):
    """decoded/target: lists of (ax,ay). Return correlation, mean angle err deg, median mag ratio."""
    d = np.asarray(decoded); t = np.asarray(target)
    ok = np.isfinite(d).all(1)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "rho_ax": float("nan"), "rho_ay": float("nan"),
                "angle_err_deg": float("nan"), "mag_ratio": float("nan")}
    rho_ax = float(np.corrcoef(t[:, 0], d[:, 0])[0, 1])
    rho_ay = float(np.corrcoef(t[:, 1], d[:, 1])[0, 1])
    dot = (d * t).sum(1)
    cos = dot / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    mr = np.linalg.norm(d, axis=1) / (np.linalg.norm(t, axis=1) + 1e-12)
    return {"n": int(len(d)), "rho_ax": round(rho_ax, 4), "rho_ay": round(rho_ay, 4),
            "angle_err_deg": round(float(np.nanmean(ang)), 2),
            "mag_ratio": round(float(np.nanmedian(mr)), 3)}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ks", default="2,4,8,16")
    p.add_argument("--num_scenes", type=int, default=100)
    p.add_argument("--cmd_scales", default="1.0,1.5,2.0,2.5,3.0", help="gain sweep for cmd_U")
    p.add_argument("--cmd_ku", type=int, default=8, help="8 -> cmd_Wu_L*.npy; 16 -> cmd_Wu_ku16_L*.npy; 0 -> whole basis")
    p.add_argument("--basis_tag", default="",
                   help="use global_basis_{tag}_L*.npy (e.g. 'spd' = U8 + supervised |a| axes) and the matching "
                        "cmd_Wu*_{tag} operator; pass --ks ...,11 for the new oracle-projected ceiling.")
    p.add_argument("--spd_scales", default="",
                   help="with --basis_tag: extra gains on the speed-axis part, adds cmd_U8_s{g}_m{gm}.")
    p.add_argument("--cmd_std", action="store_true",
                   help="load the STANDARDIZED-ridge command operator (cmd_Wu_std*_L*.npy) written by "
                        "fit_command_operators_accel.py --standardize. Base features only.")
    p.add_argument("--features",
                   choices=["base", "quad", "appc", "canon", "canon_appc",
                            "prof", "v0_prof", "probe_ax", "hybrid", "trajcanon", "transfield", "bilinear",
                            "slabvel", "curv_integ"],
                   default="base",
                   help="base/quad/appc/canon/canon_appc -> the original subspace operators. prof/v0_prof/"
                        "probe_ax/hybrid -> TEMPORAL-PROFILE operators from accel_operator_search.py "
                        "(prof_operators.npz): prof = broadcast cmd->r(dH); v0_prof = [cmd,v0]->r(dH); "
                        "probe_ax = min-norm edit along the probe axis (probe reads Delta a exactly); "
                        "hybrid = canon spatial footprint + probe-corrected temporal profile. Must match fit.")
    p.add_argument("--opsearch_dir", default="", help="dir with prof_operators.npz (for prof/probe_ax/hybrid)")
    p.add_argument("--viz_scenes", type=int, default=0,
                   help="save unsteered/cmd-U8-steered/target filmstrips for the first N scenes (0=off)")
    p.add_argument("--viz_gain", type=float, default=2.0, help="cmd-U8 gain used for the viz filmstrips")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device
    ks = [int(x) for x in args.ks.split(",")]
    cmd_scales = [float(x) for x in args.cmd_scales.split(",") if x.strip()]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[steer-a] {len(scenes)} test scenes, steering {len(scene_ids)}; layers={layers}")

    Bt = {L: np.load(art / f"ridge_Bt_L{L}.npy").astype(np.float64) for L in layers}
    btag = f"_{args.basis_tag}" if args.basis_tag else ""
    Ubasis = {L: np.load(art / f"global_basis{btag}_L{L}.npy").astype(np.float64) for L in layers}
    spd_scales = [float(x) for x in args.spd_scales.split(",") if x.strip()]
    npca = 8
    if args.basis_tag and (art / "speed_axis_meta.json").exists():
        npca = int(json.loads((art / "speed_axis_meta.json").read_text())["ku"])
    cmd_ku = args.cmd_ku if args.cmd_ku > 0 else min(Ubasis[L].shape[0] for L in layers)
    rng = np.random.default_rng(0)
    Rbasis = {L: {k: vo.random_basis(Bt[L].shape[1], k, rng) for k in ks} for L in layers}

    feat_fn = vo.command_features_quad if args.features == "quad" else vo.command_features
    feat_tag = "_quad" if args.features == "quad" else ""
    appc = args.features == "appc"
    canon = args.features == "canon"
    canon_appc = args.features == "canon_appc"
    prof_feats = args.features in ("prof", "v0_prof", "probe_ax", "hybrid")
    trajcanon = args.features == "trajcanon"
    transfield = args.features == "transfield"
    bilinear = args.features == "bilinear"
    slabvel = args.features == "slabvel"
    curv_integ = args.features == "curv_integ"
    if prof_feats:
        npz = np.load(Path(args.opsearch_dir) / "prof_operators.npz")
        Bprof = {L: npz[f"Bprof_L{L}"].astype(np.float64) for L in layers}
        Bv0 = {L: npz[f"Bv0_L{L}"].astype(np.float64) for L in layers}
        Wp = {L: npz[f"probeWp_L{L}"].astype(np.float64) for L in layers}          # (T*D, 2)
        Wpinv = {L: Wp[L] @ np.linalg.inv(Wp[L].T @ Wp[L] + 1e-6 * np.eye(2)) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        if args.features == "hybrid":
            Ucanon = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
            Wu = {L: np.load(art / f"cmd_Wu_canon_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded PROFILE operator '{args.features}' from {args.opsearch_dir}")
    elif trajcanon:
        Ucanon = {L: np.load(art / f"global_basis_trajcanon_L{L}.npy").astype(np.float64) for L in layers}
        Wu = {L: np.load(art / f"cmd_Wu_trajcanon_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded TRAJCANON operator (U rows={Ucanon[layers[0]].shape[0]})")
    elif transfield:
        # 2nd-order translation field: per-token map [Dx || Dx (x) posbasis(x_a)] -> slab, driven by the
        # command via Dx(t)=(a_b-a_a)*gvec[t]. Full-D per-layer; no U basis, no rolling.
        Btf = {L: np.load(art / f"cmd_Btransfield_L{L}.npy").astype(np.float64) for L in layers}
        gvec = np.load(art / "transfield_gvec.npy").astype(np.float64)
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded TRANSFIELD operator (F={Btf[layers[0]].shape[0]}, T={gvec.shape[0]})")
    elif slabvel:
        # temporal-composition velocity operator: per token, edit[slab t] = (cmd_feat(v_a(t),v_b(t)) @
        # W_slab) @ U_slab, with v_b(t) = v_a(t) + (a_b - a_a)*tau_t reconstructed from the reference clip.
        Uslab = {L: np.load(art / f"slabvel_basis_L{L}.npy").astype(np.float64) for L in layers}
        Wslab = {L: np.load(art / f"cmd_Wslabvel_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded SLABVEL operator (U_slab rows={Uslab[layers[0]].shape[0]}) + base Brich ref")
    elif curv_integ:
        # 2nd-order integration operator: command -> ONE curvature-change direction dA_pred in U_curv;
        # the edit at slab t is dA_pred * (t-1)t/2 (discrete double-integral -> physical t^2 profile).
        Ucurv = {L: np.load(art / f"global_basis_curv_L{L}.npy").astype(np.float64) for L in layers}
        Wu = {L: np.load(art / f"cmd_Wu_curv_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded CURV_INTEG operator (U_curv rows={Ucurv[layers[0]].shape[0]}) + base Brich ref")
    elif bilinear:
        # global reference-conditioned operator: edit = ([cmd || U@H_a || cmd (x) U@H_a] @ Wbilin) @ U.
        Ubil = {L: np.load(art / f"global_basis_bilin_L{L}.npy").astype(np.float64) for L in layers}
        Wbil = {L: np.load(art / f"cmd_Wbilin_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded BILINEAR operator (U rows={Ubil[layers[0]].shape[0]}, "
              f"F={Wbil[layers[0]].shape[0]})")
    elif canon_appc:
        # combined: [command || appc(H_a)] -> U_canon coords, then un-roll to each scene's start cell.
        Ucanon = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
        Wu = {L: np.load(art / f"cmd_Wu_canon_appc_L{L}.npy").astype(np.float64) for L in layers}
        appc_mean = {L: np.load(art / f"appc_mean_L{L}.npy").astype(np.float64) for L in layers}
        appc_basis = {L: np.load(art / f"appc_basis_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded CANON+APPC operator (U_canon rows={Ucanon[layers[0]].shape[0]}, "
              f"Pa={Wu[layers[0]].shape[0]}) + base Brich ref")
    elif canon:
        # position-canonicalized operator: synthesize the edit in the ball-start-centred frame, then
        # un-roll to each test scene's own start cell before applying.
        Ucanon = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
        Wu = {L: np.load(art / f"cmd_Wu_canon_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded CANON operator (U_canon rows={Ucanon[layers[0]].shape[0]}) + base Brich ref")
    elif appc:
        # appearance-conditioned rich operator: edit = [command || appc(H_a)] @ B_appc (full D, not U).
        # cmd_Wu/global-basis are unused for the cmd edit; Brich (base) kept only for the ridge_rich ref.
        Bappc = {L: np.load(art / f"cmd_Bappc_L{L}.npy").astype(np.float64) for L in layers}
        appc_mean = {L: np.load(art / f"appc_mean_L{L}.npy").astype(np.float64) for L in layers}
        appc_basis = {L: np.load(art / f"appc_basis_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded APPC operator (Ka={appc_basis[layers[0]].shape[0]}, "
              f"Pa={Bappc[layers[0]].shape[0]}) + base Brich ref")
    else:
        wu_tag = feat_tag + ("" if cmd_ku == 8 else f"_ku{cmd_ku}")
        if args.cmd_std:
            wu_tag = "_std" + wu_tag
        wu_tag = wu_tag + btag
        Wu = {L: np.load(art / f"cmd_Wu{wu_tag}_L{L}.npy").astype(np.float64) for L in layers}
        Brich = {L: np.load(art / f"cmd_Brich{feat_tag}_L{L}.npy").astype(np.float64) for L in layers}
        print(f"[steer-a] loaded ridge B, U basis (rows={Ubasis[layers[0]].shape[0]}), "
              f"cmd W_U (features={args.features} ku={args.cmd_ku} std={args.cmd_std})")

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    # The state head is auxiliary (never read here); size it from the checkpoint so decoders trained on
    # the 12-column schema still load against records re-extracted with the 15-column (theta/omega/alpha) one.
    _ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for _k, _v in _ck["model"].items():
        if _k.endswith("state_head.3.bias"):
            if int(_v.shape[0]) != state_dim:
                print(f"[steer-a] state_dim: records={state_dim} checkpoint={int(_v.shape[0])} -> using checkpoint")
            state_dim = int(_v.shape[0])
    del _ck
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    cmd_methods = ([f"cmd_U8_s{g:g}" for g in cmd_scales]
                   + [f"cmd_U8_s{g:g}_m{gm:g}" for g in cmd_scales for gm in spd_scales] + ["ridge_rich"])
    methods = ["full_delta", "ridge_global"] + cmd_methods + \
              [f"subspace_U{k}" for k in ks] + [f"random{k}" for k in ks]
    decoded = {m: [] for m in methods}
    targets = []
    per_scene = {}

    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        da = ab - aa
        Ha = _to_dev(sa, layers, device)
        dH = {L: vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]) for L in layers}

        edits = {
            "full_delta": dH,
            "ridge_global": {L: da @ Bt[L] for L in layers},
        }
        if prof_feats:
            # TEMPORAL-PROFILE edit: reconstruct the spatial-mean-per-frame change and broadcast it across
            # spatial tokens (spatial-mean(edit) == the target profile -> the probe reads it directly).
            cmd = vo.command_features(aa, ab)
            T = grid[0]
            if args.features == "prof":
                cU = {L: _broadcast_profile((cmd @ Bprof[L]).reshape(T, -1), grid) for L in layers}
            elif args.features == "v0_prof":
                v0 = vo.clip_velocity(sa)
                cU = {L: _broadcast_profile((np.concatenate([cmd, v0]) @ Bv0[L]).reshape(T, -1), grid)
                      for L in layers}
            elif args.features == "probe_ax":
                cU = {L: _broadcast_profile((Wpinv[L] @ da).reshape(T, -1), grid) for L in layers}
            else:  # hybrid: canon spatial footprint + broadcast correction so probe reads Delta a exactly
                sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
                cU = {}
                for L in layers:
                    ce = vo.roll_layer((cmd @ Wu[L]) @ Ucanon[L], grid, (-sh[0], -sh[1]))
                    corr = Wpinv[L] @ (da - _temporal_pool(ce, grid).reshape(-1) @ Wp[L])
                    cU[L] = ce + _broadcast_profile(corr.reshape(T, -1), grid)
            phi = cmd  # ridge_rich reference uses the base command features
        elif slabvel:
            # temporal composition: reconstruct per-token velocities from the reference clip + command,
            # apply the shared velocity operator per slab, assemble the full edit.
            T = grid[0]
            F = int(np.asarray(sa["state"]).shape[0])
            va_t = vo.clip_frame_velocities(sa, T)               # (T,2) reference per-token velocity
            tau = vo.frame_token_times(F, T)                     # (T,) mean frame index per token
            vb_t = va_t + np.outer(tau, da)                     # (T,2) target: v_a(t) + Delta a * tau_t
            phis = np.stack([vo.command_features(va_t[t], vb_t[t]) for t in range(T)], 0)  # (T,P)
            cU = {}
            for L in layers:
                coords = phis @ Wslab[L]                         # (T,ku)
                cU[L] = (coords @ Uslab[L]).reshape(-1)          # (T,slab) -> flat (T*H*W*D)
            phi = vo.command_features(aa, ab)                    # ridge_rich reference uses base command
        elif curv_integ:
            # predict ONE curvature-change direction from the command, then DOUBLE-INTEGRATE it: the edit
            # at slab t is dA_pred * (t-1)t/2 (e_0=e_1=0). This imposes the physical t^2 temporal profile
            # (zero early, growing late) that per-t/global operators averaged away, from a single high-SNR
            # direction pooled over all timesteps at fit time.
            cmd = vo.command_features(aa, ab)
            T = grid[0]
            prof = np.array([(t - 1) * t / 2.0 for t in range(T)])   # 0,0,1,3,6,10,15,21
            cU = {}
            for L in layers:
                dA_pred = (cmd @ Wu[L]) @ Ucurv[L]                    # (Dslab,)
                cU[L] = np.outer(prof, dA_pred).reshape(-1)          # (T*Dslab,) = full layer_flat
            phi = cmd
        elif bilinear:
            # global reference-conditioned: coords_a = U @ H_a (reference projection on accel subspace);
            # edit = ([cmd || coords_a || cmd (x) coords_a] @ Wbilin) @ U. H_b-free (coords_a from reference).
            cmd = vo.command_features(aa, ab)
            cU = {}
            for L in layers:
                ca = Ubil[L] @ vo.layer_flat(sa["layers"][L])
                feat = np.concatenate([cmd, ca, np.outer(cmd, ca).reshape(-1)])
                cU[L] = (feat @ Wbil[L]) @ Ubil[L]
            phi = cmd
        elif transfield:
            # 2nd-order TRANSLATION FIELD: per-token edit = [Dx(t) || Dx(t) (x) posbasis(x_a(t))] @ Btf.
            # Dx(t) = (a_b - a_a) * gvec[t] (command-derived t^2 profile); x_a(t) from the reference clip.
            T, H, W = grid
            pa = vo.clip_frame_positions(sa, T)             # (T,2) reference ball path
            dx_t = np.outer(gvec, da)                        # (T,2) per-token displacement, dir=da
            cU = {}
            for L in layers:
                slab_dim = dH[L].size // T
                pred = np.empty((T, slab_dim), dtype=np.float64)
                for t in range(T):
                    pb = vo.transfield_posbasis(pa[t])
                    feat = np.concatenate([dx_t[t], np.outer(dx_t[t], pb).reshape(-1)])
                    pred[t] = feat @ Btf[L]
                cU[L] = pred.reshape(-1)
            phi = vo.command_features(aa, ab)               # ridge_rich reference uses base command features
        elif trajcanon:
            # synthesize in the per-frame trajectory-aligned frame, then un-roll each frame back.
            cmd = vo.command_features(aa, ab)
            sh = vo.traj_shifts(vo.clip_frame_positions(sa, grid[0]), grid)
            cU = {L: vo.roll_layer_frames((cmd @ Wu[L]) @ Ucanon[L], grid, -sh) for L in layers}
            phi = cmd
        elif canon_appc:
            # combined: [command || appc(H_a)] -> U_canon coords, reconstruct centred, roll back to start.
            cmd = vo.command_features(aa, ab)
            sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
            ca = {L: (np.asarray(sa["layers"][L], dtype=np.float64).mean(0) - appc_mean[L]) @ appc_basis[L].T
                  for L in layers}
            cU = {L: vo.roll_layer((np.concatenate([cmd, ca[L]]) @ Wu[L]) @ Ucanon[L], grid, (-sh[0], -sh[1]))
                  for L in layers}
            phi = cmd
        elif canon:
            # synthesize in the ball-start-centred frame, then roll back to this scene's start cell.
            cmd = vo.command_features(aa, ab)
            sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
            cU = {L: vo.roll_layer((cmd @ Wu[L]) @ Ucanon[L], grid, (-sh[0], -sh[1])) for L in layers}
            phi = cmd  # ridge_rich reference uses the base command features
        elif appc:
            # appc(H_a) read from the reference clip (mean-pool over tokens -> project on appearance PCs);
            # cmd edit is the full-D appearance-conditioned ridge [command || appc] @ B_appc.
            cmd = vo.command_features(aa, ab)
            ca = {L: (np.asarray(sa["layers"][L], dtype=np.float64).mean(0) - appc_mean[L]) @ appc_basis[L].T
                  for L in layers}
            cU = {L: np.concatenate([cmd, ca[L]]) @ Bappc[L] for L in layers}
            phi = cmd  # ridge_rich reference uses the base command features
        else:
            phi = feat_fn(aa, ab)
            cU = {L: (phi @ Wu[L]) @ Ubasis[L][: Wu[L].shape[1]] for L in layers}
        for g in cmd_scales:
            edits[f"cmd_U8_s{g:g}"] = {L: g * cU[L] for L in layers}
        if spd_scales:  # split gain: rows [:npca] = PCA part, rows [npca:] = supervised speed axes
            cpca = {L: (phi @ Wu[L])[:npca] @ Ubasis[L][:npca] for L in layers}
            cspd = {L: (phi @ Wu[L])[npca:] @ Ubasis[L][npca: Wu[L].shape[1]] for L in layers}
            for g in cmd_scales:
                for gm in spd_scales:
                    edits[f"cmd_U8_s{g:g}_m{gm:g}"] = {L: g * (cpca[L] + gm * cspd[L]) for L in layers}
        edits["ridge_rich"] = {L: phi @ Brich[L] for L in layers}
        for k in ks:
            edits[f"subspace_U{k}"] = {L: vo.project(dH[L], Ubasis[L][:k]) for L in layers}
            edits[f"random{k}"] = {L: vo.project(dH[L], Rbasis[L][k]) for L in layers}

        if n < args.viz_scenes:
            cU_edit = {L: args.viz_gain * cU[L] for L in layers}
            f_uns = _decode_frames(decoder, Ha, grid)
            f_steer = _decode_frames(decoder, _apply_edit(Ha, cU_edit, device), grid)
            f_tgt = _decode_frames(decoder, _to_dev(sb, layers, device), grid)
            if all(f is not None for f in (f_uns, f_steer, f_tgt)):
                _save_viz(out / f"viz_scene{s:05d}.png", [f_uns, f_steer, f_tgt],
                          [f"unsteered a_a=({aa[0]:.4f},{aa[1]:.4f})",
                           f"cmd-U8 x{args.viz_gain:g}",
                           f"target H_b a_b=({ab[0]:.4f},{ab[1]:.4f})"])

        targets.append(ab)
        sc_row = {"a_a": aa.tolist(), "v_b": ab.tolist()}  # v_b key = target accel (for calibrate_cmd_gain)
        for m in methods:
            Hstar = _apply_edit(Ha, edits[m], device)
            meas = _decode_accel(decoder, Hstar, grid)
            decoded[m].append([meas["acc_x"], meas["acc_y"]])
            sc_row[m] = [round(meas["acc_x"], 6), round(meas["acc_y"], 6)]
        per_scene[f"scene{s:05d}"] = sc_row
        print(f"  scene{s:05d}: a_b=({ab[0]:.5f},{ab[1]:.5f}) "
              f"full={tuple(round(x,5) for x in decoded['full_delta'][-1])} "
              f"ridge={tuple(round(x,5) for x in decoded['ridge_global'][-1])}")

    results = {m: _agg(decoded[m], targets) for m in methods}
    summary = {"test_dir": args.test_dir, "checkpoint": args.checkpoint, "layers": layers,
               "quantity": "accel", "n_scenes": len(scene_ids), "ks": ks,
               "target": "a_b (decoded vs target acceleration)",
               "results": results, "per_scene": per_scene}
    (out / "steer2d_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[steer-a] decoded-vs-target acceleration (aggregate):")
    for m in methods:
        r = results[m]
        print(f"  {m:16s} rho_ax={r['rho_ax']!s:>7} rho_ay={r['rho_ay']!s:>7} "
              f"angle_err={r['angle_err_deg']!s:>6}deg mag_ratio={r['mag_ratio']} n={r['n']}")
    print(f"[steer-a] -> {out}/steer2d_summary.json")


if __name__ == "__main__":
    main()
