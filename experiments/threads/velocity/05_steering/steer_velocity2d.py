

#!/usr/bin/env python
"""Pixel-level proof for the 2D-velocity subspace / operator (Phases 3-5).

Loads the held-out TEST scene cache + the faithful decoder + the artifacts fit on TRAIN by
``experiments/threads/velocity/04_operators/velocity_subspace.py`` (global PCA basis U, ridge operator B, canonicalized ridge B_canon).
For each test scene it forms the anchor->extreme pair (v_a -> v_b, Delta v = v_b - v_a, Delta H = H_b - H_a)
and steers H_a by several methods, then DECODES and re-tracks the ball's 2D velocity vector:

  full_delta      H_a + Delta H                         per-pair on-manifold baseline (the r~0.95 analog)
  subspace_U[k]   H_a + P_U(Delta H)        U = top-k global PCA basis  (Phase 3: does U preserve it?)
  random[k]       H_a + P_R(Delta H)        random same-rank subspace   (Phase 3 control: should fail)
  ridge_global    H_a + B . Delta v         steer straight from velocity (Phase 4: does it transfer?)
  canon_ridge     H_a + roll^-1( B_canon . Delta v )   (Phase 5: canonicalize -> transfer recovered?)

Reports, per method, the decoded-vs-target velocity vector correlation (vx & vy), mean angle error (deg),
and speed ratio, aggregated over the steered scenes. Decoded velocity is compared to the TARGET v_b.

    python experiments/threads/velocity/05_steering/steer_velocity2d.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../moving_ball_scene_v2d/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_v2d/subspace \
        --checkpoint .../moving_ball_scene_v2d_decoder_fp/checkpoints/last.pt \
        --output_dir outputs/analysis/moving_ball_v2d/steer --ks 2,4,8 --num_scenes 30 --device cuda
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
from src.analysis import visualization as viz
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _apply_edit(Ha, edit_flat, grid, device, write_layers=None):
    """H_a (dict L->(1,Ltok,D) torch) + per-layer flat edit (numpy D,) -> new latent dict.

    ``write_layers`` restricts which layers the edit is actually applied to; layers outside it are
    passed through untouched (the decoder still reads them). ``None`` writes to every layer.
    """
    T, H, W = grid
    out = {}
    for L, t in Ha.items():
        if write_layers is not None and L not in write_layers:
            out[L] = t
            continue
        Ltok = t.shape[1]
        Dd = t.shape[2]
        e = torch.from_numpy(edit_flat[L].reshape(Ltok, Dd).astype(np.float32)).to(device)
        out[L] = t + e.unsqueeze(0)
    return out


def _transport_edit(B_per_t, M_a, M_b, va, vb, grid, layers):
    """Command-only masked-transport edit: per layer, phi(M_a,M_b,va,vb) @ B_t -> flat dH_hat.

    ``B_per_t[L]`` is (n_t, 6, D). Returns {L: flat (n_tok*D,)} consumable by ``_apply_edit``. The masks
    encode WHERE (trajectory geometry); B encodes WHAT (velocity->channel). No H_b is used.
    """
    n_t = grid[0]; n_tok = grid[1] * grid[2]
    phi = vo.transport_features(M_a, M_b, va, vb, grid)   # (n_tok*n_t, 6)
    out = {}
    for L in layers:
        D = B_per_t[L].shape[2]
        flat = np.empty((phi.shape[0], D))
        for t in range(n_t):
            sl = slice(t * n_tok, (t + 1) * n_tok)
            flat[sl] = phi[sl] @ B_per_t[L][t]
        out[L] = flat.reshape(-1)
    return out


@torch.no_grad()
def _decode_vel(decoder, latents, grid, want_frames=False):
    out = decoder(latents, grid)
    if out.frames is None:
        return {"speed": float("nan"), "vel_x": float("nan"), "vel_y": float("nan")}, None
    fr = out.frames[0].cpu()
    return measured_velocity(fr), (fr if want_frames else None)


def _agg(decoded, target):
    """decoded/target: lists of (vx,vy). Return correlation(vx,vy), angle err deg, speed ratio."""
    d = np.asarray(decoded); t = np.asarray(target)
    ok = np.isfinite(d).all(1)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "rho_vx": float("nan"), "rho_vy": float("nan"),
                "angle_err_deg": float("nan"), "speed_ratio": float("nan")}
    rho_vx = float(np.corrcoef(t[:, 0], d[:, 0])[0, 1])
    rho_vy = float(np.corrcoef(t[:, 1], d[:, 1])[0, 1])
    # per-sample angle between decoded and target velocity
    dot = (d * t).sum(1)
    cos = dot / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    sr = np.linalg.norm(d, axis=1) / (np.linalg.norm(t, axis=1) + 1e-12)
    return {"n": int(len(d)), "rho_vx": round(rho_vx, 4), "rho_vy": round(rho_vy, 4),
            "angle_err_deg": round(float(np.nanmean(ang)), 2),
            "speed_ratio": round(float(np.nanmedian(sr)), 3)}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ks", default="2,4,8", help="subspace ranks to test")
    p.add_argument("--dir_bins", default="4,8,16",
                   help="direction-conditioned canonicalized operator: n_bins values to test (needs "
                        "fit_dir_operator.py artifacts; empty string to skip)")
    p.add_argument("--num_scenes", type=int, default=30)
    p.add_argument("--transport_sigma", type=float, default=0.0,
                   help="Gaussian mask width for the transport operator; 0 = read from transport_meta.json")
    p.add_argument("--cmd_scales", default="1.0,1.5,2.0,2.5,3.0",
                   help="gain sweep for cmd_U8 (ridge shrinks magnitude; gain corrects it)")
    p.add_argument("--cmd_pos", action="store_true",
                   help="use the POSITION-AWARE command operator (cmd_Wu_pos*_L*.npy from "
                        "fit_command_operators.py --pos_features): phi = command_features_pos(va,vb,pos_a), "
                        "27-dim. For perspective data where image velocity depends on where the object is. "
                        "Still command-only -- pos_a comes from clip a, never from H_b.")
    p.add_argument("--cmd_std", action="store_true",
                   help="load the STANDARDIZED-ridge command operator (cmd_Wu_std*_L*.npy) instead "
                        "of the published one. The default fit penalises the speed-carrying feature "
                        "columns ~47x harder than the heading-carrying ones, so it steers direction "
                        "and drops magnitude; see LinearLS.solve(standardize=True).")
    p.add_argument("--cmd_ku", type=int, default=8,
                   help="which fitted command operator to load: 8 -> cmd_Wu_L*.npy (default), "
                        "16 -> cmd_Wu_ku16_L*.npy (the U16 fallback). Reconstruction adapts automatically.")
    p.add_argument("--basis_tag", default="",
                   help="use global_basis_{tag}_L*.npy (e.g. 'spd' = U8 + supervised speed axes from "
                        "speed_axis.py) and the matching cmd_Wu*_{tag} operator. subspace_U{k} for k<=8 "
                        "is unchanged (the first 8 rows are U8); pass --ks 2,4,8,11 to get the new ceiling.")
    p.add_argument("--spd_scales", default="",
                   help="with --basis_tag: extra gains for the SPEED-axis part of the cmd edit, e.g. "
                        "'1.5,2,3'. Adds cmd_U8_s{g}_m{gm} = g*(U8 part) + g*gm*(speed-axis part).")
    p.add_argument("--edit_layers", default="",
                   help="Comma-separated encoder layers to actually WRITE to, e.g. '18'. The decoder "
                        "still reads every cached layer; layers outside this set are passed through "
                        "unedited. Empty (default) edits every cached layer, the historical "
                        "behaviour. This is what makes a read-locus/write-locus sweep possible: "
                        "readability is a per-layer property, but every steering result before this "
                        "flag wrote to all four layers at once and so could not attribute the write.")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device
    ks = [int(x) for x in args.ks.split(",")]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    write_layers = ([int(x) for x in args.edit_layers.split(",") if x.strip()]
                    if args.edit_layers.strip() else list(layers))
    unknown = set(write_layers) - set(layers)
    if unknown:
        raise SystemExit(f"--edit_layers names layers not in the cache: {sorted(unknown)} "
                         f"(cached layers are {layers})")
    print(f"[steer2d] {len(scenes)} test scenes, steering {len(scene_ids)}; layers={layers}; "
          f"writing to {write_layers}")

    # artifacts
    Bt = {L: np.load(art / f"ridge_Bt_L{L}.npy").astype(np.float64) for L in layers}
    Bt_canon = {L: np.load(art / f"ridge_canon_Bt_L{L}.npy").astype(np.float64) for L in layers}
    btag = f"_{args.basis_tag}" if args.basis_tag else ""
    Ubasis = {L: np.load(art / f"global_basis{btag}_L{L}.npy").astype(np.float64) for L in layers}
    spd_scales = [float(x) for x in args.spd_scales.split(",") if x.strip()]
    npca = 8   # rows of Ubasis that are the PCA part (the rest are supervised speed axes)
    if args.basis_tag and (art / f"{args.basis_tag.replace('spd', 'speed_axis')}_meta.json").exists():
        npca = int(json.loads((art / f"{args.basis_tag.replace('spd', 'speed_axis')}_meta.json").read_text())["ku"])
    rng = np.random.default_rng(0)
    Rbasis = {L: {k: vo.random_basis(Bt[L].shape[1], k, rng) for k in ks} for L in layers}

    # masked trajectory-transport operator (fit_transport_operator.py artifacts), if present.
    # transport_B_L*.npy is per-temporal-token B (n_t, 6, D); we steer command-only (no H_b) by building
    # the source mask from clip-a's geometry and the target mask by forward-sim from (start, v_b).
    Bt_transport = {}      # {L: (n_t, 6, D)}
    have_transport = False
    try:
        Bt_transport = {L: np.load(art / f"transport_B_L{L}.npy").astype(np.float64) for L in layers}
        have_transport = True
        tsigma = args.transport_sigma
        if tsigma <= 0:  # default to the sigma the operator was fit at
            try:
                tsigma = float(json.loads((art / "transport_meta.json").read_text())["saved_sigma"])
            except (FileNotFoundError, KeyError):
                tsigma = 1.0
        print(f"[steer2d] loaded transport operator (per-t B), sigma={tsigma}, layers {layers}")
    except FileNotFoundError:
        print("[steer2d] no transport-operator artifacts; skipping transport methods")

    # command-only subspace-synthesis operators (fit_command_operators.py artifacts), if present.
    # W_U: command_features -> U8 coords (reconstruct edit = coords @ U8); B_rich: rich command -> full dH.
    Wu = {}; Brich = {}; have_cmd = False
    cmd_ku = args.cmd_ku if args.cmd_ku > 0 else min(Ubasis[L].shape[0] for L in layers)  # 0 = whole basis
    wu_tag = "" if cmd_ku == 8 else f"_ku{cmd_ku}"
    if args.cmd_pos:
        wu_tag = "_pos" + wu_tag
    if args.cmd_std:
        wu_tag = "_std" + wu_tag
    wu_tag = wu_tag + btag
    try:
        Wu = {L: np.load(art / f"cmd_Wu{wu_tag}_L{L}.npy").astype(np.float64) for L in layers}
        Brich = ({} if (args.cmd_pos or args.cmd_std)
                 else {L: np.load(art / f"cmd_Brich_L{L}.npy").astype(np.float64) for L in layers})
        have_cmd = True
        print(f"[steer2d] loaded command operators (W_U, B_rich) for layers {layers}")
    except FileNotFoundError:
        print("[steer2d] no command-operator artifacts; skipping cmd_U8 / ridge_rich methods")

    # direction-conditioned canonicalized operator (fit_dir_operator.py artifacts), if present
    dir_bins = [int(x) for x in args.dir_bins.split(",") if x.strip()]
    Bt_dir = {}  # {N: {b: {L: (2,D)}}}
    for N in list(dir_bins):
        try:
            Bt_dir[N] = {b: {L: np.load(art / f"ridge_dirbin{N}_b{b}_Bt_L{L}.npy").astype(np.float64)
                             for L in layers} for b in range(N)}
        except FileNotFoundError:
            print(f"[steer2d] no dir-operator artifacts for n_bins={N}; skipping")
            dir_bins.remove(N)

    # decoder
    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    # The auxiliary state head must be built at the width it was TRAINED at, which need not match the
    # cache: `_state_keys()` grew 12 -> 14 -> 15 columns over the project, so a decoder trained against
    # an older cache mismatches on `state_head.3` alone. Frame decoding never reads that head, so take
    # the width from the checkpoint and leave everything else untouched. (Same fix as dump_filmstrip.py.)
    _ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _hw = [v for k, v in _ck["model"].items() if k.endswith("state_head.3.weight")]
    if _hw and int(_hw[0].shape[0]) != state_dim:
        print(f"[steer2d] state_head width {int(_hw[0].shape[0])} in checkpoint vs {state_dim} in "
              f"cache -> building the head at the checkpoint's width")
        state_dim = int(_hw[0].shape[0])
    del _ck, _hw
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    transport_methods = ["transport", "transport_oracle", "transport_shuffle"] if have_transport else []
    cmd_scales = [float(x) for x in args.cmd_scales.split(",") if x.strip()]
    cmd_methods = ([f"cmd_U8_s{g:g}" for g in cmd_scales]
                   + [f"cmd_U8_s{g:g}_m{gm:g}" for g in cmd_scales for gm in spd_scales]
                   + ([] if (args.cmd_pos or args.cmd_std) else ["ridge_rich"])) if have_cmd else []
    # ridge_projU8 needs only the always-loaded ridge B + U basis; hybrid needs transport
    synth_methods = ["ridge_projU8"] + (["hybrid_tr_ridge"] if have_transport else [])
    methods = ["full_delta", "ridge_global", "canon_ridge"] + synth_methods + cmd_methods + \
              transport_methods + [f"dircanon{N}" for N in dir_bins] + \
              [f"subspace_U{k}" for k in ks] + [f"random{k}" for k in ks]
    decoded = {m: [] for m in methods}
    # shuffle control: derange scene ids so transport_shuffle uses ANOTHER scene's geometry (wrong
    # placement) with this scene's velocities — isolates whether correct geometry is what helps.
    shuf = {s: scene_ids[(i + 1) % len(scene_ids)] for i, s in enumerate(scene_ids)}
    targets = []
    per_scene = {}

    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb)
        dv = vb - va
        pos = vo.clip_start_pos(sa)
        sh = vo.canon_shift(pos, grid)
        Ha = _to_dev(sa, layers, device)

        dH = {L: vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]) for L in layers}
        edits = {
            "full_delta": dH,
            "ridge_global": {L: dv @ Bt[L] for L in layers},
            "canon_ridge": {L: vo.roll_layer(dv @ Bt_canon[L], grid, (-sh[0], -sh[1])) for L in layers},
            # denoise the command ridge into the decoder-friendly global subspace U8 (free; no refit)
            "ridge_projU8": {L: vo.project(dv @ Bt[L], Ubasis[L][:8]) for L in layers},
        }
        if have_cmd:
            # position-aware phi (27,) for perspective data, else the base (13,). pos is clip a's
            # frame-0 centre, so this stays command-only (no H_b).
            phi = vo.command_features_pos(va, vb, pos) if args.cmd_pos else vo.command_features(va, vb)
            # cmd_U8: synthesize the edit straight in U from the command (targets the subspace ceiling).
            # Ridge shrinks the predicted coord magnitude, so sweep a global gain to undo the shrinkage.
            # ku is read from the fitted operator (P, ku) so this adapts to U8 or a U16 refit unchanged.
            cU8 = {L: (phi @ Wu[L]) @ Ubasis[L][: Wu[L].shape[1]] for L in layers}
            for g in cmd_scales:
                edits[f"cmd_U8_s{g:g}"] = {L: g * cU8[L] for L in layers}
            if spd_scales:  # split gain: rows [:8] = PCA part, rows [8:] = supervised speed axes
                cpca = {L: (phi @ Wu[L])[:npca] @ Ubasis[L][:npca] for L in layers}
                cspd = {L: (phi @ Wu[L])[npca:] @ Ubasis[L][npca: Wu[L].shape[1]] for L in layers}
                for g in cmd_scales:
                    for gm in spd_scales:
                        edits[f"cmd_U8_s{g:g}_m{gm:g}"] = {L: g * (cpca[L] + gm * cspd[L]) for L in layers}
            if Brich:  # B_rich: 13-dim features only, and not fit in the --cmd_std arm
                edits["ridge_rich"] = {L: phi @ Brich[L] for L in layers}
        if have_transport:
            n_t = grid[0]
            pa = vo.clip_positions(sa)                      # clip-a per-frame centers (known at test)
            ca = vo.temporal_token_centers(pa, n_t)
            M_a = vo.gaussian_mask(ca, grid, tsigma)
            # deployable target mask: forward-sim the trajectory from (start, v_b) — no H_b
            cb_sim = vo.temporal_token_centers(vo.forward_sim_positions(pa[0], vb, pa.shape[0]), n_t)
            M_b_dep = vo.gaussian_mask(cb_sim, grid, tsigma)
            # oracle target mask: clip-b's true per-frame centers (upper bound on geometry)
            cb_orc = vo.temporal_token_centers(vo.clip_positions(sb), n_t)
            M_b_orc = vo.gaussian_mask(cb_orc, grid, tsigma)
            edits["transport"] = _transport_edit(Bt_transport, M_a, M_b_dep, va, vb, grid, layers)
            edits["transport_oracle"] = _transport_edit(Bt_transport, M_a, M_b_orc, va, vb, grid, layers)
            # shuffle control: same va,vb but masks from another scene's geometry (wrong placement)
            so = ds[scenes[shuf[s]][sorted(scenes[shuf[s]])[0]]]
            pa_o = vo.clip_positions(so)
            M_a_sh = vo.gaussian_mask(vo.temporal_token_centers(pa_o, n_t), grid, tsigma)
            M_b_sh = vo.gaussian_mask(
                vo.temporal_token_centers(vo.forward_sim_positions(pa_o[0], vb, pa_o.shape[0]), n_t),
                grid, tsigma)
            edits["transport_shuffle"] = _transport_edit(Bt_transport, M_a_sh, M_b_sh, va, vb, grid, layers)
            # hybrid: keep the global command term (the decoder needs it) + add local placement on top
            edits["hybrid_tr_ridge"] = {L: edits["transport"][L] + dv @ Bt[L] for L in layers}
            if n < 6:  # mask-overlay sanity figure on the GT target frame
                try:
                    viz.transport_mask_overlay(sb["frames"], M_a, M_b_dep,
                                               out / f"scene{s:05d}_masks.png")
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] mask overlay scene{s:05d} failed: {e}")
        for N in dir_bins:
            bidx = vo.direction_bin(vb, N)   # condition on TARGET heading
            edits[f"dircanon{N}"] = {
                L: vo.roll_layer(dv @ Bt_dir[N][bidx][L], grid, (-sh[0], -sh[1])) for L in layers}
        for k in ks:
            edits[f"subspace_U{k}"] = {L: vo.project(dH[L], Ubasis[L][:k]) for L in layers}
            edits[f"random{k}"] = {L: vo.project(dH[L], Rbasis[L][k]) for L in layers}

        targets.append(vb)
        keep_frames = {}
        sc_row = {"v_a": va.tolist(), "v_b": vb.tolist()}
        for m in methods:
            Hstar = _apply_edit(Ha, edits[m], grid, device, write_layers)
            want = m in ("full_delta", "ridge_global", "transport") and n < 6
            meas, fr = _decode_vel(decoder, Hstar, grid, want_frames=want)
            decoded[m].append([meas["vel_x"], meas["vel_y"]])
            sc_row[m] = [round(meas["vel_x"], 5), round(meas["vel_y"], 5)]
            if fr is not None:
                keep_frames[m] = fr
        per_scene[f"scene{s:05d}"] = sc_row
        if keep_frames:
            try:  # never let a plotting hiccup lose the numeric results
                viz.steering_filmstrip(
                    {i: keep_frames[m] for i, m in enumerate(keep_frames)},
                    out / f"scene{s:05d}_methods.png")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] filmstrip scene{s:05d} failed: {e}")
        tp = (f" transport={tuple(round(x,4) for x in decoded['transport'][-1])}"
              if have_transport else "")
        print(f"  scene{s:05d}: v_b=({vb[0]:.4f},{vb[1]:.4f}) "
              f"full={tuple(round(x,4) for x in decoded['full_delta'][-1])} "
              f"ridge={tuple(round(x,4) for x in decoded['ridge_global'][-1])}{tp}")

    results = {m: _agg(decoded[m], targets) for m in methods}
    summary = {"test_dir": args.test_dir, "checkpoint": args.checkpoint, "layers": layers,
               "write_layers": write_layers,
               "n_scenes": len(scene_ids), "ks": ks, "target": "v_b (decoded vs target velocity)",
               "results": results, "per_scene": per_scene}
    (out / "steer2d_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[steer2d] decoded-vs-target velocity (aggregate):")
    for m in methods:
        r = results[m]
        print(f"  {m:16s} rho_vx={r['rho_vx']!s:>7} rho_vy={r['rho_vy']!s:>7} "
              f"angle_err={r['angle_err_deg']!s:>6}deg speed_ratio={r['speed_ratio']} n={r['n']}")
    print(f"[steer2d] -> {out}/steer2d_summary.json")


if __name__ == "__main__":
    main()
