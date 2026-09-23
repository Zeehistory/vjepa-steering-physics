

#!/usr/bin/env python
"""DECODER-IN-THE-LOOP acceleration steering (test-time edit optimization).

Every linear latent operator plateaus at ~14.5deg because it optimizes a latent proxy (cos with the true
Delta H) that the decoder does NOT invert (read != write). This script drops the proxy: it optimizes the
edit e DIRECTLY against the frozen decoder's pixel output. For a held-out scene, using ONLY the reference
latent H_a and the acceleration command a_b (NO H_b), it finds e so the decoded ball follows the known
target trajectory

    x_b(t) = x_a(t) + 1/2 (a_b - a_a) t^2      (ranks share pos0 + v0, so this is exact and command-derivable)

via a differentiable soft-centroid tracker and backprop through the frozen decoder. This can express the
high-rank per-scene footprint the linear command maps cannot, because it uses the decoder itself (which
knows latent->footprint) and searches its input.

Modes:
  subspace  edit = sum_L coords_L @ U_L   (coords optimized; stays in the accel subspace -> renders well,
            few params, not adversarial). init from the canon operator's coords.
  free      edit = full-D per-layer tensor with L2 (expressive upper bound: is the ceiling reachable by ANY
            edit this frozen decoder renders?). init from the canon edit.

Reports the HONEST (non-differentiable) tracked acceleration angle + magnitude vs a_b, before and after
optimization, aggregated over held-out scenes. Leakage-free: no learned params across scenes, no H_b.

    python experiments/threads/acceleration/05_steering/steer_accel_decopt.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../test/vjepa2_large --artifacts_dir .../subspace --checkpoint .../last.pt \
        --output_dir .../steer_decopt --mode subspace --basis global_basis_canon --rank 128 \
        --steps 200 --lr 0.02 --num_scenes 30 --device cuda
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
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def parabola_accel_operator(T: int, device) -> torch.Tensor:
    """Row of the LS pseudo-inverse that maps centroids (T,) -> the quadratic coeff c2 (accel = 2 c2).

    Fit cen(f) ~ c0 + c1 f + c2 f^2 by least squares with design A=[1,f,f^2]. accel = 2 c2 =
    2 * ((A^T A)^-1 A^T)[2] @ cen. Returns the (T,) row so ``accel_axis = 2 * (row @ cen_axis)`` is a
    differentiable linear readout of the decoded acceleration -- the SAME quantity the honest parabola
    tracker measures, so the optimization targets the curvature directly, not the path.
    """
    f = np.arange(T, dtype=np.float64)
    A = np.stack([np.ones_like(f), f, f * f], 1)            # (T,3)
    M = np.linalg.inv(A.T @ A) @ A.T                        # (3,T)
    return torch.tensor(2.0 * M[2], dtype=torch.float32, device=device)  # (T,)


def soft_centroids(frames: torch.Tensor) -> torch.Tensor:
    """Differentiable per-frame ball centroid (T,2)=(x,y) in [0,1] from decoded frames (T,C,H,W).

    The ball is darker than the background (tracker invariant, appearance-robust). Weight each pixel by how
    much darker than the frame mean it is; centroid = weighted mean of the [0,1] pixel coordinates.
    """
    T, C, H, W = frames.shape
    bright = frames.mean(1)                                   # (T,H,W)
    m = bright.mean(dim=(1, 2), keepdim=True)                 # per-frame mean brightness
    w = torch.relu(m - bright)                                # dark pixels (ball) get weight
    w = w + 1e-8
    ys = torch.linspace(0, 1, H, device=frames.device).view(1, H, 1)
    xs = torch.linspace(0, 1, W, device=frames.device).view(1, 1, W)
    denom = w.sum(dim=(1, 2))                                 # (T,)
    cx = (w * xs).sum(dim=(1, 2)) / denom
    cy = (w * ys).sum(dim=(1, 2)) / denom
    return torch.stack([cx, cy], dim=1)                      # (T,2)


def _decode(decoder, latents, grid):
    return decoder(latents, grid).frames  # (1,T,C,H,W) or None -> here [0]


@torch.no_grad()
def _honest_accel(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    if fr is None:
        return [float("nan"), float("nan")]
    m = measured_acceleration(fr[0].cpu())
    return [m["acc_x"], m["acc_y"]]


@torch.no_grad()
def _frames(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    return None if fr is None else fr[0].cpu().numpy()          # (T,C,H,W)


def _save_filmstrip(path, rows, titles, max_frames=8):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = rows[0].shape[0]
    idx = np.linspace(0, T - 1, min(max_frames, T)).round().astype(int)
    nr, nc = len(rows), len(idx)
    fig, axes = plt.subplots(nr, nc, figsize=(1.5 * nc, 1.7 * nr), squeeze=False)
    for r, (frames, title) in enumerate(zip(rows, titles)):
        for c, fi in enumerate(idx):
            ax = axes[r][c]
            ax.imshow(np.transpose(frames[fi], (1, 2, 0)).clip(0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title, fontsize=8)
            if r == 0:
                ax.set_title(f"t={fi}", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _save_gif(path, rows, titles):
    """Side-by-side animated gif of the conditions over time (best-effort; skipped if imageio missing)."""
    try:
        import imageio
    except Exception:
        return
    T = rows[0].shape[0]
    frames = []
    for t in range(T):
        strip = np.concatenate([np.transpose(r[t], (1, 2, 0)).clip(0, 1) for r in rows], axis=1)
        frames.append((strip * 255).astype(np.uint8))
    imageio.mimsave(path, frames, duration=0.15)


def _agg(dec, tgt):
    d, t = np.asarray(dec), np.asarray(tgt)
    ok = np.isfinite(d).all(1)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d))}
    cos = (d * t).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    md, mt = np.linalg.norm(d, axis=1), np.linalg.norm(t, axis=1)
    return {"n": int(len(d)), "angle_err_deg": round(float(ang.mean()), 2),
            "mag_ratio_median": round(float(np.median(md / (mt + 1e-12))), 3),
            "mag_corr_r": round(float(np.corrcoef(md, mt)[0, 1]), 3)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["subspace", "free"], default="subspace")
    p.add_argument("--basis", default="global_basis_canon", help="subspace mode: basis file prefix in art dir")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--init", choices=["canon", "zero"], default="canon")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--l2", type=float, default=1e-3, help="free mode: L2 on the edit")
    p.add_argument("--anchor", type=float, default=0.2,
                   help="weight on the trajectory-anchor term (keeps the ball on the path; accel error is "
                        "the primary, relative loss)")
    p.add_argument("--basis_dir", default="", help="dir for the subspace basis (default = artifacts_dir)")
    p.add_argument("--num_scenes", type=int, default=30)
    p.add_argument("--scene_start", type=int, default=0, help="slice sorted scenes [start:end] for sharding")
    p.add_argument("--scene_end", type=int, default=0, help="0 = to the end")
    p.add_argument("--dump_dir", default="", help="if set, save each optimized free edit (fp16) + command "
                   "here as scene{ID}.npz -- the supervised distillation targets for amortization")
    p.add_argument("--viz_scenes", type=int, default=0,
                   help="save unsteered/decopt-steered/target filmstrip PNGs (+gif) for the first N scenes")
    p.add_argument("--verbose", action="store_true", help="print per-step trajectory loss")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    dev = args.device

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    _all = sorted(scenes)
    _all = _all[args.scene_start: (args.scene_end or len(_all))]
    scene_ids = _all[: args.num_scenes]
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"[decopt] {len(scenes)} test scenes, optimizing {len(scene_ids)}; layers={layers}; "
          f"mode={args.mode} rank={args.rank} steps={args.steps} lr={args.lr}", flush=True)

    # canon operator (for init + as the baseline we must beat). Optional: datasets without the canon
    # artifacts (e.g. gravity) fall back to zero-init and an unsteered baseline.
    have_canon = all((art / f"global_basis_canon_L{L}.npy").exists() and
                     (art / f"cmd_Wu_canon_L{L}.npy").exists() for L in layers)
    init_mode = args.init if have_canon else "zero"
    if have_canon:
        Wu = {L: np.load(art / f"cmd_Wu_canon_L{L}.npy").astype(np.float64) for L in layers}
        Ucanon_np = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
    else:
        Wu = Ucanon_np = None
        print("[decopt] no canon artifacts -> init=zero, baseline=unsteered", flush=True)
    if args.mode == "subspace":
        bdir = Path(args.basis_dir) if args.basis_dir else art
        Ub = {L: torch.tensor(np.load(bdir / f"{args.basis}_L{L}.npy")[: args.rank], dtype=torch.float32,
                              device=dev) for L in layers}

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=dev)
    for pm in decoder.parameters():
        pm.requires_grad_(False)

    decoded_init, decoded_opt, targets, per_scene = [], [], [], {}
    for n, s in enumerate(scene_ids):
        if dump_dir is not None and (dump_dir / f"scene{s:05d}.npz").exists():
            print(f"  scene{s:05d} already dumped, skip", flush=True)
            continue
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        da = ab - aa
        Ha = _to_dev(sa, layers, dev)

        # Target per-frame trajectory (command-derivable): x_b(f) = x_a(f) + da * f(f-1)/2.
        #
        # NOT 1/2 da f^2. The generator integrates the DISCRETE roll-out vel_t = v0 + acc*t,
        # pos_{t+1} = pos_t + vel_t, i.e. pos_t = pos0 + v0*t + acc*t(t-1)/2
        # (src/data/moving_ball.py:550-558 `_accel_rel_positions`). The continuous-time 1/2 da f^2 used
        # here through 2026-08-25 differs by da*f/2 -- a spurious velocity offset growing to ~7.7px by
        # f=15, which biased the anchor term (it does not bias the c2 readout, so the reported metric
        # was unaffected). Every decopt number predating 2026-08-26 was optimized against the old target.
        refpos = vo.clip_positions(sa)                       # (F,2)
        F = refpos.shape[0]
        ff = np.arange(F, dtype=np.float64)
        tgt = refpos + np.outer(ff * (ff - 1.0) / 2.0, da)   # (F,2) normalized (x,y)
        tgt_t = torch.tensor(tgt, dtype=torch.float32, device=dev)

        # canon init edit (flat per layer); zeros when canon artifacts are unavailable
        if have_canon:
            cmd = vo.command_features(aa, ab)
            sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
            canon_edit = {L: torch.tensor(
                vo.roll_layer((cmd @ Wu[L]) @ Ucanon_np[L], grid, (-sh[0], -sh[1])),
                dtype=torch.float32, device=dev) for L in layers}
        else:
            canon_edit = {L: torch.zeros(Ha[L].shape[1] * Ha[L].shape[2], device=dev) for L in layers}

        # parameters
        if args.mode == "subspace":
            # init coords so that coords @ Ub ~= canon edit  (least-squares proj of canon edit onto Ub)
            coords = {}
            for L in layers:
                ce = canon_edit[L]
                c0 = (Ub[L] @ ce) if init_mode == "canon" else torch.zeros(Ub[L].shape[0], device=dev)
                coords[L] = c0.clone().detach().requires_grad_(True)
            params = list(coords.values())
        else:
            eparam = {L: (canon_edit[L].clone() if init_mode == "canon"
                          else torch.zeros_like(canon_edit[L])).detach().requires_grad_(True) for L in layers}
            params = list(eparam.values())

        def build_edit():
            if args.mode == "subspace":
                return {L: coords[L] @ Ub[L] for L in layers}
            return {L: eparam[L] for L in layers}

        def apply(edit):
            return {L: Ha[L] + edit[L].view(1, Ha[L].shape[1], Ha[L].shape[2]) for L in layers}

        # honest tracked accel of the INIT edit (canon)
        decoded_init.append(_honest_accel(decoder, apply({L: canon_edit[L] for L in layers}), grid))

        ab_t = torch.tensor(ab, dtype=torch.float32, device=dev)
        ab_sq = float(ab @ ab) + 1e-10
        opt = torch.optim.Adam(params, lr=args.lr)
        log_every = max(1, args.steps // 5)
        acc0 = None
        Mrow = None
        for it in range(args.steps):
            opt.zero_grad()
            edit = build_edit()
            fr = _decode(decoder, apply(edit), grid)
            if fr is None:
                break
            cen = soft_centroids(fr[0])                       # (T,2)
            if Mrow is None or Mrow.shape[0] != cen.shape[0]:
                Mrow = parabola_accel_operator(cen.shape[0], dev)   # (T,)
            accel_dec = torch.stack([Mrow @ cen[:, 0], Mrow @ cen[:, 1]])  # (2,); Mrow already = 2*c2 row
            # accel error (primary, relative) + trajectory anchor (keeps ball on the path)
            acc_err = ((accel_dec - ab_t) ** 2).sum() / ab_sq
            Tt = min(cen.shape[0], tgt_t.shape[0])
            traj_loss = ((cen[:Tt] - tgt_t[:Tt]) ** 2).sum()
            loss = acc_err + args.anchor * traj_loss + (args.l2 * sum((edit[L] ** 2).sum() for L in layers)
                                                        if args.mode == "free" else 0.0)
            loss.backward()
            opt.step()
            if acc0 is None:
                acc0 = float(acc_err)
            if args.verbose and (it % log_every == 0 or it == args.steps - 1):
                with torch.no_grad():
                    print(f"    [s{s:05d} it{it:3d}] acc_err={float(acc_err):.3f} traj={float(traj_loss):.4f} "
                          f"accel_dec=({float(accel_dec[0]):.5f},{float(accel_dec[1]):.5f}) "
                          f"a_b=({ab[0]:.5f},{ab[1]:.5f})", flush=True)

        with torch.no_grad():
            fin_edit = build_edit()
            meas = _honest_accel(decoder, apply({L: fin_edit[L].detach() for L in layers}), grid)
        decoded_opt.append(meas)
        if dump_dir is not None:
            np.savez(dump_dir / f"scene{s:05d}.npz",
                     a_a=aa.astype(np.float32), a_b=ab.astype(np.float32),
                     opt=np.asarray(meas, dtype=np.float32),
                     acc_err0=np.float32(acc0 if acc0 is not None else -1),
                     acc_errF=np.float32(float(acc_err.detach())),
                     **{f"L{L}": fin_edit[L].detach().cpu().numpy().astype(np.float16) for L in layers})
        if n < args.viz_scenes:
            f_uns = _frames(decoder, Ha, grid)
            f_steer = _frames(decoder, apply({L: fin_edit[L].detach() for L in layers}), grid)
            f_tgt = _frames(decoder, _to_dev(sb, layers, dev), grid)
            if all(f is not None for f in (f_uns, f_steer, f_tgt)):
                rows = [f_uns, f_steer, f_tgt]
                titles = [f"unsteered a_a=({aa[0]:.4f},{aa[1]:.4f})",
                          f"decopt-steered opt=({meas[0]:.4f},{meas[1]:.4f})",
                          f"target H_b a_b=({ab[0]:.4f},{ab[1]:.4f})"]
                _save_filmstrip(out / f"viz_scene{s:05d}.png", rows, titles)
                _save_gif(out / f"viz_scene{s:05d}.gif", rows, titles)
        targets.append(ab)
        per_scene[f"scene{s:05d}"] = {"a_b": ab.tolist(),
                                      "init_canon": [round(x, 6) for x in decoded_init[-1]],
                                      "opt": [round(x, 6) for x in meas],
                                      "acc_err0": round(float(acc0) if acc0 is not None else -1, 4),
                                      "acc_errF": round(float(acc_err.detach()), 4)}
        print(f"  scene{s:05d} a_b=({ab[0]:.4f},{ab[1]:.4f}) init={tuple(round(x,4) for x in decoded_init[-1])} "
              f"opt={tuple(round(x,4) for x in meas)} acc_err {float(acc0) if acc0 is not None else -1:.3f}->"
              f"{float(acc_err.detach()):.3f}", flush=True)

    res = {"init_canon": _agg(decoded_init, targets), "optimized": _agg(decoded_opt, targets)}
    summary = {"mode": args.mode, "rank": args.rank, "steps": args.steps, "lr": args.lr,
               "n_scenes": len(scene_ids), "checkpoint": args.checkpoint,
               # Full knob set + scene ids. The 5.07deg run is not reproducible from its own summary --
               # --anchor, --l2 and --init were never recorded, and its scene ids (00000-00029) turn out
               # to be the VAL half, so it was never comparable to a test-half number.
               "args": {k: v for k, v in vars(args).items() if k != "overrides"},
               "scene_ids": [int(s) for s in scene_ids],
               "traj_target": "x_a(f) + da*f(f-1)/2 (discrete roll-out; was 0.5*da*f^2 before 2026-08-26)",
               "references": {"canon_committed": 14.46, "full_delta_ceiling": 10.07},
               "results": res, "per_scene": per_scene}
    (out / "decopt_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[decopt] INIT canon: {res['init_canon']}")
    print(f"[decopt] OPTIMIZED : {res['optimized']}")
    print(f"[decopt] -> {out}/decopt_summary.json", flush=True)


if __name__ == "__main__":
    main()
