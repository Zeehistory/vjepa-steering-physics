

#!/usr/bin/env python
"""DECODER-IN-THE-LOOP angular-velocity steering (test-time edit optimization).

The angular-velocity sibling of ``experiments/threads/acceleration/05_steering/steer_accel_decopt.py``. The linear command / subspace operators
all fail for angular velocity (cmd_U8 held-out rho ~ -0.25, chance) for two reasons: (1) command->Delta H is
null -- per-scene rotation subspaces are ~88deg apart, so there is NO shared "angular-velocity axis" to
synthesize along; and (2) injecting the true Delta H at the 4 probed layers SMEARS the decoded rotation
(even the full_delta ceiling is untrackable on ~half the scenes). BUT the base scenes decode with the
correct rotation rate on 100/100 (the decoder CAN render rotation), so an edit that the decoder renders as
the target spin should exist. This script searches for it directly: optimize a full-D per-layer edit ``e``
against the FROZEN decoder's pixels so the decoded marker spins at the commanded ``omega_b``.

Leakage-free: per held-out scene it uses ONLY the reference latent H_a and the scalar command omega_b (NO
H_b, no learned params shared across scenes). A differentiable soft rotor tracker reads the decoded
orientation every frame and backprops through the frozen decoder into ``e`` (Adam).

Differentiable readout (mirrors ``ball_tracking.measured_angvel`` but gradient-friendly, no unwrap):
  * BODY centroid (dark, red-suppressed) -> rotation CENTRE; MARKER centroid (red weight) -> leading end;
  * per-frame unit direction u_t = (marker - centre) / |.|;
  * inter-frame signed rotation dphi_t = atan2(cross, dot) of (u_t, u_{t+1}) (|dphi| << pi so unambiguous);
  * omega_dec = mean_t dphi_t  -- the SAME slope the honest tracker fits, so we target omega directly.

Loss = (omega_dec - omega_b)^2 / omega_b^2                              (primary, relative)
     + anchor_phase  * mean_t (1 - u_t . u_tgt(t))                       (absolute-orientation anchor)
     + anchor_centre * mean_t |centre_t - centre_a|^2                    (anti-gaming: centre stays fixed)
     + l2 * |e|^2
where u_tgt(t) = R(omega_b * t) u_a(0) is the commanded orientation path (u_a(0), centre_a read once from
the UNSTEERED H_a decode -- command-derivable, no H_b). The centre anchor stops the optimizer faking spin by
translating the whole object; the phase anchor pins the absolute rotation so sign + rate both land.

Reports the HONEST (non-differentiable) tracked omega vs omega_b, before (unsteered) and after optimization,
aggregated over held-out scenes: Pearson rho, sign accuracy, magnitude ratio, MAE -- the same scalar metrics
as ``steer_angvel2d.py`` so the number is directly comparable to the failed linear methods.

    python experiments/threads/angular-velocity/05_steering/steer_angvel_decopt.py --config configs/train/moving_ball_scene_angvel_decoder.yaml \
        --test_dir .../moving_ball_scene_angvel2d/test/vjepa2_large \
        --checkpoint .../moving_ball_scene_angvel2d_decoder_fp/checkpoints/last.pt \
        --output_dir .../steer_decopt --steps 300 --lr 0.05 --num_scenes 30 --device cuda
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
from src.analysis.ball_tracking import measured_angvel
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def soft_rotor(frames: torch.Tensor, red_thresh: float = 0.25):
    """Differentiable rotor readout from decoded frames ``(T,C,H,W)`` in [0,1].

    Returns per-frame body centroid ``(bx,by)`` and marker centroid ``(mx,my)`` (each ``(T,)`` in [0,1]),
    plus the marker mass ``mm`` (T,). Marker weight = redness ``R-max(G,B)``; body weight = darkness with red
    softly suppressed (so the red end does not pull the centre). No hard thresholds on the differentiable
    path -- soft weights keep gradients everywhere.
    """
    T, C, H, W = frames.shape
    r, g, b = frames[:, 0], frames[:, 1], frames[:, 2]        # (T,H,W)
    gray = frames.mean(1)                                     # ~1 bg, <1 object
    dark = (1.0 - gray).clamp(min=0.0)
    redness = (r - torch.maximum(g, b)).clamp(min=0.0)        # marker >> body/bg
    marker_w = redness + 1e-8
    # suppress red pixels in the body weight (soft gate around red_thresh)
    body_w = dark * torch.sigmoid((red_thresh - redness) * 40.0) + 1e-8
    ys = torch.linspace(0, 1, H, device=frames.device).view(1, H, 1)
    xs = torch.linspace(0, 1, W, device=frames.device).view(1, 1, W)

    def _cen(w):
        m = w.sum(dim=(1, 2))                                 # (T,)
        cx = (w * xs).sum(dim=(1, 2)) / m
        cy = (w * ys).sum(dim=(1, 2)) / m
        return cx, cy, m

    bx, by, _ = _cen(body_w)
    mx, my, mm = _cen(marker_w)
    return bx, by, mx, my, mm


def _unit_dir(bx, by, mx, my):
    ux, uy = mx - bx, my - by
    n = torch.sqrt(ux * ux + uy * uy) + 1e-8
    return ux / n, uy / n


def soft_omega(frames: torch.Tensor):
    """Differentiable omega (rad/frame) = mean inter-frame signed rotation of the marker direction."""
    bx, by, mx, my, _ = soft_rotor(frames)
    ux, uy = _unit_dir(bx, by, mx, my)
    cross = ux[:-1] * uy[1:] - uy[:-1] * ux[1:]
    dot = ux[:-1] * ux[1:] + uy[:-1] * uy[1:]
    dphi = torch.atan2(cross, dot)                            # (T-1,), |.| << pi
    return dphi.mean()


# Big-object decoder renders rotation faithfully at these tracker thresholds (gate: rho=0.98, 16/16 valid);
# the measured_angvel defaults (0.5, 0.25) barely fire on the decoded frames (gate rho=0.13). Use the tight
# thresholds for every honest read so the best-by-honest selection isn't sabotaged by NaNs.
HONEST_DARK, HONEST_RED = 0.25, 0.08


@torch.no_grad()
def _honest_omega(decoder, latents, grid, dark=HONEST_DARK, red=HONEST_RED):
    fr = decoder(latents, grid).frames
    if fr is None:
        return float("nan")
    return float(measured_angvel(fr[0].cpu(), darkness_thresh=dark, red_thresh=red)["omega"])


@torch.no_grad()
def _frames(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    return None if fr is None else fr[0].cpu().numpy()


def _save_filmstrip(path, rows, titles, max_frames=8):
    import matplotlib
    matplotlib.use("Agg")
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
    """Scalar omega arrays -> {n, rho, sign_acc, mag_ratio, mae} (matches steer_angvel2d._agg)."""
    d, t = np.asarray(dec, float), np.asarray(tgt, float)
    ok = np.isfinite(d) & np.isfinite(t)
    d, t = d[ok], t[ok]
    if len(d) < 2:
        return {"n": int(len(d))}
    return {"n": int(len(d)), "rho": round(float(np.corrcoef(t, d)[0, 1]), 4),
            "sign_acc": round(float(np.mean(np.sign(d) == np.sign(t))), 3),
            "mag_ratio": round(float(np.median(np.abs(d) / (np.abs(t) + 1e-9))), 3),
            "mae": round(float(np.mean(np.abs(d - t))), 5)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=1e-3, help="L2 on the edit (stay on-manifold)")
    p.add_argument("--anchor_phase", type=float, default=2.0,
                   help="weight on the absolute-orientation anchor u_t . u_tgt(t)")
    p.add_argument("--anchor_centre", type=float, default=5.0,
                   help="weight on keeping the rotation centre fixed (anti-gaming)")
    p.add_argument("--anchor_mass", type=float, default=1.0,
                   help="weight keeping the red-marker mass from dissolving (else honest tracker NaNs)")
    p.add_argument("--eval_every", type=int, default=20,
                   help="eval the HONEST tracker every K steps and keep the best-by-honest-error edit "
                        "(immune to soft-tracker gaming + last-step blowup)")
    p.add_argument("--num_scenes", type=int, default=30)
    p.add_argument("--scene_start", type=int, default=0)
    p.add_argument("--scene_end", type=int, default=0, help="0 = to the end")
    p.add_argument("--viz_scenes", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    _all = sorted(scenes)
    _all = _all[args.scene_start: (args.scene_end or len(_all))]
    scene_ids = _all[: args.num_scenes]
    print(f"[w-decopt] {len(scenes)} test scenes, optimizing {len(scene_ids)}; layers={layers}; "
          f"steps={args.steps} lr={args.lr} anchor_phase={args.anchor_phase} "
          f"anchor_centre={args.anchor_centre} anchor_mass={args.anchor_mass} "
          f"eval_every={args.eval_every}", flush=True)

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

    decoded_uns, decoded_opt, targets, per_scene = [], [], [], {}
    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        wa, wb = vo.clip_angvel(sa), vo.clip_angvel(sb)      # [omega, 0]
        wb_s = float(wb[0])
        Ha = _to_dev(sa, layers, dev)

        # honest unsteered omega (baseline) + target orientation path from the UNSTEERED H_a decode
        with torch.no_grad():
            fr_a = decoder(Ha, grid).frames
            uns_omega = float(measured_angvel(fr_a[0].cpu(), darkness_thresh=HONEST_DARK,
                                              red_thresh=HONEST_RED)["omega"]) if fr_a is not None else float("nan")
            bx0, by0, mx0, my0, mm0 = soft_rotor(fr_a[0])
            ux0, uy0 = _unit_dir(bx0, by0, mx0, my0)
            u0x, u0y = float(ux0[0]), float(uy0[0])           # frame-0 direction (shared phi0)
            cen_a = torch.stack([bx0.mean(), by0.mean()])     # fixed centre estimate (a shares centre with b)
            mm0_mean = float(mm0.mean())                       # unsteered marker mass (keep it from dissolving)
        decoded_uns.append(uns_omega)

        Tt = int(fr_a.shape[1]) if fr_a is not None else cfg.decoder.out_num_frames
        tt = torch.arange(Tt, dtype=torch.float32, device=dev)
        ang = wb_s * tt
        tgt_x = torch.cos(ang) * u0x - torch.sin(ang) * u0y  # R(omega_b t) u0
        tgt_y = torch.sin(ang) * u0x + torch.cos(ang) * u0y

        eparam = {L: torch.zeros(Ha[L].shape[1] * Ha[L].shape[2], device=dev).requires_grad_(True)
                  for L in layers}
        opt = torch.optim.Adam(list(eparam.values()), lr=args.lr)
        wb_sq = wb_s * wb_s + 1e-10
        log_every = max(1, args.steps // 5)
        oerr0 = None
        best = {"herr": float("inf"), "omega": float("nan"), "edit": None, "it": -1}

        def _eval_and_track(it):
            """Honest-tracker eval of the current edit; keep the best-by-honest-error edit."""
            with torch.no_grad():
                lat = {L: Ha[L] + eparam[L].detach().view(1, Ha[L].shape[1], Ha[L].shape[2]) for L in layers}
                ho = _honest_omega(decoder, lat, grid)
            if np.isfinite(ho):
                herr = abs(ho - wb_s)
                if herr < best["herr"]:
                    best.update(herr=herr, omega=float(ho), it=it,
                                edit={L: eparam[L].detach().clone() for L in layers})
            return ho

        for it in range(args.steps):
            opt.zero_grad()
            latents = {L: Ha[L] + eparam[L].view(1, Ha[L].shape[1], Ha[L].shape[2]) for L in layers}
            fr = decoder(latents, grid).frames
            if fr is None:
                break
            bx, by, mx, my, mm = soft_rotor(fr[0])
            ux, uy = _unit_dir(bx, by, mx, my)
            cross = ux[:-1] * uy[1:] - uy[:-1] * ux[1:]
            dot = ux[:-1] * ux[1:] + uy[:-1] * uy[1:]
            omega_dec = torch.atan2(cross, dot).mean()
            omega_err = (omega_dec - wb_s) ** 2 / wb_sq
            phase = (1.0 - (ux * tgt_x + uy * tgt_y)).mean()
            centre = ((bx - cen_a[0]) ** 2 + (by - cen_a[1]) ** 2).mean()
            mass = (torch.relu(mm0_mean - mm) / (mm0_mean + 1e-8)).pow(2).mean()   # marker must stay visible
            l2 = sum((eparam[L] ** 2).sum() for L in layers)
            loss = (omega_err + args.anchor_phase * phase + args.anchor_centre * centre
                    + args.anchor_mass * mass + args.l2 * l2)
            loss.backward()
            opt.step()
            if oerr0 is None:
                oerr0 = float(omega_err.detach())
            if it % args.eval_every == 0 or it == args.steps - 1:
                ho = _eval_and_track(it)
                if args.verbose:
                    print(f"    [s{s:05d} it{it:3d}] omega_err={float(omega_err):.3f} "
                          f"phase={float(phase):.4f} centre={float(centre):.5f} mass={float(mass):.4f} "
                          f"omega_soft={float(omega_dec):+.5f} omega_honest={ho:+.5f} omega_b={wb_s:+.5f}",
                          flush=True)

        # best-by-honest edit; fall back to the last edit if the honest tracker never read finite
        if best["edit"] is not None:
            fin_edit = best["edit"]; meas = best["omega"]
        else:
            fin_edit = {L: eparam[L].detach() for L in layers}
            meas = _honest_omega(decoder, {L: Ha[L] + fin_edit[L].view(1, Ha[L].shape[1], Ha[L].shape[2])
                                           for L in layers}, grid)
        latents = {L: Ha[L] + fin_edit[L].view(1, Ha[L].shape[1], Ha[L].shape[2]) for L in layers}
        decoded_opt.append(meas)
        targets.append(wb_s)

        if n < args.viz_scenes:
            f_uns = _frames(decoder, Ha, grid)
            f_steer = _frames(decoder, latents, grid)
            f_tgt = _frames(decoder, _to_dev(sb, layers, dev), grid)
            if all(f is not None for f in (f_uns, f_steer, f_tgt)):
                _save_filmstrip(out / f"viz_scene{s:05d}.png", [f_uns, f_steer, f_tgt],
                                [f"unsteered w={uns_omega:+.4f}", f"decopt w={meas:+.4f}",
                                 f"target H_b w_b={wb_s:+.4f}"])
                _save_gif(out / f"viz_scene{s:05d}.gif", [f_uns, f_steer, f_tgt],
                          ["unsteered", "decopt", "target"])

        per_scene[f"scene{s:05d}"] = {"omega_b": round(wb_s, 6), "omega_a": round(float(wa[0]), 6),
                                      "unsteered": round(uns_omega, 6), "opt": round(float(meas), 6),
                                      "best_it": best["it"],
                                      "omega_err0": round(float(oerr0) if oerr0 is not None else -1, 4)}
        print(f"  scene{s:05d} w_b={wb_s:+.4f} unsteered={uns_omega:+.4f} opt={meas:+.4f} "
              f"best_it={best['it']} omega_err0={float(oerr0) if oerr0 is not None else -1:.3f}", flush=True)

    res = {"unsteered": _agg(decoded_uns, targets), "optimized": _agg(decoded_opt, targets)}
    summary = {"steps": args.steps, "lr": args.lr, "anchor_phase": args.anchor_phase,
               "anchor_centre": args.anchor_centre, "l2": args.l2, "n_scenes": len(scene_ids),
               "checkpoint": args.checkpoint,
               "references": {"cmd_U8_heldout_rho": -0.25, "full_delta_ceiling_rho": 0.25,
                              "full_delta_ceiling_untrackable": "53/100"},
               "results": res, "per_scene": per_scene}
    (out / "decopt_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[w-decopt] UNSTEERED: {res['unsteered']}")
    print(f"[w-decopt] OPTIMIZED: {res['optimized']}")
    print(f"[w-decopt] -> {out}/decopt_summary.json", flush=True)


if __name__ == "__main__":
    main()
