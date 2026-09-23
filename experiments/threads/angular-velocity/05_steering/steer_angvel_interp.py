#!/usr/bin/env python
"""LATENT-INTERPOLATION controllability of angular velocity (decoder-verified, non-circular).

Free-edit TTO stalls (the latent->omega map is locally stiff: decode(H_a+free_edit) only nudges omega ~15%).
But the decoder renders the full omega continuum faithfully (gate: decode(true H_b) tracks GT omega rho=0.98).
This asks the controllability question directly: decode H_a + alpha*(H_b - H_a) for alpha in [0,1] and read the
HONEST tracker omega. If honest omega tracks the linear target omega(alpha)=omega_a+alpha*(omega_b-omega_a)
across held-out scenes, angular velocity IS a controllable latent direction (the decoder renders every
intermediate rate), and the barrier is purely FINDING the edit (free-TTO), not writability. Uses the gate-
proven tracker thresholds (0.25, 0.08). Non-circular: the judge is the pixel tracker, not a latent probe.
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

import argparse, json
from pathlib import Path
import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_angvel
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

DARK, RED = 0.25, 0.08


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def honest_omega(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    if fr is None:
        return float("nan")
    return float(measured_angvel(fr[0].cpu(), darkness_thresh=DARK, red_thresh=RED)["omega"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_scenes", type=int, default=25)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    dev = args.device
    alphas = [float(a) for a in args.alphas.split(",")]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]

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
    print(f"[interp] {len(sids)} scenes, alphas={alphas}, thr=({DARK},{RED})", flush=True)

    tgt_all, hon_all, alpha_all = [], [], []
    per_alpha = {a: [] for a in alphas}
    for n, s in enumerate(sids):
        ranks = sorted(scenes[s])
        ia = scenes[s][ranks[0]]
        sa = ds[ia]
        grid = tuple(int(x) for x in sa["grid"])
        wa = float(vo.clip_angvel(sa)[0])
        Ha = _to_dev(sa, layers, dev)
        # base rank = smallest |omega|; targets = the other ranks
        for rb in ranks[1:]:
            sb = ds[scenes[s][rb]]
            wb = float(vo.clip_angvel(sb)[0])
            Hb = _to_dev(sb, layers, dev)
            for a in alphas:
                lat = {L: Ha[L] + a * (Hb[L] - Ha[L]) for L in layers}
                ho = honest_omega(decoder, lat, grid)
                tgt = wa + a * (wb - wa)
                if np.isfinite(ho):
                    tgt_all.append(tgt); hon_all.append(ho); alpha_all.append(a)
                    per_alpha[a].append((tgt, ho))
        print(f"  scene {n+1}/{len(sids)} done", flush=True)

    tgt = np.array(tgt_all); hon = np.array(hon_all)
    rho = float(np.corrcoef(tgt, hon)[0, 1])
    sign = float(np.mean(np.sign(hon) == np.sign(tgt)))
    slope = float(np.polyfit(tgt, hon, 1)[0])
    mae = float(np.mean(np.abs(hon - tgt)))
    n_valid = len(tgt)
    print(f"\n=== INTERPOLATION controllability (n={n_valid} decoded points) ===")
    print(f"  honest omega vs target: rho={rho:+.3f} sign_acc={sign:.3f} mag_ratio(slope)={slope:+.3f} MAE={mae:.4f}")
    print("  per-alpha mean |honest - target| and mean honest (should ramp with alpha):")
    for a in alphas:
        arr = np.array(per_alpha[a])
        if len(arr):
            print(f"    alpha={a:.2f}: n={len(arr)} mean_target={arr[:,0].mean():+.3f} "
                  f"mean_honest={arr[:,1].mean():+.3f} MAE={np.mean(np.abs(arr[:,1]-arr[:,0])):.4f}")
    # ceiling = alpha 1.0 only
    c = np.array(per_alpha[alphas[-1]]) if alphas[-1] == 1.0 else None
    if c is not None and len(c) > 2:
        crho = float(np.corrcoef(c[:,0], c[:,1])[0,1])
        print(f"  alpha=1 CEILING: rho={crho:+.3f} (matches the decode-faithfulness gate ~0.98)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(rho=rho, sign_acc=sign, mag_ratio=slope, mae=mae, n=n_valid,
                       alphas=alphas), open(args.out, "w"), indent=2)
        print(f"[interp] wrote {args.out}")


if __name__ == "__main__":
    main()
