

#!/usr/bin/env python
"""Pixel-level magnitude check for the VANILLA edit ``H_a + alpha*(H_b - H_a)`` (accel).

Companion to ``probe_accel_vanilla.py`` (latent). Decodes the interpolated latent for a sweep of alpha and
re-tracks the ball's 2D acceleration (parabola fit, a = 2*c2). Reports, per alpha and aggregated over
held-out test scenes: the decoded-vs-expected acceleration angle error AND the magnitude ratio / magnitude
correlation, where the expected accel at interpolation alpha is ``E(alpha) = a_a + alpha*(a_b - a_a)``.

alpha=1 reproduces ``full_delta`` (the on-manifold ceiling). Sweeping alpha past 1 tests whether the
decoded magnitude keeps scaling (linear latent accel axis) or saturates. This is the decoder's view of the
same question the probe answers in latent space.

    python experiments/threads/acceleration/05_steering/steer_accel_vanilla_alpha.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../test/vjepa2_large --checkpoint .../last.pt \
        --output_dir .../steer_vanilla --alphas 0.5,1.0,1.5,2.0,2.5,3.0 --num_scenes 60 --device cuda
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
        return [float("nan"), float("nan")]
    m = measured_acceleration(out.frames[0].cpu())
    return [m["acc_x"], m["acc_y"]]


def _agg(dec, exp):
    d, e = np.asarray(dec), np.asarray(exp)
    ok = np.isfinite(d).all(1)
    d, e = d[ok], e[ok]
    if len(d) < 2:
        return {"n": int(len(d))}
    cos = (d * e).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(e, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    md, me = np.linalg.norm(d, axis=1), np.linalg.norm(e, axis=1)
    return {"n": int(len(d)), "angle_err_deg": round(float(ang.mean()), 2),
            "mag_ratio_median": round(float(np.median(md / (me + 1e-12))), 3),
            "mag_corr_r": round(float(np.corrcoef(md, me)[0, 1]), 3),
            "mean_decoded_mag": round(float(md.mean()), 6), "mean_expected_mag": round(float(me.mean()), 6)}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alphas", default="0.5,1.0,1.5,2.0,2.5,3.0")
    p.add_argument("--num_scenes", type=int, default=60)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[vanilla-px] {len(scenes)} test scenes, steering {len(scene_ids)}; layers={layers}", flush=True)

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(args.device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=args.device)

    decoded = {a: [] for a in alphas}
    expected = {a: [] for a in alphas}
    for n, s in enumerate(scene_ids):
        ranks = sorted(scenes[s])
        ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        da = ab - aa
        Ha = _to_dev(sa, layers, args.device)
        dH = {L: vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]) for L in layers}
        for a in alphas:
            edit = {L: a * dH[L] for L in layers}
            decoded[a].append(_decode_accel(decoder, _apply_edit(Ha, edit, args.device), grid))
            expected[a].append((aa + a * da).tolist())
        print(f"  scene{s:05d}: a_a=({aa[0]:.4f},{aa[1]:.4f}) a_b=({ab[0]:.4f},{ab[1]:.4f})", flush=True)

    results = {f"{a:g}": _agg(decoded[a], expected[a]) for a in alphas}
    summary = {"edit": "vanilla H_a + alpha*(H_b - H_a)", "quantity": "accel",
               "checkpoint": args.checkpoint, "n_scenes": len(scene_ids), "alphas": alphas,
               "per_alpha": results}
    (out / "vanilla_alpha_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[vanilla-px] decoded-vs-expected accel by alpha:")
    for a in alphas:
        r = results[f"{a:g}"]
        print(f"  alpha={a:<4g} angle={r.get('angle_err_deg')}deg mag_ratio={r.get('mag_ratio_median')} "
              f"mag_corr_r={r.get('mag_corr_r')} |dec|={r.get('mean_decoded_mag')} |exp|={r.get('mean_expected_mag')}",
              flush=True)
    print(f"[vanilla-px] -> {out}/vanilla_alpha_summary.json", flush=True)


if __name__ == "__main__":
    main()
