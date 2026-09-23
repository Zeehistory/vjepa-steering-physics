#!/usr/bin/env python
"""Does the command-only velocity operator EXTRAPOLATE past its training speed band?

Every reported velocity number is measured on commands drawn from the same band the operator was
fitted on (2-D mixed ``[0.012, 0.024]``/frame, ``rolling_ball3d`` ``[0.010, 0.022]``/frame in image
units). That leaves the obvious question unanswered: the operator is a LINEAR map of the command
features, so it will happily synthesise an edit for a speed it has never seen -- does the latent
actually move that far, and does the decoder draw it?

For each held-out scene this sweeps the COMMANDED speed over a grid that runs from well below the
band to well above it, holding the heading fixed at the scene's fastest rank's heading and the
anchor fixed at its slowest rank, then measures what came out:

  achieved_px    |v| tracked in the decoded pixels             -- the deployable end-to-end number
  achieved_lat   |v| read off the steered LATENT by a linear   -- separates "the edit did not land
                 readout fit on the in-band cached clips          in the latent" from "the decoder
                                                                  cannot draw a ball this fast"

Both readouts are linear/unbiased by construction, so a flat ``achieved`` against a rising
``commanded`` is saturation, not a metric artifact. Nothing is fit on OOD data.

Nothing here is fit on the sweep; it replays the saved operator at the calibrated gain.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def _decode(decoder, latents, grid):
    out = decoder(latents, grid)
    if out.frames is None:
        return None
    return out.frames[0].detach().cpu().float().clamp(0, 1).numpy()


def _flat(sample, layers):
    """Concatenate a sample's per-layer token grids into one flat feature vector."""
    return np.concatenate([vo.layer_flat(sample["layers"][L]).ravel() for L in layers])


def _fit_latent_readout(ds, layers, ridge=1e3):
    """Least-squares H -> v on every cached clip. In-band fit, but LINEAR, so it extrapolates."""
    X, Y = [], []
    for i in range(len(ds)):
        s = ds[i]
        X.append(_flat(s, layers))
        Y.append(np.asarray(vo.clip_velocity(s), dtype=np.float64).reshape(2))
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    # ridge in the n<<d regime via the Gram matrix
    G = Xc @ Xc.T
    A = np.linalg.solve(G + ridge * np.eye(len(G)), Y)      # (n, 2)
    W = Xc.T @ A                                            # (d, 2)
    pred = (X - mu) @ W
    r2 = 1.0 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    return mu, W, float(r2)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gain", type=float, required=True)
    p.add_argument("--cmd_ku", type=int, default=8)
    p.add_argument("--num_scenes", type=int, default=12)
    p.add_argument("--band", type=float, nargs=2, required=True,
                   help="the training speed band in image units/frame, e.g. 0.012 0.024")
    p.add_argument("--speeds", type=float, nargs="*", default=None,
                   help="commanded speeds to sweep; default = 0.25x..3.0x the band, 13 points")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    lo, hi = args.band
    speeds = args.speeds or list(np.round(np.linspace(0.25 * lo, 3.0 * hi, 13), 5))

    cfg = load_config(args.config, args.overrides)
    art = Path(args.artifacts_dir)
    device = args.device

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]

    tag = "" if args.cmd_ku == 8 else f"_ku{args.cmd_ku}"
    Wu = {L: np.load(art / f"cmd_Wu{tag}_L{L}.npy").astype(np.float64) for L in layers}
    Ub = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}

    mu, Wlat, lat_r2 = _fit_latent_readout(ds, layers)
    print(f"[ood] latent readout fitted on {len(ds)} in-band clips: R2={lat_r2:.4f}")

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head_w = [v for k, v in ck["model"].items() if k.endswith("state_head.3.weight")]
    if head_w and int(head_w[0].shape[0]) != state_dim:
        state_dim = int(head_w[0].shape[0])
    del ck
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    rows = []
    for s in scene_ids:
        ranks = sorted(scenes[s])
        q = {r: np.asarray(vo.clip_velocity(ds[scenes[s][r]]), dtype=np.float64).reshape(2)
             for r in ranks}
        mag = {r: float(np.linalg.norm(q[r])) for r in ranks}
        ra = min(mag, key=mag.get)            # anchor = the scene's slowest clip
        rb = max(mag, key=mag.get)            # heading = the scene's fastest clip
        qa = q[ra]
        head = q[rb] / (mag[rb] + 1e-12)

        sa = ds[scenes[s][ra]]
        grid = tuple(int(x) for x in sa["grid"])
        Ha = _to_dev(sa, layers, device)
        flat_a = _flat(sa, layers)

        for sp in speeds:
            qb = head * sp
            phi = vo.command_features(qa, qb)
            edit = {L: args.gain * ((phi @ Wu[L]) @ Ub[L][: Wu[L].shape[1]]) for L in layers}

            Hs, ed_flat = {}, []
            for L in layers:
                e = torch.from_numpy(np.asarray(edit[L], dtype=np.float32)).to(device)
                Hs[L] = Ha[L] + e.reshape(1, Ha[L].shape[1], -1)
                ed_flat.append(np.asarray(edit[L], dtype=np.float64).ravel())
            v_lat = ((flat_a + np.concatenate(ed_flat)) - mu.ravel()) @ Wlat

            frames = _decode(decoder, Hs, grid)
            try:
                m = measured_velocity(torch.from_numpy(frames))
                v_px = np.array([float(m["vel_x"]), float(m["vel_y"])])
                n_valid = float(m.get("n_valid", np.nan))
            except Exception as e:                                   # noqa: BLE001
                v_px, n_valid = np.array([np.nan, np.nan]), np.nan
                print(f"  scene{s:05d} sp={sp}: readout failed: {str(e)[:120]}")

            rows.append({
                "scene": int(s), "rank_a": int(ra), "rank_head": int(rb),
                "anchor_speed": mag[ra], "commanded_speed": float(sp),
                "in_band": bool(lo <= sp <= hi),
                "achieved_px_speed": float(np.linalg.norm(v_px)),
                "achieved_px_x": float(v_px[0]), "achieved_px_y": float(v_px[1]),
                "achieved_lat_speed": float(np.linalg.norm(v_lat)),
                "achieved_lat_x": float(v_lat[0]), "achieved_lat_y": float(v_lat[1]),
                "n_valid": n_valid,
            })
        print(f"  scene{s:05d}: anchor r{ra} |v|={mag[ra]:.4f}, swept {len(speeds)} commands")

    out = {"band": [lo, hi], "speeds": [float(x) for x in speeds], "gain": args.gain,
           "cmd_ku": args.cmd_ku, "layers": layers, "test_dir": args.test_dir,
           "checkpoint": args.checkpoint, "artifacts_dir": str(art),
           "latent_readout_r2_in_band": lat_r2, "n_scenes": len(scene_ids),
           "note": "anchor = scene's slowest rank; heading = scene's fastest rank's heading; "
                   "commanded speed swept. Nothing fitted on OOD data.",
           "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))

    # console summary: median achieved vs commanded, per commanded speed
    print("\n commanded | in-band | median achieved (pixels) | median achieved (latent) | n")
    for sp in speeds:
        sel = [r for r in rows if r["commanded_speed"] == sp]
        px = np.array([r["achieved_px_speed"] for r in sel], dtype=float)
        lt = np.array([r["achieved_lat_speed"] for r in sel], dtype=float)
        ok = np.isfinite(px)
        flag = "  yes  " if (lo <= sp <= hi) else "  OOD  "
        print(f"  {sp:9.5f} |{flag}| {np.nanmedian(px):20.5f}     | {np.median(lt):20.5f}     "
              f"| {int(ok.sum())}")
    print(f"\n[ood] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
