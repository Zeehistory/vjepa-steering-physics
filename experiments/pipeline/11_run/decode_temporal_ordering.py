

#!/usr/bin/env python
"""EXP 1 (user 2026-07-07): verify the latent's temporal-token -> video-frame ordering.

The latent is 8 TEMPORAL tokens x 256 spatial x 1024 (tubelet size 2). The hypothesis: latent token t
governs video frames 2t, 2t+1 (Z_0 -> frames 0,1 ; ... ; Z_7 -> frames 14,15). We test it CAUSALLY:
perturb one temporal slab Z_t (in every decoder-input layer) and measure which decoded frames change.

For each test clip and each t: decode a baseline, then decode with slab t perturbed (gaussian noise scaled
to that slab's per-channel std, or zeroed), and record the per-frame change c[t, f] = mean_pixels
|frame_f(perturbed) - frame_f(baseline)|. Averaged over clips + seeds and row-normalized, a clean
2t/2t+1-banded diagonal confirms the ordering (and thus that the Exp-3 double-integration writes curvature
to the right frames). Saves the 8x16 matrix + a heatmap PNG.

    python experiments/pipeline/11_run/decode_temporal_ordering.py --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir .../test/vjepa2_large --checkpoint .../last.pt --output_dir .../temporal_order \
        --num_clips 8 --mode gaussian --scale 1.0 --seeds 2
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
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def _frames(decoder, latents, grid):
    fr = decoder(latents, grid).frames
    return None if fr is None else fr[0].cpu().numpy()   # (F,C,H,W)


def perturb(latents, t, grid, mode, scale, rng, device):
    """Return a copy of latents with temporal slab t perturbed in every layer."""
    T, H, W = grid
    n = H * W
    sl = slice(t * n, (t + 1) * n)
    out = {}
    for L, x in latents.items():
        y = x.clone()
        block = y[:, sl, :]
        if mode == "zero":
            block = torch.zeros_like(block)
        elif mode == "mean":
            block = block.mean(dim=1, keepdim=True).expand_as(block).clone()
        else:  # gaussian
            sd = block.std(dim=1, keepdim=True)           # per-channel std within the slab
            noise = torch.tensor(rng.standard_normal(tuple(block.shape)), dtype=block.dtype, device=device)
            block = block + scale * sd * noise
        y[:, sl, :] = block
        out[L] = y
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_clips", type=int, default=8)
    p.add_argument("--mode", choices=["gaussian", "zero", "mean"], default="gaussian")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())

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

    n_clips = min(args.num_clips, len(ds))
    grid = tuple(int(x) for x in ds[0]["grid"])
    T = grid[0]
    mat = np.zeros((T, 0))            # filled after first decode gives F
    counts = 0
    mat_sum = None
    for i in range(n_clips):
        sample = ds[i]
        latents = _to_dev(sample, layers, dev)
        base = _frames(decoder, latents, grid)
        if base is None:
            continue
        F = base.shape[0]
        if mat_sum is None:
            mat_sum = np.zeros((T, F))
        for t in range(T):
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * i + 17 * t + seed)
                pl = perturb(latents, t, grid, args.mode, args.scale, rng, dev)
                fr = _frames(decoder, pl, grid)
                if fr is None:
                    continue
                diff = np.abs(fr - base).reshape(F, -1).mean(axis=1)   # (F,)
                mat_sum[t] += diff
        counts += 1
        print(f"  clip {i}: done ({counts})", flush=True)

    mat = mat_sum / max(1, counts * args.seeds)     # (T, F) mean absolute per-frame change
    row_norm = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)
    peak_frame = row_norm.argmax(axis=1)
    expected = np.array([min(2 * t, mat.shape[1] - 1) for t in range(T)])
    banded = float(np.mean(np.abs(peak_frame - expected) <= 1))       # within +-1 frame of 2t

    # heatmap
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(row_norm, aspect="auto", cmap="magma")
        ax.set_xlabel("decoded frame f"); ax.set_ylabel("perturbed latent token t")
        ax.set_title(f"per-frame sensitivity (row-normalized), mode={args.mode} scale={args.scale}\n"
                     f"banded(peak within +-1 of 2t)={banded:.2f}")
        for t in range(T):
            ax.plot([expected[t]], [t], "c+", ms=8)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "temporal_order_heatmap.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"  (heatmap skipped: {e})", flush=True)

    summary = {"mode": args.mode, "scale": args.scale, "n_clips": counts, "seeds": args.seeds,
               "T": int(T), "F": int(mat.shape[1]),
               "sensitivity_row_normalized": row_norm.round(4).tolist(),
               "peak_frame_per_token": peak_frame.tolist(),
               "expected_peak_2t": expected.tolist(),
               "banded_fraction": banded}
    (out / "temporal_order_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[order] peak frame per token t: {peak_frame.tolist()}")
    print(f"[order] expected (2t):          {expected.tolist()}")
    print(f"[order] banded fraction (within +-1) = {banded:.2f}")
    print(f"[order] -> {out}/temporal_order_summary.json", flush=True)


if __name__ == "__main__":
    main()
