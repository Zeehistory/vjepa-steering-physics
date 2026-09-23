

#!/usr/bin/env python
"""Look at what the spin decoder actually renders, next to the truth. Run when the gate fails.

The gate reports numbers; this reports pixels. When ``decoder_gate.py`` says omega and speed are both
uncorrelated with truth, there are two very different explanations and no scalar distinguishes them:

  * the decoder is under-trained / broken, and renders a blur that tracks nothing;
  * the decoder renders a clean ball but the READOUT is wrong on decoded frames (thresholds tuned on
    crisp renders, colour shifted, marker dimmed below the mask cut).

A side-by-side of ground-truth and decoded frames settles it in one glance, which is why this exists
rather than another statistic. Rows alternate GT / decoded for a few held-out clips; the tracked ball
centroid and marker centroid are overlaid on both, so a readout failure shows up as markers landing in
the wrong place on a perfectly good image.

    PYTHONPATH=. python experiments/threads/restitution-spin/07_eval/diag_decode_sheet.py --config ... --test_dir ... \
        --checkpoint ... --output_dir ...
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
from pathlib import Path

import sys
from pathlib import Path as _P

import numpy as np
import torch

from src.analysis import spin_tracking as st
from src.analysis.ball_tracking import ball_centroids
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_clips", type=int, default=3)
    p.add_argument("--n_frames", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers, max_cached_shards=1)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    dec = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(args.device).eval()
    if hasattr(dec, "prime_layers"):
        dec.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, dec, map_location=args.device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrow = 2 * args.n_clips
    fig, axes = plt.subplots(nrow, args.n_frames, figsize=(2.1 * args.n_frames, 2.1 * nrow))
    for ci in range(args.n_clips):
        sam = ds[ci * 7]
        grid = tuple(int(x) for x in sam["grid"])
        H = {int(k): v.to(args.device) for k, v in latent_collate([sam])["layers"].items()
             if int(k) in layers}
        dec_fr = dec(H, grid).frames[0].cpu().clamp(0, 1)
        gt = sam["frames"].float()
        if gt.max() > 1.5:
            gt = gt / 255.0

        idx = np.linspace(0, dec_fr.shape[0] - 1, args.n_frames).astype(int)
        for tag, fr, r in (("GT", gt, 2 * ci), ("DEC", dec_fr, 2 * ci + 1)):
            bc = ball_centroids(fr)
            mc = st.marker_centroids(fr)
            warm = float((fr[:, 0] - fr[:, 2] - st.MARKER_RB_THRESH).clamp(min=0).sum())
            for k, t in enumerate(idx):
                ax = axes[r, k]
                ax.imshow(fr[t].permute(1, 2, 0).numpy())
                if np.isfinite(bc[t]).all():
                    ax.plot(bc[t, 0] * fr.shape[-1], bc[t, 1] * fr.shape[-2], "c+", ms=9, mew=2)
                if np.isfinite(mc[t]).all():
                    ax.plot(mc[t, 0] * fr.shape[-1], mc[t, 1] * fr.shape[-2], "rx", ms=8, mew=2)
                ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel(f"clip{ci} {tag}\nwarm={warm:.0f}", fontsize=7)
    fig.suptitle("ground truth vs decoded (cyan + = ball centroid, red x = marker centroid)",
                 fontsize=11)
    fig.savefig(out / "decode_sheet.png", dpi=120, bbox_inches="tight")
    print(f"wrote {out/'decode_sheet.png'}")


if __name__ == "__main__":
    main()
