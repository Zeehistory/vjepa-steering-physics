

#!/usr/bin/env python
"""Step 3: steer a physical quantity in latent space and decode the result with the trained decoder.

Pipeline:

1. **Learn** a unit direction + an independent readout for a quantity (e.g. ``vel``, ``gravity``) from a
   *labelled* latent cache — normally the synthetic-physics cache, which has exact ground truth.
2. **Steer** the latents of clips in a *target* cache (e.g. real Physics-IQ) by ``z' = z + alpha * d``.
3. **Decode** the steered latents to pixels with the trained transformer decoder and save filmstrips /
   panel videos across the ``alpha`` sweep.
4. **Verify** controllability + transfer: the independent readout (fit on the labelled cache) applied to
   the steered target latents should move monotonically with ``alpha``. This is the Step-3 question —
   do synthetic directions transfer to real video?

Example
-------
    python experiments/pipeline/05_steering/steer_decode.py \
        --config configs/train/physics_iq_transformer_large.yaml \
        --source_latent_dir outputs/latents/synthetic_solid/vjepa2_large \
        --target_latent_dir outputs/latents/physics_iq/vjepa2_large \
        --checkpoint outputs/runs/physics_iq_decoder_large/checkpoints/last.pt \
        --variable vel --layer 18 --output_dir outputs/analysis/steer/vel --device cuda
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
from scipy.stats import spearmanr

from src.analysis import steering
from src.analysis import visualization as viz
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--source_latent_dir", required=True, help="labelled cache to learn the direction from")
    p.add_argument("--target_latent_dir", required=True, help="cache to steer + decode (e.g. physics_iq)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variable", default="vel", help="state-key substring: vel, acc, gravity, ...")
    p.add_argument("--layer", type=int, default=-1, help="intervention layer; -1 = deepest available")
    p.add_argument("--alphas", default="-3,-1.5,0,1.5,3")
    p.add_argument("--method", default="regression", choices=["regression", "diff_means"])
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    device = args.device

    target = LatentDataset(args.target_latent_dir, layers=cfg.encoder.layers)
    layer = max(target.available_layers()) if args.layer < 0 else args.layer

    # 1. learn direction + readout on the labelled source cache.
    direction_np, readout = steering.discover_quantity_direction(
        args.source_latent_dir, layer, args.variable, method=args.method)
    direction = torch.from_numpy(direction_np).to(device)

    # 2. build decoder + load checkpoint.
    rec0 = target.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(target.available_layers())
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    # 3 + 4. steer, decode, verify per sample.
    per_sample_curves: dict[str, list[dict]] = {}
    all_pred = {a: [] for a in alphas}
    for i in range(min(args.num_samples, len(target))):
        batch = latent_collate([target[i]])
        sid = batch["id"][0]
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}

        # latent-space readout curve (works without ground truth).
        curve = steering.readout_along_direction(latents, direction, layer, alphas, readout)
        per_sample_curves[sid] = curve
        for r in curve:
            all_pred[r["alpha"]].append(r["readout"])

        # decode steered frames + save a filmstrip (+ panel video for the first sample).
        decoded = steering.decode_intervention(decoder, latents, grid, direction, layer, alphas)
        frames_by_alpha = {a: decoded[a].frames[0].cpu() for a in alphas if decoded[a].frames is not None}
        if frames_by_alpha:
            viz.steering_filmstrip(frames_by_alpha, out / f"{sid}_filmstrip.png")
            if i == 0:
                base = frames_by_alpha[min(alphas, key=abs)]
                for a in (min(alphas), max(alphas)):
                    if a in frames_by_alpha:
                        viz.panel_video(base, frames_by_alpha[a], out / f"{sid}_alpha{a:g}.mp4",
                                        fps=cfg.data.fps)

    # aggregate transfer / controllability curve + monotonicity.
    mean_curve = [{"alpha": a, "readout": float(np.mean(v))} for a, v in sorted(all_pred.items())]
    rho = float(spearmanr([r["alpha"] for r in mean_curve], [r["readout"] for r in mean_curve]).statistic)
    viz.steering_sweep_plot(
        {"mean (target)": mean_curve, **{f"{k}": v for k, v in list(per_sample_curves.items())[:6]}},
        out / "controllability.png",
        title=f"Steering '{args.variable}' @layer{layer} (transfer to target)",
        ylabel=f"predicted {args.variable} (readout)")

    summary = {
        "variable": args.variable, "layer": int(layer), "method": args.method, "alphas": alphas,
        "monotonicity_spearman": round(rho, 4),
        "mean_readout_curve": [{"alpha": r["alpha"], "readout": round(r["readout"], 5)} for r in mean_curve],
        "source_latent_dir": args.source_latent_dir, "target_latent_dir": args.target_latent_dir,
    }
    (out / "steering_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out / f"direction_{args.variable}_layer{layer}.npy", direction_np)
    print(f"[steer_decode] '{args.variable}' @layer{layer}: monotonicity ρ={rho:.3f} -> {out}")


if __name__ == "__main__":
    main()
