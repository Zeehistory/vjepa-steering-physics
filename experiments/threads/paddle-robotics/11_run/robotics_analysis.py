

#!/usr/bin/env python
"""Step 4: detect achievable-vs-non-achievable latent structure, then steer failures toward success.

1. **Detect** — linear probe accuracy for success vs. failure across layers (with shuffled control).
2. **Direction** — success direction (failure -> success) at the chosen layer.
3. **Steer** — for failed clips, add ``alpha * success_direction`` to the latents and decode the result;
   also interpolate failed -> matched-successful latent trajectories and decode the path.

Works on DROID (real) or the ``robot_toy`` synthetic fallback — both expose success via ``category``.

Example
-------
    python experiments/threads/paddle-robotics/11_run/robotics_analysis.py \
        --config configs/train/physics_iq_transformer_large.yaml \
        --latent_dir outputs/latents/robot_toy/vjepa2_large \
        --checkpoint outputs/runs/robot_decoder/checkpoints/last.pt \
        --output_dir outputs/analysis/robotics --device cuda
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

from src.analysis import robotics_steering as rs
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
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--checkpoint", default=None, help="trained decoder; omit to run detection only")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layer", type=int, default=-1, help="steer layer; -1 = deepest available")
    p.add_argument("--alphas", default="0,1,2,4")
    p.add_argument("--num_steer", type=int, default=3, help="failed clips to steer/decode")
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]

    dataset = LatentDataset(args.latent_dir, layers=cfg.encoder.layers)
    layers = dataset.available_layers()
    steer_layer = max(layers) if args.layer < 0 else args.layer

    # 1. detection across layers.
    detection = [{"layer": li, **rs.achievability_report(args.latent_dir, li)["detection"]} for li in layers]
    viz.steering_sweep_plot(
        {"linear accuracy": [{"alpha": d["layer"], "readout": d["accuracy"]} for d in detection],
         "shuffled ctrl": [{"alpha": d["layer"], "readout": d["ctrl_shuffled_accuracy"]} for d in detection]},
        out / "achievability_by_layer.png",
        title="Success-vs-failure detection by layer", ylabel="accuracy")

    report = rs.achievability_report(args.latent_dir, steer_layer)
    direction_np = report["direction"]
    np.save(out / f"success_direction_layer{steer_layer}.npy", direction_np)

    summary = {"latent_dir": args.latent_dir, "steer_layer": int(steer_layer), "alphas": alphas,
               "detection_by_layer": [{k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in d.items()} for d in detection],
               "n_success": report["n_success"], "n_failure": report["n_failure"]}

    # 2 + 3. steer failed clips (needs a trained decoder).
    if args.checkpoint:
        direction = torch.from_numpy(direction_np).to(args.device)
        rec0 = dataset.records[0]
        enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
        cfg.decoder.state_dim = state_dim
        if cfg.decoder.out_num_frames <= 0:
            cfg.decoder.out_num_frames = cfg.data.num_frames
        decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(args.device).eval()
        if hasattr(decoder, "prime_layers"):
            decoder.prime_layers(layers)
        load_checkpoint(args.checkpoint, decoder, map_location=args.device)

        fail_idx = [i for i in range(len(dataset)) if dataset[i]["category"] == "failure"][:args.num_steer]
        succ_idx = [i for i in range(len(dataset)) if dataset[i]["category"] == "success"]
        for i in fail_idx:
            batch = latent_collate([dataset[i]])
            sid = batch["id"][0]
            grid = tuple(int(x) for x in batch["grid"])
            latents = {int(k): v.to(args.device) for k, v in batch["layers"].items()}
            decoded = steering.decode_intervention(decoder, latents, grid, direction, steer_layer, alphas)
            frames_by_alpha = {a: decoded[a].frames[0].cpu() for a in alphas
                               if decoded[a].frames is not None}
            if frames_by_alpha:
                viz.steering_filmstrip(frames_by_alpha, out / f"{sid}_steer_filmstrip.png")

            # failure -> matched-success trajectory interpolation (decode the path).
            if succ_idx:
                sbatch = latent_collate([dataset[succ_idx[0]]])
                s_lat = {int(k): v.to(args.device) for k, v in sbatch["layers"].items()}
                path = steering.interpolate_trajectories(latents, s_lat, steps=len(alphas))
                interp = {float(t): decoder(p, grid).frames[0].cpu()
                          for t, p in zip(np.linspace(0, 1, len(path)), path)}
                interp = {t: f for t, f in interp.items() if f is not None}
                if interp:
                    viz.steering_filmstrip(interp, out / f"{sid}_interp_to_success.png")
        summary["steered_failures"] = [dataset[i]["id"] for i in fail_idx]

    (out / "robotics_summary.json").write_text(json.dumps(summary, indent=2))
    best = max(detection, key=lambda d: d["accuracy"])
    print(f"[robotics_analysis] best detection acc={best['accuracy']:.3f} @layer{best['layer']} "
          f"(ctrl={best['ctrl_shuffled_accuracy']:.3f}) -> {out}")


if __name__ == "__main__":
    main()
