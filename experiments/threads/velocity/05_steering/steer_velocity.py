

#!/usr/bin/env python
"""Step 2 (velocity-first): steer the VELOCITY subspace, decode, and *see* the ball's speed change.

This is the interim milestone the supervisor asked for: not just "velocity is decodable" (Sonia's paper
already showed that), but **successful steering of the velocity subspace** — add ``alpha * d_v`` to the
latents of a clip, decode with the trained transformer decoder, and verify the decoded ball actually
moves faster / slower / in a new direction.

Direction ``d_v`` is learned on the labelled moving-ball cache for a chosen ``--target``
(``speed`` | ``vel_x`` | ``vel_y``) and applied to every cached layer (``--all_layers``, the default) or
a single layer. ``alpha`` is a fraction of each layer's per-token L2 norm (comparable across layers).

Verification is twofold and **non-circular**:
* **latent readout** — an independent Ridge readout (fit on the labelled cache) predicts the target from
  the steered pooled latents; it should move monotonically with ``alpha``.
* **decoded pixels** — we re-track the ball in the *decoded* frames (intensity-weighted centroid) and
  measure its empirical speed/velocity. This is the visual evidence: the measured decoded speed should
  rise (or fall) with ``alpha``. Filmstrips + side-by-side mp4s are written for eyeballing.

Example
-------
    python experiments/threads/velocity/05_steering/steer_velocity.py \
        --config configs/train/physics_iq_transformer_large.yaml \
        --source_latent_dir outputs/latents/moving_ball_velocity/vjepa2_large \
        --target_latent_dir outputs/latents/moving_ball_velocity/vjepa2_large \
        --checkpoint outputs/runs/moving_ball_decoder/checkpoints/last.pt \
        --target speed --all_layers \
        --alphas -1.0,-0.5,0,0.5,1.0 \
        --output_dir outputs/analysis/moving_ball_velocity/steer_speed --device cuda
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
from src.analysis.ball_tracking import measured_velocity
from src.analysis.intervention import apply_intervention_multi
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--source_latent_dir", required=True,
                   help="LABELLED cache the velocity direction is learned from (moving_ball_velocity)")
    p.add_argument("--target_latent_dir", required=True,
                   help="cache to steer + decode (same as source, or a real cache for transfer)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target", default="speed", choices=["speed", "vel_x", "vel_y"])
    p.add_argument("--method", default="regression", choices=["regression", "diff_means"])
    p.add_argument("--all_layers", action="store_true", help="steer every cached layer (recommended)")
    p.add_argument("--layer", type=int, default=-1, help="single steer/read layer; -1 = deepest")
    p.add_argument("--alphas", default="-1.0,-0.5,0,0.5,1.0",
                   help="sweep as a FRACTION of each layer's per-token norm (negative = slow down)")
    p.add_argument("--num_samples", type=int, default=4, help="how many target clips to steer")
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    device = args.device

    target = LatentDataset(args.target_latent_dir, layers=cfg.encoder.layers)
    # Two DISTINCT layer sets — conflating them caused two separate steering bugs:
    #   * `loaded`  = layers ACTUALLY present in each sample (filtered to cfg.encoder.layers, e.g.
    #     [6,12,18,23]). The steerable/readable set must come from here, else a layer like 0 is in
    #     the metadata but absent from the per-sample `latents` dict -> KeyError (fixed in 66acb03).
    #   * `primed`  = the full metadata layer set (available_layers(), e.g. all 24). Training primed
    #     the decoder's layer_embedding table with THIS set (train_decoder.py uses
    #     available_layers()), so the checkpoint has an embedding per metadata layer. The steer-time
    #     decoder must be primed identically or load_state_dict fails on the extra/missing
    #     layer_embedding.* keys (the 4th bug). Extra primed embeddings for unused layers are simply
    #     never read in the forward pass, so priming the full set is safe.
    loaded = sorted(int(k) for k in target[0]["layers"].keys())
    primed = [int(x) for x in target.available_layers()]
    read_layer = max(loaded) if args.layer < 0 else args.layer
    steer_layers = sorted(loaded) if args.all_layers else [read_layer]

    # 1. per-layer UNIT velocity direction learned on the labelled SOURCE cache.
    unit_dirs: dict[int, torch.Tensor] = {}
    readout = None
    dinfo: dict[str, dict] = {}
    for L in steer_layers:
        d, ro, info = steering.discover_velocity_direction(
            args.source_latent_dir, L, target=args.target, method=args.method)
        unit_dirs[L] = torch.from_numpy(d).to(device)
        dinfo[str(L)] = info
        if L == read_layer:
            readout = ro
    if readout is None:  # read layer not among steer layers
        _, readout, _ = steering.discover_velocity_direction(
            args.source_latent_dir, read_layer, target=args.target, method=args.method)

    # 2. choose target clips to steer (first N).
    steer_idx = list(range(min(args.num_samples, len(target))))

    # 3. build decoder.
    rec0 = target.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(primed)  # full metadata set -> matches the trained checkpoint exactly
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    # 4. steer, decode, and measure (latent readout + pixel-tracked velocity) per clip.
    per_sample: dict[str, list[dict]] = {}
    readout_by_alpha = {a: [] for a in alphas}
    measured_by_alpha = {a: [] for a in alphas}
    for n, i in enumerate(steer_idx):
        batch = latent_collate([target[i]])
        sid = batch["id"][0]
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}
        norms = {L: float(latents[L].norm(dim=-1).mean()) for L in steer_layers}
        scaled = {L: unit_dirs[L] * norms[L] for L in steer_layers}
        if n == 0:
            print(f"[steer_velocity] target={args.target} steer_layers={steer_layers} "
                  f"per-token norms={ {L: round(v, 1) for L, v in norms.items()} }")

        curve, frames_by_alpha = [], {}
        for a in alphas:
            perturbed = apply_intervention_multi(latents, scaled, a)
            pooled = perturbed[read_layer].mean(dim=(0, 1)).cpu().numpy()[None, :]
            pred = float(readout(pooled)[0])
            dec = decoder(perturbed, grid)
            meas = {"speed": float("nan")}
            if dec.frames is not None:
                fr = dec.frames[0].cpu()
                frames_by_alpha[a] = fr
                meas = measured_velocity(fr)
            curve.append({"alpha": a, "readout": pred, **{f"measured_{k}": v for k, v in meas.items()}})
            readout_by_alpha[a].append(pred)
            measured_by_alpha[a].append(meas.get(args.target if args.target != "speed" else "speed",
                                                 meas["speed"]))
        per_sample[sid] = curve

        if frames_by_alpha:
            viz.steering_filmstrip(frames_by_alpha, out / f"{sid}_filmstrip.png")
            base = frames_by_alpha[min(alphas, key=abs)]
            for a in (min(alphas), max(alphas)):
                if a in frames_by_alpha:
                    viz.panel_video(base, frames_by_alpha[a], out / f"{sid}_alpha{a:g}.mp4",
                                    fps=cfg.data.fps)

    # 5. aggregate controllability: readout(alpha) and *measured decoded* velocity(alpha).
    def _curve(d):
        return [{"alpha": a, "value": float(np.nanmean(v))} for a, v in sorted(d.items())]
    ro_curve = _curve(readout_by_alpha)
    me_curve = _curve(measured_by_alpha)
    rho_ro = float(spearmanr([r["alpha"] for r in ro_curve], [r["value"] for r in ro_curve]).statistic)
    finite = [(r["alpha"], r["value"]) for r in me_curve if np.isfinite(r["value"])]
    rho_me = (float(spearmanr([a for a, _ in finite], [v for _, v in finite]).statistic)
              if len(finite) >= 2 else float("nan"))

    viz.steering_sweep_plot(
        {"latent readout": [{"alpha": r["alpha"], "readout": r["value"]} for r in ro_curve],
         "decoded (pixel-tracked)": [{"alpha": r["alpha"], "readout": r["value"]} for r in me_curve]},
        out / "velocity_controllability.png",
        title=f"Velocity steering ({args.target}, {'all layers' if args.all_layers else f'L{read_layer}'})",
        ylabel=args.target)

    summary = {
        "target": args.target, "method": args.method, "all_layers": bool(args.all_layers),
        "steer_layers": steer_layers, "read_layer": int(read_layer),
        "alphas_fraction_of_norm": alphas, "n_steered": len(steer_idx),
        "source_latent_dir": args.source_latent_dir, "target_latent_dir": args.target_latent_dir,
        "direction_info": dinfo,
        "readout_monotonicity_spearman": round(rho_ro, 4),
        "decoded_measured_monotonicity_spearman": round(rho_me, 4),
        "readout_curve": ro_curve, "decoded_measured_curve": me_curve,
        "per_sample": per_sample,
    }
    (out / "velocity_steering_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / f"velocity_directions_{args.target}.npz",
             **{f"layer{L}": unit_dirs[L].cpu().numpy() for L in steer_layers})
    print(f"[steer_velocity] {args.target}: readout rho={rho_ro:.3f}  "
          f"decoded-measured rho={rho_me:.3f}  over {len(steer_idx)} clips -> {out}")


if __name__ == "__main__":
    main()
