

#!/usr/bin/env python
"""Step 3 (category steering): turn a fluid clip "solid-like" along a category direction, then decode.

This is the wild experiment: from the Step-1 category probe we get a contrast direction between two
categories (e.g. solid vs fluid). We take a real **fluid** Physics-IQ clip, add ``alpha * direction`` to
its latents, and **decode** the result with the trained transformer decoder — does the scene start to
look/behave solid-like as ``alpha`` grows? Our understanding of the physics subspace is crude, so there
is no guarantee; the decoded filmstrips are the evidence to eyeball.

Two knobs that matter (the probe-weight / single-deepest-layer defaults often move the readout but not
the pixels — the decoder ignores or denoises the edit):

* ``--method`` — ``probe`` uses the linear probe's per-class weight contrast ``w_to - w_from`` (mapped
  to raw latent space; *discriminative*, max-margin, often along low-variance dims). ``diff_means`` uses
  the difference of class **centroids** ``mean(to) - mean(from)`` — the actual *translation* between the
  clouds, which a generative decoder is far more likely to render.
* ``--all_layers`` — steer **every** cached layer at once (each with its own direction) instead of one
  layer; a single late layer is easily averaged away by the decoder's multi-layer fusion.

``alpha`` is expressed as a **fraction of each layer's per-token L2 norm**, so the same sweep is
comparable across layers of very different scale and across single/all-layer modes (``alpha=1.0`` ~=
"shift each token by 100% of its own norm along the direction").

Verification stays non-circular: an **independent** solid-vs-fluid logistic readout (fit on the target
cache, excluding the steered clips) gives ``P(solid)`` along the sweep — a margin read off the steering
direction itself would be monotonic by construction and prove nothing.

Example
-------
    python experiments/pipeline/05_steering/steer_category.py \
        --config configs/train/physics_iq_transformer_large.yaml \
        --target_latent_dir outputs/latents/physics_iq/vjepa2_large \
        --checkpoint outputs/runs/physics_iq_decoder_large/checkpoints/last.pt \
        --method diff_means --all_layers \
        --from_category fluid_dynamics --to_category solid_mechanics \
        --alphas 0,0.25,0.5,0.75,1.0 --output_dir outputs/analysis/steer_category/fluid_to_solid \
        --device cuda
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
from src.analysis.intervention import apply_intervention_multi
from src.data.physics_iq_categories import scenario_for_id
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _distinct_scenario_clips(target: LatentDataset, category: str, n: int) -> list[int]:
    """Indices of up to ``n`` clips of ``category``, one per *distinct scenario*.

    Physics-IQ ships each scenario as ~6 near-duplicate clips (perspectives x takes); naively taking
    the first ``n`` clips returns perspectives of the *same* scene. Deduping by scenario gives ``n``
    genuinely different fluid phenomena (e.g. balloon, juice, pour) instead of one scene six times.
    """
    seen: set[str] = set()
    idx: list[int] = []
    for i in range(len(target)):
        s = target[i]
        if s["category"] != category:
            continue
        sc = scenario_for_id(s["id"])
        if sc in seen:
            continue
        seen.add(sc)
        idx.append(i)
        if len(idx) >= n:
            break
    return idx


def _unit_direction(args, latent_dir: str, layer: int) -> tuple[np.ndarray, dict]:
    """Per-layer unit steering direction via the chosen method (``probe`` or ``diff_means``)."""
    if args.method == "diff_means":
        return steering.category_mean_direction(latent_dir, layer, args.from_category, args.to_category)
    if not args.directions:
        raise SystemExit("--method probe requires --directions (the category_directions.npz).")
    return steering.category_steering_direction(args.directions, layer, args.from_category,
                                                args.to_category)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--target_latent_dir", required=True, help="cache to steer + decode (e.g. physics_iq)")
    p.add_argument("--directions", default=None, help="category_directions.npz (required for --method probe)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", default="probe", choices=["probe", "diff_means"],
                   help="probe: w_to-w_from (discriminative); diff_means: mean(to)-mean(from) (translation)")
    p.add_argument("--all_layers", action="store_true",
                   help="steer every cached layer at once (else only --layer)")
    p.add_argument("--from_category", default="fluid_dynamics", help="category of the clips we steer")
    p.add_argument("--to_category", default="solid_mechanics", help="category we steer toward")
    p.add_argument("--layer", type=int, default=-1, help="intervention layer; -1 = deepest available")
    p.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.3",
                   help="sweep, as a FRACTION of each layer's per-token norm (fine low range: the "
                        "readout saturates and the decode degrades well before 1.0)")
    p.add_argument("--num_samples", type=int, default=4, help="how many from_category clips to steer")
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    device = args.device

    target = LatentDataset(args.target_latent_dir, layers=cfg.encoder.layers)
    available = target.available_layers()
    read_layer = max(available) if args.layer < 0 else args.layer
    steer_layers = sorted(available) if args.all_layers else [read_layer]

    # 1. per-layer UNIT steering directions (probe contrast or class-mean translation).
    unit_dirs: dict[int, torch.Tensor] = {}
    dinfo: dict[int, dict] = {}
    for L in steer_layers:
        dnp, info = _unit_direction(args, args.target_latent_dir, L)
        unit_dirs[L] = torch.from_numpy(dnp).to(device)
        dinfo[L] = info

    # which clips to steer — one per distinct scenario (avoid near-duplicate perspectives/takes).
    steer_idx = _distinct_scenario_clips(target, args.from_category, args.num_samples)
    if not steer_idx:
        raise SystemExit(f"No clips with category '{args.from_category}' in {args.target_latent_dir}.")
    steer_ids = {target[i]["id"] for i in steer_idx}
    print(f"[steer_category] steering {len(steer_idx)} distinct '{args.from_category}' scenarios: "
          f"{[scenario_for_id(target[i]['id']) for i in steer_idx]}")

    # 2. independent readout: P(to_category) fit on the rest of the cache (excludes the steered clips).
    readout = steering.category_readout(args.target_latent_dir, read_layer, args.to_category,
                                        exclude_ids=steer_ids)

    # 3. build decoder + load checkpoint.
    rec0 = target.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(available)
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    # 4. steer, decode, verify per clip. alpha is a fraction of each layer's per-token norm, so we scale
    #    each unit direction by that layer's norm (measured per clip) before applying.
    per_sample_curves: dict[str, list[dict]] = {}
    all_pred = {a: [] for a in alphas}
    for n, i in enumerate(steer_idx):
        batch = latent_collate([target[i]])
        sid = batch["id"][0]
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}

        norms = {L: float(latents[L].norm(dim=-1).mean()) for L in steer_layers}
        scaled = {L: unit_dirs[L] * norms[L] for L in steer_layers}  # edit of "1.0" == one token norm
        if n == 0:
            method = f"{args.method}{'/all_layers' if args.all_layers else ''}"
            print(f"[steer_category] method={method} steer_layers={steer_layers} "
                  f"per-token norms={ {L: round(v, 1) for L, v in norms.items()} } "
                  f"alpha(frac)={min(alphas):g}..{max(alphas):g}")

        curve = []
        frames_by_alpha = {}
        for a in alphas:
            perturbed = apply_intervention_multi(latents, scaled, a)
            pooled = perturbed[read_layer].mean(dim=(0, 1)).cpu().numpy()[None, :]
            pred = float(readout(pooled)[0])
            curve.append({"alpha": a, "readout": pred})
            all_pred[a].append(pred)
            dec = decoder(perturbed, grid)
            if dec.frames is not None:
                frames_by_alpha[a] = dec.frames[0].cpu()
        per_sample_curves[sid] = curve

        if frames_by_alpha:
            viz.steering_filmstrip(frames_by_alpha, out / f"{sid}_filmstrip.png")
            if n == 0:
                base = frames_by_alpha[min(alphas, key=abs)]
                for a in (min(alphas), max(alphas)):
                    if a in frames_by_alpha:
                        viz.panel_video(base, frames_by_alpha[a], out / f"{sid}_alpha{a:g}.mp4",
                                        fps=cfg.data.fps)

    # aggregate controllability curve: does P(to_category) rise with alpha?
    mean_curve = [{"alpha": a, "readout": float(np.mean(v))} for a, v in sorted(all_pred.items())]
    rho = float(spearmanr([r["alpha"] for r in mean_curve], [r["readout"] for r in mean_curve]).statistic)
    viz.steering_sweep_plot(
        {"mean (steered clips)": mean_curve, **{k: v for k, v in list(per_sample_curves.items())[:6]}},
        out / "controllability.png",
        title=f"Steering {args.from_category} -> {args.to_category} ({args.method}"
              f"{', all layers' if args.all_layers else f', L{read_layer}'})",
        ylabel=f"P({args.to_category}) (independent readout)")

    summary = {
        "from_category": args.from_category, "to_category": args.to_category,
        "method": args.method, "all_layers": bool(args.all_layers),
        "steer_layers": steer_layers, "readout_layer": int(read_layer),
        "alphas_fraction_of_norm": alphas, "n_steered": len(steer_idx),
        "direction_info": {str(L): info for L, info in dinfo.items()},
        "readout_monotonicity_spearman": round(rho, 4),
        "mean_readout_curve": [{"alpha": r["alpha"], "readout": round(r["readout"], 5)} for r in mean_curve],
        "target_latent_dir": args.target_latent_dir, "directions": args.directions,
    }
    (out / "category_steering_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / f"directions_{args.from_category}_to_{args.to_category}.npz",
             **{f"layer{L}": unit_dirs[L].cpu().numpy() for L in steer_layers})
    tag = f"{args.method}{'/all' if args.all_layers else f'/L{read_layer}'}"
    print(f"[steer_category] {args.from_category}->{args.to_category} [{tag}]: "
          f"readout monotonicity rho={rho:.3f} over {len(steer_idx)} clips -> {out}")


if __name__ == "__main__":
    main()
