

#!/usr/bin/env python
"""Evaluate a trained decoder: reconstruction + physics metrics, with baselines and controls.

Writes ``<output_dir>/metrics.json``. The JSON separates the trained model from baselines
(copy-first-frame / mean-frame / random-frame) and the oracle-state upper bound, and tags physics
metrics with whether ground-truth state was available — so claims stay honest.

Example
-------
    python experiments/pipeline/07_eval/eval_decoder.py --config configs/train/smoke_synthetic.yaml \
        --latent_dir outputs/smoke/latents --checkpoint outputs/smoke/runs/decoder/checkpoints/last.pt \
        --output_dir outputs/smoke/eval
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
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.eval.baselines import frame_baselines, oracle_state
from src.eval.physics_metrics import physics_metrics
from src.eval.reconstruction_metrics import reconstruction_metrics
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


_WEIGHT_KEY = "_weight"


def _weighted(metrics: dict[str, Any], weight: int) -> dict[str, Any]:
    out = dict(metrics)
    out[_WEIGHT_KEY] = int(weight)
    return out


def _avg(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not dicts:
        return {}
    keys = {k for d in dicts for k in d if k != _WEIGHT_KEY}
    out: dict[str, Any] = {}
    for k in keys:
        weighted_vals = [
            (float(d[k]), float(d.get(_WEIGHT_KEY, 1.0)))
            for d in dicts
            if d.get(k) is not None
        ]
        denom = sum(w for _, w in weighted_vals)
        out[k] = float(sum(v * w for v, w in weighted_vals) / denom) if denom else None
    return out


def _new_recon_bucket() -> dict[str, Any]:
    return {
        "model": [],
        "baselines": {k: [] for k in ("copy_first_frame", "mean_frame", "random_frame")},
    }


def _summarize_recon_bucket(bucket: dict[str, Any], count: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": _avg(bucket["model"]) if bucket["model"] else None,
        "baselines": {
            k: _avg(v)
            for k, v in bucket["baselines"].items()
            if v and any(item for item in v)
        },
    }
    if count is not None:
        out = {"num_clips": int(count), **out}
    return out


def _summarize_physics_bucket(
    model: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    count: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": _avg(model) if model else None,
        "oracle_upper_bound": _avg(oracle) if oracle else None,
        "state_available": bool(model),
    }
    if count is not None:
        out = {"num_clips": int(count), **out}
    return out


@torch.no_grad()
def evaluate_decoder(cfg, latent_dir: str, checkpoint: str, output_dir: str | Path,
                     device: str = "cpu") -> dict[str, Any]:
    dataset = LatentDataset(latent_dir, layers=cfg.encoder.layers)
    rec0 = dataset.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames

    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(dataset.available_layers())
    load_checkpoint(checkpoint, decoder, map_location=device)

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=latent_collate)
    recon = _new_recon_bucket()
    recon_by_category = defaultdict(_new_recon_bucket)
    category_counts: Counter[str] = Counter()
    phys_model, phys_oracle = [], []
    phys_by_category = defaultdict(lambda: {"model": [], "oracle": []})
    state_keys = dataset[0]["state_keys"]

    for batch in loader:
        grid = tuple(int(x) for x in batch["grid"])
        latents = {int(k): v.to(device) for k, v in batch["layers"].items()}
        out = decoder(latents, grid)
        target = batch["frames"].to(device)
        categories = [str(c) for c in batch["category"]]
        category_counts.update(categories)
        batch_size = len(categories)

        if out.frames is not None:
            recon["model"].append(_weighted(reconstruction_metrics(out.frames, target), batch_size))
            baselines = frame_baselines(target)
            for name, base in baselines.items():
                recon["baselines"][name].append(_weighted(reconstruction_metrics(base, target), batch_size))

            for category in sorted(set(categories)):
                indices = [i for i, c in enumerate(categories) if c == category]
                bucket = recon_by_category[category]
                bucket["model"].append(
                    _weighted(reconstruction_metrics(out.frames[indices], target[indices]), len(indices))
                )
                for name, base in baselines.items():
                    bucket["baselines"][name].append(
                        _weighted(reconstruction_metrics(base[indices], target[indices]), len(indices))
                    )

        if out.state is not None and batch["state_mask"].sum() > 0:
            tgt_state, mask = batch["state"].to(device), batch["state_mask"][0]
            phys_model.append(_weighted(physics_metrics(out.state, tgt_state, state_keys, mask), batch_size))
            phys_oracle.append(
                _weighted(physics_metrics(oracle_state(tgt_state), tgt_state, state_keys, mask), batch_size)
            )

            for category in sorted(set(categories)):
                indices = [i for i, c in enumerate(categories) if c == category]
                category_mask = batch["state_mask"][indices[0]]
                if category_mask.sum() <= 0:
                    continue
                bucket = phys_by_category[category]
                pred_state = out.state[indices]
                target_state = tgt_state[indices]
                bucket["model"].append(
                    _weighted(physics_metrics(pred_state, target_state, state_keys, category_mask), len(indices))
                )
                bucket["oracle"].append(
                    _weighted(
                        physics_metrics(oracle_state(target_state), target_state, state_keys, category_mask),
                        len(indices),
                    )
                )

    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "num_clips": len(dataset),
        "category_counts": dict(sorted(category_counts.items())),
        "reconstruction": {
            **_summarize_recon_bucket(recon),
            "by_category": {
                category: _summarize_recon_bucket(bucket, category_counts[category])
                for category, bucket in sorted(recon_by_category.items())
            },
        },
        "physics": {
            **_summarize_physics_bucket(phys_model, phys_oracle),
            "by_category": {
                category: _summarize_physics_bucket(
                    bucket["model"], bucket["oracle"], category_counts[category]
                )
                for category, bucket in sorted(phys_by_category.items())
            },
        },
        "claim_taxonomy_note": (
            "reconstruction.model beating baselines => pixels are decodable; physics.model near "
            "oracle => state is decodable. These are distinct claims; see docs/method.md."
        ),
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    res = evaluate_decoder(cfg, args.latent_dir, args.checkpoint, args.output_dir, args.device)
    print(f"[eval_decoder] metrics -> {args.output_dir}/metrics.json")
    print(json.dumps(res["reconstruction"]["model"], indent=2))


if __name__ == "__main__":
    main()
