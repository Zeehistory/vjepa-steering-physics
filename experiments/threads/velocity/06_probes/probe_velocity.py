

#!/usr/bin/env python
"""Step 2 (velocity-first): temporal velocity probe on the clean moving-ball latent cache.

Probes **velocity** (signed vector), **speed**, and **direction** from V-JEPA2 latents, comparing three
representations per layer so we can answer the supervisor's question directly:

* ``clip_pool``    — the old 1x1024 (pool away time),
* ``temporal``     — the full 8x1024 temporal sequence (keep time),
* ``temporal_diff``— consecutive-frame differences (motion = displacement between temporal tokens).

Writes ``velocity_decodability.csv`` (per layer x representation x target x {linear,mlp}, with
shuffled-latent + randomized-label controls) and an R²-by-layer plot per target. Run on a SYNTHETIC
latent cache that has exact ground-truth state (e.g. moving_ball_velocity).

Example
-------
    python experiments/threads/velocity/06_probes/probe_velocity.py \
        --latent_dir outputs/latents/moving_ball_velocity/vjepa2_large \
        --output_dir outputs/analysis/moving_ball_velocity/velocity_probe
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


from src.analysis import visualization as viz
from src.training import probe_velocity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = "all" if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    records = probe_velocity(args.latent_dir, layers=layers, seed=args.seed,
                             output_csv=out / "velocity_decodability.csv")
    for tgt in ("vel", "speed", "angle"):
        if any(r["target"] == tgt for r in records):
            viz.velocity_probe_plot(records, out / f"velocity_probe_{tgt}.png", target=tgt)

    # console headline: best layer/representation for the signed velocity vector
    vel = [r for r in records if r["target"] == "vel" and r["probe"] == "linear"]
    if vel:
        best = max(vel, key=lambda r: r["r2"])
        print(f"[probe_velocity] best linear vel R²={best['r2']} at layer {best['layer']} "
              f"using '{best['representation']}' (ctrl shuf-latent R²={best['ctrl_shuffled_latent_r2']})")
    print(f"[probe_velocity] {len(records)} probe results -> {out}")


if __name__ == "__main__":
    main()
