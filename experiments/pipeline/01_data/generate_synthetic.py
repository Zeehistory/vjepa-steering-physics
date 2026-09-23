

#!/usr/bin/env python
"""Step 2: generate / preview the synthetic-physics datasets (exact ground-truth labels).

The synthetic generator is deterministic and rendered on the fly, so extraction reads it directly — you
do *not* need to materialise clips to disk to train probes. This script is for **sanity-checking** the
simulators and for eyeballing the ground-truth state: it renders a few preview mp4s per scenario, dumps
a per-clip label manifest, and plots a trajectory overlay.

Example
-------
    python experiments/pipeline/01_data/generate_synthetic.py --config configs/data/synthetic_solid.yaml \
        --output_dir outputs/synthetic/solid --num_preview 4
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

from omegaconf import OmegaConf

from src.analysis import visualization as viz
from src.data import build_dataset
from src.utils.config import load_config
from src.utils.video_io import save_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="a data config, e.g. configs/data/synthetic_solid.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_preview", type=int, default=4, help="preview clips per distinct scenario")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    # Data config files are flat (name/scenarios/...); merge into the .data section, then apply CLI
    # overrides last so e.g. `data.num_clips=6` wins over the file's value.
    cfg = load_config(None)
    cfg.data = OmegaConf.merge(cfg.data, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(cfg.data, encoder_image_size=cfg.data.image_size, encoder_frames=cfg.data.num_frames)
    manifest = []
    seen_per_scenario: dict[str, int] = {}
    for i in range(len(ds)):
        s = ds[i]
        cat = s["category"]
        manifest.append({
            "id": s["id"], "scenario": cat,
            "state_keys": s["state_keys"],
            "state_first": s["state"][0].tolist(),
            "state_last": s["state"][-1].tolist(),
            "meta": {k: (v if isinstance(v, (int, float, str)) else str(v))
                     for k, v in s["meta"].items()},
        })
        if seen_per_scenario.get(cat, 0) < args.num_preview:
            seen_per_scenario[cat] = seen_per_scenario.get(cat, 0) + 1
            save_video(s["frames"], out / f"{s['id']}.mp4", fps=cfg.data.fps)
            if float(s["state_mask"].sum()) > 0:
                viz.trajectory_overlay(s["frames"], s["state"], s["state_keys"],
                                       out / f"{s['id']}_trajectory.png")

    (out / "labels.json").write_text(json.dumps(manifest, indent=2))
    print(f"[generate_synthetic] {len(ds)} clips ({sorted(seen_per_scenario)}); previews + labels -> {out}")


if __name__ == "__main__":
    main()
