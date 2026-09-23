

#!/usr/bin/env python
"""Step 2 (velocity-first): generate a handful of clean moving-ball DEMO clips for eyeballing.

Per the supervisor's guideline: *generate demo videos first and inspect them qualitatively before
going to full-scale*. This script needs **no encoder and no GPU** — it only runs the deterministic
:class:`MovingBall` generator and dumps, for each scenario, a few mp4s + a frame-strip PNG + a JSON of
the exact ground-truth state (so you can verify the rendered motion matches the labelled velocity).

Run locally before extracting latents on the cluster:

    python experiments/pipeline/01_data/generate_ball_demos.py --output_dir outputs/demos/moving_ball --n 4

What to check by eye:
* the ball stays fully inside the frame for the whole clip (no clipping at edges),
* constant-velocity clips move in a straight line at a steady rate,
* occlusion clips: the ball disappears behind the grey wall and reappears on the other side,
* rotated clips: same apparent speed, different directions.
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


from src.analysis import visualization as viz
from src.data.moving_ball import MovingBall
from src.utils.video_io import save_frames_grid, save_video

SCENARIOS = ["constant_velocity", "occlusion", "rotated"]


def _state_summary(clip) -> dict:
    keys = clip.state_keys
    st = clip.state
    def col(name):
        return st[:, keys.index(name)].tolist()
    return {
        "scenario": clip.meta["scenario"],
        "fps": clip.meta["fps"],
        "radius": round(clip.meta["radius"], 4),
        "speed_per_frame": round(clip.meta["speed"], 5),
        "angle_rad": round(clip.meta["angle"], 4),
        "vel_xy": [round(clip.meta["vel_x"], 5), round(clip.meta["vel_y"], 5)],
        "n_hidden_frames": clip.meta.get("n_hidden_frames", 0),
        "pos_x": [round(v, 4) for v in col("obj0_pos_x")],
        "pos_y": [round(v, 4) for v in col("obj0_pos_y")],
        "visible": [int(v) for v in col("obj0_visible")],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/demos/moving_ball")
    p.add_argument("--n", type=int, default=4, help="clips per scenario")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--camera_rotation", action="store_true",
                   help="for the 'rotated' scenario, also roll the camera per clip")
    p.add_argument("--scenarios", default=",".join(SCENARIOS),
                   help="comma-separated subset of: " + ",".join(SCENARIOS))
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in args.scenarios.split(",") if s]

    manifest: dict[str, list[dict]] = {}
    for scenario in scenarios:
        gen = MovingBall(
            image_size=args.image_size, num_frames=args.num_frames, fps=args.fps,
            scenario=scenario, camera_rotation=args.camera_rotation, seed=0,
        )
        sdir = out / scenario
        sdir.mkdir(parents=True, exist_ok=True)
        manifest[scenario] = []
        for i in range(args.n):
            clip = gen.generate(i)
            stem = f"{scenario}_{i:03d}"
            save_video(clip.frames, sdir / f"{stem}.mp4", fps=args.fps)
            save_frames_grid(clip.frames, sdir / f"{stem}_strip.png", ncols=8)
            # trajectory + velocity-arrow overlay on the last frame (verify motion matches the label)
            viz.trajectory_overlay(clip.frames, clip.state, clip.state_keys,
                                   sdir / f"{stem}_traj.png")
            summ = _state_summary(clip)
            manifest[scenario].append(summ)
            (sdir / f"{stem}_state.json").write_text(json.dumps(summ, indent=2))
        print(f"[demos] {scenario}: wrote {args.n} clips -> {sdir}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[demos] manifest + {sum(len(v) for v in manifest.values())} clips -> {out}")


if __name__ == "__main__":
    main()
