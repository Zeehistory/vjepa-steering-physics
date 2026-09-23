

# --- repo-root shim: make ``src`` importable however this script is invoked ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = next(p for p in _Path(__file__).resolve().parents
                  if (p / "pyproject.toml").is_file())
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -----------------------------------------------------------------------------

#!/usr/bin/env python
"""Local sanity check for the scene_velocity moving-ball dataset (no encoder, CPU only)."""
import numpy as np
import torch

from src.data.moving_ball import MovingBall

K = 4
gen = MovingBall(image_size=256, num_frames=16, fps=4, scenario="scene_velocity",
                 speed_range=(0.012, 0.045), radius_range=(0.085, 0.085),
                 clips_per_scene=K, seed=0)

ok = True
for scene in range(3):
    clips = [gen.generate(scene * K + r) for r in range(K)]
    first_frames = [c.frames[0] for c in clips]
    speeds = [c.meta["speed"] for c in clips]
    # 1) identical first frame across the scene
    max_dev = max(float((first_frames[0] - f).abs().max()) for f in first_frames[1:])
    # 2) speeds strictly increasing with rank
    incr = all(speeds[i] < speeds[i + 1] for i in range(K - 1))
    # 3) ball stays in frame: dark mass > 0 every frame, for the fastest clip
    fast = clips[-1].frames
    gray = fast.mean(1)               # (T,H,W)
    dark_mass = (1.0 - gray).clamp(min=0).flatten(1).sum(1)  # (T,)
    in_frame = bool((dark_mass > 1.0).all())
    # 4) same direction within scene
    angles = [c.meta["angle"] for c in clips]
    same_dir = max(abs(a - angles[0]) for a in angles) < 1e-9
    print(f"scene {scene}: first_frame_max_dev={max_dev:.2e}  speeds={[round(s,4) for s in speeds]}  "
          f"incr={incr}  same_dir={same_dir}  ball_in_frame_all_T={in_frame}  "
          f"grid_first_frame_white_frac={float((gray[0]>0.9).float().mean()):.3f}")
    ok = ok and (max_dev < 1e-6) and incr and same_dir and in_frame

# scene disjointness across seeds (train vs test must differ)
g_test = MovingBall(image_size=256, num_frames=16, fps=4, scenario="scene_velocity",
                    speed_range=(0.012, 0.045), radius_range=(0.085, 0.085),
                    clips_per_scene=K, seed=2)
a = gen.generate(0).frames
b = g_test.generate(0).frames
disjoint = float((a - b).abs().max()) > 1e-3
print(f"train(seed0) vs test(seed2) scene0 differ: {disjoint}")
print("ALL CHECKS PASSED" if (ok and disjoint) else "*** CHECK FAILED ***")
