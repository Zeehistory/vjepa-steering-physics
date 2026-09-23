"""Re-derive every tuned number in the 3D rolling-ball dataset, from scratch, on the real renderer.

The scene constants in ``src/data/rolling_ball3d.py`` (camera elevation, lighting, speed range) are not
arbitrary — each was set by a measurement, and each fixes a specific failure mode that silently corrupts
the steering metric. ``tests/test_rolling_ball3d.py`` pins them cheaply; this script is the slower,
human-readable audit that PRINTS the evidence, so the choices can be re-checked or re-tuned rather than
taken on faith.

    MUJOCO_GL=egl python experiments/threads/velocity/01_data/validate_rb3d.py [--scenes 25] [--sweep]

Reports:
  1. tracker vs analytic projection      -> the measurement ceiling (must be ~0; the shadow can break it)
  2. constant-velocity rolling            -> the "one velocity per clip" label must not be a lie
  3. perspective anisotropy               -> real 3D, but not so extreme the commands collapse
  4. image speed band vs the 2D reference -> makes this an apples-to-apples transfer test
  5. shared-start feasibility             -> speeds must not be silently shrunk
``--sweep`` additionally re-runs the world-speed-range search that picked [0.06, 0.115].
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from src.analysis.ball_tracking import ball_centroids, measured_velocity  # noqa: E402
from src.data.rolling_ball3d import BALL_R, TABLE_H, RollingBall3D  # noqa: E402

REF_2D_BAND = (0.012, 0.024)   # image speed/frame of moving_ball_scene_velocity2d(_mixed)


def check_tracker(gen: RollingBall3D, n_scenes: int) -> None:
    """(1) The whole pipeline scores steering by tracking a darkness centroid in DECODED pixels; our
    labels are an analytic camera projection. If they disagree, every angle error inherits the bias."""
    cerr, aerr, oob = [], [], 0
    for i in range(n_scenes * gen.clips_per_scene):
        c = gen.generate(i)
        keys, st = c.state_keys, c.state.numpy()
        gt = np.stack([st[:, keys.index("obj0_pos_x")], st[:, keys.index("obj0_pos_y")]], axis=1)
        tr = ball_centroids(c.frames)
        cerr.append(np.linalg.norm(gt - tr, axis=1).mean() * gen.image_size)
        label = np.array([st[:, keys.index("obj0_vel_x")].mean(), st[:, keys.index("obj0_vel_y")].mean()])
        m = measured_velocity(c.frames)
        meas = np.array([m["vel_x"], m["vel_y"]])
        cos = label @ meas / (np.linalg.norm(label) * np.linalg.norm(meas) + 1e-12)
        aerr.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        if gt.min() < 0.02 or gt.max() > 0.98:
            oob += 1
    print(f"1. TRACKER vs LABEL over {len(cerr)} clips")
    print(f"   centroid err : mean {np.mean(cerr):5.2f} px   max {np.max(cerr):5.2f} px")
    print(f"   velocity ang : mean {np.mean(aerr):5.3f} deg  max {np.max(aerr):5.3f} deg  <- METRIC CEILING")
    print(f"   clips leaving the frame: {oob}")
    verdict = "OK" if np.mean(cerr) < 2.0 and np.mean(aerr) < 2.0 and oob == 0 else "*** FAIL ***"
    print(f"   => {verdict} (a large centroid err usually means the contact shadow darkened past 0.5 grey)")


def check_constant_velocity(gen: RollingBall3D, n: int = 6) -> None:
    """(2) scene_velocity2d's contract is one constant velocity per clip. Rolling without slipping with
    zero rolling friction should preserve that in 3D; a decaying ball would make the label wrong."""
    mujoco, model, data, _r, _cam = gen._lazy_sim()
    drifts = []
    for i in range(n):
        c = gen.generate(i)
        p0 = np.array(c.meta["world_pos0"])
        v = np.array([c.meta["world_vel_x"], c.meta["world_vel_y"]])
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [p0[0], p0[1], TABLE_H + BALL_R]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[:3] = [v[0], v[1], 0.0]
        data.qvel[3:6] = [-v[1] / BALL_R, v[0] / BALL_R, 0.0]
        mujoco.mj_forward(model, data)
        steps = int(round((1.0 / gen.fps) / model.opt.timestep))
        sp = []
        for _ in range(gen.num_frames):
            sp.append(float(np.linalg.norm(data.qvel[:2])))
            for _ in range(steps):
                mujoco.mj_step(model, data)
        sp = np.array(sp)
        drifts.append((sp.max() - sp.min()) / sp.mean())
    print(f"\n2. CONSTANT-VELOCITY ROLLING over {n} clips")
    print(f"   world speed drift (max-min)/mean: max {np.max(drifts):.2e}")
    print(f"   => {'OK' if np.max(drifts) < 0.01 else '*** FAIL ***'} (rolling w/o slipping, no rolling friction)")


def check_perspective(gen: RollingBall3D) -> None:
    """(3) The point of the 3D scene is perspective. Too little => it is 2D with extra steps; too much =>
    the 8 headings collapse onto a squashed ellipse and the velocity command degenerates."""
    heads = [np.linalg.norm(gen._image_velocity(np.zeros(2), 0.08 * np.array([np.cos(a), np.sin(a)])))
             for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    at = [np.linalg.norm(gen._image_velocity(np.array([px, py]), np.array([0.08, 0.0])))
          for px in (-0.3, 0.0, 0.3) for py in (-0.3, 0.0, 0.3)]
    print("\n3. PERSPECTIVE (impossible in the 2D dataset: there image velocity is a global constant)")
    print(f"   same world speed, 24 headings   : img speed {min(heads):.4f}..{max(heads):.4f}"
          f"  ratio {max(heads)/min(heads):.3f}")
    print(f"   same world v, 9 table positions : img speed {min(at):.4f}..{max(at):.4f}"
          f"  ratio {max(at)/min(at):.3f}")
    ok = 1.05 < max(heads) / min(heads) < 1.6 and max(at) / min(at) > 1.05
    print(f"   => {'OK' if ok else '*** FAIL ***'} (real foreshortening, commands still well spread)")


def image_speed_stats(gen: RollingBall3D, n_scenes: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Analytic (no render): every rank's image speed + the per-scene shared-start shrink factor."""
    sp, scales = [], []
    for s in range(n_scenes):
        srng = np.random.default_rng(gen.seed * 100_003 + 7919 * (s + 1))
        K, (lo, hi) = gen.clips_per_scene, gen.speed_range
        base = float(srng.uniform(0, 2 * np.pi))
        angles = np.array([(base + 2 * np.pi * j / K + float(srng.uniform(-0.20, 0.20))) % (2 * np.pi)
                           for j in range(K)])
        speeds = np.sort(srng.uniform(lo, hi, size=K))
        for j in range(1, K):
            gap = 0.4 * (hi - lo) / K
            if speeds[j] - speeds[j - 1] < gap:
                speeds[j] = min(hi, speeds[j - 1] + gap)
        speeds = speeds[srng.permutation(K)]
        vels = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)
        pos0, scale = gen._shared_start(srng, vels)
        scales.append(scale)
        sp += [float(np.linalg.norm(gen._image_velocity(pos0, v))) for v in vels * scale]
    return np.array(sp), np.array(scales)


def check_speed_band(gen: RollingBall3D) -> None:
    """(4)+(5) Image speeds must land in the 2D dataset's band (else it is a different task, not a
    transfer test), and the shared start must not be silently shrinking the scene's speeds."""
    sp, scales = image_speed_stats(gen)
    p2, p50, p98 = (np.percentile(sp, q) for q in (2, 50, 98))
    print(f"\n4. IMAGE SPEED BAND (world speed_range {gen.speed_range} m/s)")
    print(f"   img speed/frame: p2={p2:.4f} p50={p50:.4f} p98={p98:.4f}"
          f"   | 2D reference band {REF_2D_BAND}")
    print(f"   net displacement over 15 frames: p2={p2*15:.2f} p98={p98*15:.2f} image widths")
    print(f"   => {'OK' if 0.008 < p2 and p98 < 0.028 else '*** FAIL ***'}")
    print(f"\n5. SHARED-START FEASIBILITY over {len(scales)} scenes")
    print(f"   scenes whose speeds were shrunk: {(scales < 0.999).mean()*100:.1f}%  (mean scale {scales.mean():.3f})")
    print(f"   => {'OK' if (scales < 0.999).mean() < 0.02 else '*** shrinking: speeds are being rescaled ***'}")


def sweep(gen: RollingBall3D) -> None:
    print("\n6. WORLD SPEED-RANGE SWEEP (how [0.06, 0.115] was chosen)")
    print("   want: image band ~= the 2D reference, with ~0% shrink")
    for sr in ([0.05, 0.10], [0.06, 0.115], [0.07, 0.13], [0.08, 0.15]):
        g = RollingBall3D(image_size=gen.image_size, num_frames=gen.num_frames, fps=gen.fps,
                          speed_range=tuple(sr), clips_per_scene=gen.clips_per_scene, seed=gen.seed)
        sp, scales = image_speed_stats(g)
        print(f"   world {str(sr):14s} -> img p2={np.percentile(sp,2):.4f} p50={np.percentile(sp,50):.4f} "
              f"p98={np.percentile(sp,98):.4f} | shrink {(scales<0.999).mean()*100:5.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=int, default=25, help="scenes to render for the tracker audit")
    p.add_argument("--sweep", action="store_true", help="also re-run the speed-range search")
    args = p.parse_args()

    gen = RollingBall3D(image_size=256, num_frames=16, fps=4, clips_per_scene=8, seed=0)
    print(f"rolling_ball3d: {gen.clips_per_scene} clips/scene, {gen.num_frames} frames @ {gen.fps} fps, "
          f"{gen.image_size}px, world speed_range {gen.speed_range} m/s\n")
    check_tracker(gen, args.scenes)
    check_constant_velocity(gen)
    check_perspective(gen)
    check_speed_band(gen)
    if args.sweep:
        sweep(gen)


if __name__ == "__main__":
    main()
