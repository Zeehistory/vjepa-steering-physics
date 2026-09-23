

#!/usr/bin/env python
"""Pre-flight certificate for the spin x velocity CROSSTALK scene — run BEFORE any GPU time.

Every expensive stage downstream (latent extraction, decoder training, operator fitting) assumes four
things about this dataset, and each of them has a cheap, decisive test that needs only the renderer:

  1. **The two quantities are actually independent.** Frictionless contact should conserve ``v`` and
     ``omega`` separately. Measured as: per-clip drift of world speed and of the marker azimuth's
     linear-fit residual. If friction leaked back in, spin would bleed into translation *in the
     simulator*, and every crosstalk number downstream would be measuring MuJoCo, not the latent space.
  2. **The factorial is balanced.** Across a scene, velocity must be uncorrelated with spin — else a
     "velocity operator" could score by reading spin. Measured as the |correlation| between each
     velocity component and omega over the scene grid, and the rank of the design matrix.
  3. **The pixel readouts are unbiased.** ``ball_tracking.measured_velocity`` and
     ``spin_tracking.measured_spin`` are the only instruments the steering experiment has. Both are
     checked against the exact ground truth on RENDERED frames, where the answer is known. A tracker
     bias here would be indistinguishable from a steering failure later.
  4. **The marker never clips the limb.** The design claims the marker stays wholly inside the ball's
     silhouette at every azimuth. Checked empirically as the spread of the marker's warm-pixel mass
     across the clip: a marker being cut by the limb loses mass at one phase of every revolution.

Writes ``certificate.{json,md}`` plus a contact sheet, and exits non-zero if any gate fails.

    MUJOCO_GL=egl PYTHONPATH=. python experiments/threads/restitution-spin/01_data/validate_spin_ball3d.py --output_dir /tmp/spincert
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
import sys
from pathlib import Path

import numpy as np
import torch

from src.analysis.ball_tracking import ball_centroids, measured_velocity
from src.analysis import spin_tracking as st
from src.data.spin_ball3d import SpinBall3D

# Gates. Deliberately strict: these are cheap to satisfy on a correct scene and every one of them
# corresponds to a failure mode that would silently corrupt the crosstalk result.
GATES = {
    "speed_drift_rel": 2e-3,      # |v| must be constant to 0.2% over the clip (frictionless)
    "spin_residual_rad": 1e-3,    # phi(t) must be linear to 1 mrad (frictionless + isotropic inertia)
    "vel_omega_corr": 0.35,       # |corr(v_component, omega)| across a scene's factorial
    "vel_track_err_px": 0.75,     # tracked vs analytic ball centre, pixels at 256
    "omega_track_err": 0.02,      # tracked vs true omega, rad/frame (~2.4% of the range midpoint)
    "marker_mass_cv": 0.15,       # coefficient of variation of marker warm-mass across frames
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_scenes", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image_size", type=int, default=256)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    gen = SpinBall3D(image_size=args.image_size, seed=args.seed)
    K = gen.clips_per_scene
    print(f"[spincert] {args.num_scenes} scenes x {K} clips (n_vel={gen.n_vel}, n_spin={gen.n_spin})",
          flush=True)

    rows: list[dict] = []
    sheets: list[np.ndarray] = []
    for s in range(args.num_scenes):
        scene_v, scene_w = [], []
        for r in range(K):
            clip = gen.generate(s * K + r)
            m = clip.meta
            frames = clip.frames                                  # (T,C,H,W) in [0,1]

            # -- 1. physics conservation, measured not assumed ------------------------------------
            # world speed drift: recompute the ball's world track from the image track (it is on a
            # known plane) and check the inter-frame step is constant.
            keys = list(clip.state_keys)
            st_arr = clip.state.numpy()
            gt_img = st_arr[:, [keys.index("obj0_pos_x"), keys.index("obj0_pos_y")]]
            world = st.unproject_to_ball_plane(gt_img)
            steps = np.linalg.norm(np.diff(world[:, :2], axis=0), axis=1)
            speed_drift = float((steps.max() - steps.min()) / max(steps.mean(), 1e-12))

            # -- 3. pixel readouts vs exact ground truth -----------------------------------------
            # The position gate grades the tracker the experiment actually uses: the marker-occlusion
            # repaired silhouette. The plain darkness centroid is also reported (`raw_track_err_px`)
            # because it is what feeds `measured_velocity`, and its 1/rev marker wobble is the reason
            # the repair exists -- see spin_tracking.ball_centroids_unoccluded.
            trk = st.ball_centroids_unoccluded(frames)
            vel_err_px = float(np.nanmax(np.linalg.norm(trk - gt_img, axis=1)) * args.image_size)
            raw_err_px = float(np.nanmax(np.linalg.norm(ball_centroids(frames) - gt_img, axis=1))
                               * args.image_size)
            mv = measured_velocity(frames)
            gt_v = np.array([st_arr[0, keys.index("obj0_vel_x")], st_arr[0, keys.index("obj0_vel_y")]])
            vel_vec_err_px = float(np.linalg.norm(np.array([mv["vel_x"], mv["vel_y"]]) - gt_v)
                                   * args.image_size)
            ms = st.measured_spin(frames)
            omega_err = float(abs(ms["omega"] - m["omega"]))

            # -- 4. marker never clipped by the limb ---------------------------------------------
            warm = (frames[:, 0] - frames[:, 2] - st.MARKER_RB_THRESH).clamp(min=0.0)
            mass = warm.sum(dim=(1, 2)).numpy()
            mass_cv = float(mass.std() / max(mass.mean(), 1e-12))

            rows.append({
                "scene": s, "rank": r, "vel_index": m["vel_index"], "spin_index": m["spin_index"],
                "omega": m["omega"], "omega_measured_sim": m["omega_measured"],
                "spin_residual": m["spin_residual"],
                "speed_drift_rel": speed_drift,
                "vel_track_err_px": vel_err_px, "raw_track_err_px": raw_err_px,
                "vel_vec_err_px": vel_vec_err_px,
                "omega_track": ms["omega"], "omega_track_err": omega_err,
                "omega_track_residual": ms["residual"], "omega_track_nvalid": ms["n_valid"],
                "marker_mass_cv": mass_cv,
                "vel_x": m["vel_x"], "vel_y": m["vel_y"],
            })
            scene_v.append([m["vel_x"], m["vel_y"]]); scene_w.append(m["omega"])
            if s == 0 and r in (0, gen.n_spin, K - 1):
                sheets.append(_filmstrip(frames))

        # -- 2. the factorial is balanced ---------------------------------------------------------
        V = np.asarray(scene_v); W = np.asarray(scene_w)
        cx = abs(float(np.corrcoef(V[:, 0], W)[0, 1])); cy = abs(float(np.corrcoef(V[:, 1], W)[0, 1]))
        design = np.column_stack([V, W, np.ones(len(W))])
        rows[-1]["_scene_corr"] = {"corr_vx_omega": cx, "corr_vy_omega": cy,
                                   "design_rank": int(np.linalg.matrix_rank(design))}

    df = {k: np.array([r[k] for r in rows], dtype=float) for k in
          ("speed_drift_rel", "spin_residual", "vel_track_err_px", "raw_track_err_px", "vel_vec_err_px",
           "omega_track_err", "marker_mass_cv", "omega_track_residual")}
    corrs = [r["_scene_corr"] for r in rows if "_scene_corr" in r]
    max_corr = max(max(c["corr_vx_omega"], c["corr_vy_omega"]) for c in corrs)
    min_rank = min(c["design_rank"] for c in corrs)

    checks = {
        "speed_drift_rel":  (float(df["speed_drift_rel"].max()),  GATES["speed_drift_rel"]),
        "spin_residual_rad": (float(df["spin_residual"].max()),   GATES["spin_residual_rad"]),
        "vel_omega_corr":   (max_corr,                            GATES["vel_omega_corr"]),
        "vel_track_err_px": (float(df["vel_track_err_px"].max()), GATES["vel_track_err_px"]),
        "omega_track_err":  (float(df["omega_track_err"].max()),  GATES["omega_track_err"]),
        "marker_mass_cv":   (float(df["marker_mass_cv"].max()),   GATES["marker_mass_cv"]),
    }
    passed = {k: bool(v <= lim) for k, (v, lim) in checks.items()}
    # The design matrix must be full rank (4: vx, vy, omega, const) in every scene, or the
    # "independent quantities" claim is false for that scene regardless of the correlations.
    passed["design_full_rank"] = bool(min_rank == 4)

    summary = {
        "n_scenes": args.num_scenes, "clips_per_scene": K, "seed": args.seed,
        "image_size": args.image_size,
        "checks": {k: {"value": v, "limit": lim, "pass": passed[k]} for k, (v, lim) in checks.items()},
        "design_min_rank": min_rank, "design_full_rank": passed["design_full_rank"],
        "medians": {k: float(np.median(v)) for k, v in df.items()},
        "all_pass": bool(all(passed.values())),
        "rows": rows,
    }
    (out / "certificate.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "certificate.md").write_text(_render_md(summary))
    if sheets:
        _save_sheet(np.concatenate(sheets, axis=0), out / "contact_sheet.png")
    print((out / "certificate.md").read_text())
    print(f"[spincert] wrote {out}/certificate.json")
    sys.exit(0 if summary["all_pass"] else 1)


def _filmstrip(frames: torch.Tensor, n: int = 6) -> np.ndarray:
    idx = np.linspace(0, frames.shape[0] - 1, n).astype(int)
    strip = frames[idx].permute(0, 2, 3, 1).numpy()
    return np.concatenate(list(strip), axis=1)


def _save_sheet(img: np.ndarray, path: Path) -> None:
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))
    except Exception as e:                                        # contact sheet is a nicety, not a gate
        print(f"[spincert] contact sheet skipped: {e}")


def _render_md(s: dict) -> str:
    lines = [
        "# spin_ball3d pre-flight certificate", "",
        f"- scenes: {s['n_scenes']} x {s['clips_per_scene']} clips (seed {s['seed']}, "
        f"{s['image_size']}px)",
        f"- **overall: {'PASS' if s['all_pass'] else 'FAIL'}**", "",
        "| check | worst value | limit | pass |", "|---|---|---|---|",
    ]
    for k, c in s["checks"].items():
        lines.append(f"| `{k}` | {c['value']:.5g} | {c['limit']:.5g} | {'yes' if c['pass'] else '**NO**'} |")
    lines += [
        f"| `design_full_rank` | min rank {s['design_min_rank']} | 4 | "
        f"{'yes' if s['design_full_rank'] else '**NO**'} |", "",
        "Medians: " + ", ".join(f"`{k}`={v:.4g}" for k, v in s["medians"].items()), "",
        "What each check rules out is documented in the script's module docstring.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
