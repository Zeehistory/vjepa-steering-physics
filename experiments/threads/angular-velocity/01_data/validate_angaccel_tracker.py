

#!/usr/bin/env python
"""LINCHPIN: validate ``measured_angaccel`` against ground-truth rendered frames.

Every angular-acceleration number downstream (the decoder rendering gate, the zero-shot steer) is read off
decoded pixels by ``measured_angaccel``. If the tracker cannot recover ``alpha`` from GT frames -- where the
answer is exact by construction -- then no downstream number means anything. This is the angular-acceleration
analog of the checks that validated the linear-accel parabola tracker (0.05 deg) and the angvel line tracker
(<0.2%) before they were trusted.

Checks, on the BIG-object rotor (radius 0.32-0.42, the configuration whose decoder renders rotation
faithfully):
  1. per-clip relative error of recovered alpha vs GT obj0_alpha, and frame validity (want all 16 readable);
  2. that the recovered INITIAL rate omega0 matches GT (guards a parabola fit that trades curvature against
     slope);
  3. the SIGN convention (a reversed rotor convention would silently invert every steering result);
  4. the scene contract: frame 0 bit-identical across ranks, so H_b - H_a isolates Delta alpha;
  5. the unwrap safety margin: max per-frame rotation must stay well below pi.
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

import argparse, json
from pathlib import Path
import numpy as np
import torch

from src.analysis.ball_tracking import measured_angaccel, measured_angvel, rotor_orientation
from src.data.moving_ball import MovingBall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=24)
    ap.add_argument("--clips_per_scene", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--dark", type=float, default=0.25)
    ap.add_argument("--red", type=float, default=0.08)
    ap.add_argument("--scenario", default="scene_angaccel2d")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gen = MovingBall(image_size=args.image_size, num_frames=args.num_frames, fps=8,
                     scenario=args.scenario, radius_range=(0.32, 0.42),
                     clips_per_scene=args.clips_per_scene, seed=11)
    K = args.clips_per_scene
    rows, first_frame_ok, max_step = [], [], 0.0
    for s in range(args.n_scenes):
        f0_ref = None
        for r in range(K):
            clip = gen.generate(s * K + r)
            keys = list(clip.state_keys)
            st = clip.state.numpy()
            gt_alpha = float(st[:, keys.index("obj0_alpha")].mean())
            gt_om0 = float(st[0, keys.index("obj0_omega")])
            theta = st[:, keys.index("obj0_theta")]
            max_step = max(max_step, float(np.abs(np.diff(theta)).max()))

            m = measured_angaccel(clip.frames, darkness_thresh=args.dark, red_thresh=args.red)
            rows.append(dict(scene=s, rank=r, gt_alpha=gt_alpha, gt_omega0=gt_om0,
                             alpha=m["alpha"], omega0=m["omega0"], resid=m["resid"],
                             n_valid=m["n_valid"]))
            # scene contract: frame 0 identical across ranks (theta(0)=theta0, omega(0)=omega0 for all)
            if f0_ref is None:
                f0_ref = clip.frames[0].clone()
            else:
                first_frame_ok.append(bool(torch.equal(f0_ref, clip.frames[0])))

    a_gt = np.array([r["gt_alpha"] for r in rows])
    a_hat = np.array([r["alpha"] for r in rows])
    w_gt = np.array([r["gt_omega0"] for r in rows])
    w_hat = np.array([r["omega0"] for r in rows])
    nv = np.array([r["n_valid"] for r in rows])
    finite = np.isfinite(a_hat)
    rel = np.abs(a_hat[finite] - a_gt[finite]) / (np.abs(a_gt[finite]) + 1e-12)

    res = dict(
        scenario=args.scenario, n_clips=len(rows), thresholds=dict(dark=args.dark, red=args.red),
        n_valid_min=int(nv.min()), n_valid_mean=float(nv.mean()), n_finite=int(finite.sum()),
        alpha_rel_err_mean=float(rel.mean()), alpha_rel_err_max=float(rel.max()),
        alpha_rho=float(np.corrcoef(a_gt[finite], a_hat[finite])[0, 1]),
        alpha_slope=float(np.polyfit(a_gt[finite], a_hat[finite], 1)[0]),
        alpha_sign_acc=float(np.mean(np.sign(a_hat[finite]) == np.sign(a_gt[finite]))),
        omega0_rho=float(np.corrcoef(w_gt[finite], w_hat[finite])[0, 1]),
        omega0_slope=float(np.polyfit(w_gt[finite], w_hat[finite], 1)[0]),
        max_per_frame_rotation_rad=max_step, unwrap_margin_vs_pi=float(np.pi / max(max_step, 1e-9)),
        gt_alpha_range=[float(a_gt.min()), float(a_gt.max())],
        first_frame_identical_across_ranks=bool(all(first_frame_ok)) if first_frame_ok else None,
        resid_mean=float(np.mean([r["resid"] for r in rows if np.isfinite(r["resid"])])),
    )
    print("\n============== measured_angaccel vs GROUND-TRUTH frames ==============")
    print(f"  clips={res['n_clips']}  finite={res['n_finite']}  n_valid min/mean={res['n_valid_min']}/{res['n_valid_mean']:.1f} of {args.num_frames}")
    print(f"  alpha  rel_err mean={res['alpha_rel_err_mean']*100:.3f}%  max={res['alpha_rel_err_max']*100:.3f}%")
    print(f"  alpha  rho={res['alpha_rho']:+.6f}  slope={res['alpha_slope']:+.4f} (want +1)  sign_acc={res['alpha_sign_acc']:.3f}")
    print(f"  omega0 rho={res['omega0_rho']:+.6f}  slope={res['omega0_slope']:+.4f} (want +1)")
    print(f"  GT alpha range: {res['gt_alpha_range']}   fit resid={res['resid_mean']:.2e} rad")
    print(f"  max per-frame rotation={max_step:.3f} rad  -> unwrap margin {res['unwrap_margin_vs_pi']:.1f}x below pi")
    print(f"  scene contract (frame 0 identical across ranks): {res['first_frame_identical_across_ranks']}")
    ok = (res["alpha_rel_err_max"] < 0.01 and res["alpha_slope"] > 0.99 and res["alpha_slope"] < 1.01
          and res["n_valid_min"] == args.num_frames and res["first_frame_identical_across_ranks"] is not False
          and res["unwrap_margin_vs_pi"] > 4)
    res["PASS"] = bool(ok)
    print(f"\n  {'PASS' if ok else 'FAIL'}: tracker {'recovers' if ok else 'does NOT recover'} GT angular acceleration.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
