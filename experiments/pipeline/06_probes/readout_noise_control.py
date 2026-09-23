

#!/usr/bin/env python
"""CONTROL: is the alpha ceiling gap a DECODER failure, or is alpha simply harder to READ from imperfect pixels?

The constant-omega decoder renders angular velocity at ceiling rho=0.977 but angular acceleration at only
0.809 (decoding the TRUE H_b in both cases). The tempting reading is "the decoder renders a ramping spin
worse than a constant one". But there is a confound: alpha is a CURVATURE -- a second derivative of the
marker phase -- while omega is a slope. Second-order fits amplify pixel noise far more than first-order
fits, so an imperfect renderer costs alpha more than omega even if it renders both equally well.

This separates the two, with no decoder and no latents. Take GROUND-TRUTH frames (where the tracker is
exact: alpha rel_err 0.14%, omega <0.2%), corrupt them with increasing noise/blur as a stand-in for
imperfect rendering, and measure how fast rho(GT, measured) decays for each quantity. If alpha decays much
faster than omega at matched corruption, then a large part of the 0.977 -> 0.809 ceiling gap is READOUT
difficulty intrinsic to estimating curvature, and attributing it to the decoder would overstate the case.
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

from src.analysis.ball_tracking import measured_angaccel, measured_angvel
from src.analysis.velocity_ops import clip_angaccel
from src.data.moving_ball import MovingBall

DARK, RED = 0.25, 0.08


def corrupt(frames, sigma, rng):
    """Additive Gaussian pixel noise -- a neutral stand-in for imperfect rendering."""
    if sigma <= 0:
        return frames
    x = frames + torch.from_numpy(rng.normal(0, sigma, tuple(frames.shape)).astype(np.float32))
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=16)
    ap.add_argument("--clips_per_scene", type=int, default=8)
    ap.add_argument("--sigmas", default="0,0.02,0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    K = args.clips_per_scene
    rng = np.random.default_rng(0)

    gens = {
        "angvel": MovingBall(image_size=128, num_frames=16, fps=4, scenario="scene_angvel2d",
                             clips_per_scene=K, omega_range=(0.06, 0.20), radius_range=(0.32, 0.42), seed=11),
        "angaccel": MovingBall(image_size=128, num_frames=16, fps=4, scenario="scene_angaccel2d",
                               clips_per_scene=K, omega0_range=(-0.06, 0.06), alpha_range=(0.005, 0.014),
                               radius_range=(0.32, 0.42), seed=11),
    }
    clips = {q: [g.generate(i) for i in range(args.n_scenes * K)] for q, g in gens.items()}
    gt = {
        "angvel": np.array([float(np.asarray(c.state)[0, list(c.state_keys).index("obj0_omega")])
                            for c in clips["angvel"]]),
        "angaccel": np.array([float(np.asarray(c.state)[:, list(c.state_keys).index("obj0_alpha")].mean())
                              for c in clips["angaccel"]]),
    }

    rows = []
    print("\n=========== READOUT-NOISE CONTROL: omega (slope) vs alpha (curvature) on GT frames ===========")
    print(f"  {'sigma':>6} | {'omega rho':>10} {'valid':>6} | {'alpha rho':>10} {'valid':>6}")
    print("  " + "-" * 52)
    for sg in sigmas:
        rec = {"sigma": sg}
        for q in ("angvel", "angaccel"):
            hat = []
            for c in clips[q]:
                fr = corrupt(c.frames, sg, rng)
                if q == "angvel":
                    hat.append(measured_angvel(fr, darkness_thresh=DARK, red_thresh=RED)["omega"])
                else:
                    hat.append(measured_angaccel(fr, darkness_thresh=DARK, red_thresh=RED)["alpha"])
            hat = np.array(hat)
            ok = np.isfinite(hat)
            rho = float(np.corrcoef(gt[q][ok], hat[ok])[0, 1]) if ok.sum() > 3 else float("nan")
            rec[q] = dict(rho=rho, frac_valid=float(ok.mean()))
        rows.append(rec)
        print(f"  {sg:>6.2f} | {rec['angvel']['rho']:>+10.3f} {rec['angvel']['frac_valid']:>6.2f} | "
              f"{rec['angaccel']['rho']:>+10.3f} {rec['angaccel']['frac_valid']:>6.2f}")

    # the corruption at which each quantity falls to the observed decoder ceiling
    print("\n  Observed decoder ceilings (decode of the TRUE H_b): omega 0.977, alpha 0.809")
    for q, ceil in (("angvel", 0.977), ("angaccel", 0.809)):
        hit = next((r["sigma"] for r in rows if np.isfinite(r[q]["rho"]) and r[q]["rho"] < ceil), None)
        print(f"    {q:<9}: GT-frame readout first drops below its ceiling at sigma = {hit}")
    print("\n  If alpha's rho collapses at a much smaller sigma than omega's, then estimating CURVATURE from"
          "\n  imperfect pixels -- not the renderer -- accounts for much of the 0.977 -> 0.809 ceiling gap.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(sigmas=sigmas, rows=rows, ceilings=dict(angvel=0.977, angaccel=0.809)),
                  open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
