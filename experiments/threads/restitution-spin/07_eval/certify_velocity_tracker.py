

#!/usr/bin/env python
"""Does the VELOCITY readout leak spin? Certified on ground-truth renders, no decoder involved.

**Why this has to run before any crosstalk number.** The study's headline question is "steer velocity,
how much does spin move" -- and its mirror, "does changing spin move the measured velocity". Both are
read off pixels by ``measured_velocity``, which takes a DARKNESS-WEIGHTED centroid. On this dataset the
ball carries an amber marker whose darkness (0.383) differs from the body's (~0.90), so the marker is a
differently-weighted patch orbiting the disc, and the centroid it produces oscillates at the spin
frequency. An instrument whose velocity reading moves when only spin changes would MANUFACTURE the
crosstalk we are trying to measure.

**Ground-truth frames are the right test bed.** They are the renderer's own output, so the true velocity
is known exactly (``obj0_vel_x/y``) and the marker is definitely present. Any dependence of the readout
error on omega here is the instrument's, with no decoder to blame.

**What is reported**, for the current tracker and the marker-invariant one:
  * ``corr`` of measured vs true velocity components, and median absolute error;
  * ``err_vs_omega_corr`` -- correlation between the readout's velocity ERROR and the clip's omega. This
    is the leak. A clean instrument reads ~0; a contaminated one reads high in magnitude.

A tracker only earns its place if it matches the current one on velocity accuracy AND kills the omega
dependence. Improving the leak by degrading the velocity readout would trade one bias for another.

    PYTHONPATH=. python experiments/threads/restitution-spin/07_eval/certify_velocity_tracker.py \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large --out .../tracker_certification.json
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

import sys
from pathlib import Path as _P

import numpy as np
import torch

from src.analysis.ball_tracking import measured_velocity, measured_velocity_marker_invariant
from src.encoders.feature_extractor import LatentDataset


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_clips", type=int, default=200)
    args = ap.parse_args()

    ds = LatentDataset(args.test_dir, layers=[23])
    keys = list(ds[0]["state_keys"])
    print(f"[cert] state_keys: {keys}", flush=True)

    def col(name):
        return keys.index(name) if name in keys else None

    i_vx, i_vy, i_om = col("obj0_vel_x"), col("obj0_vel_y"), col("obj0_omega")
    if i_vx is None or i_vy is None or i_om is None:
        raise SystemExit(f"need obj0_vel_x/y and obj0_omega in state_keys, got {keys}")

    rows = []
    n = min(args.n_clips, len(ds))
    for i in range(n):
        s = ds[i]
        frames = s["frames"]                       # (T,C,H,W) ground-truth render
        st = np.asarray(s["state"], dtype=np.float64)
        st = st[0] if st.ndim == 2 else st         # per-clip constants
        gt_vx, gt_vy, om = float(st[i_vx]), float(st[i_vy]), float(st[i_om])

        cur = measured_velocity(frames)
        mi = measured_velocity_marker_invariant(frames)
        rows.append({"gt_vx": gt_vx, "gt_vy": gt_vy, "omega": om,
                     "cur_vx": cur["vel_x"], "cur_vy": cur["vel_y"],
                     "mi_vx": mi["vel_x"], "mi_vy": mi["vel_y"]})
        if (i + 1) % 50 == 0:
            print(f"[cert] {i + 1}/{n}", flush=True)

    A = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}
    out = {"n_clips": len(rows)}
    for tag in ("cur", "mi"):
        ex, ey = A[f"{tag}_vx"] - A["gt_vx"], A[f"{tag}_vy"] - A["gt_vy"]
        err = np.hypot(ex, ey)
        out[tag] = {
            "corr_vx": round(_corr(A[f"{tag}_vx"], A["gt_vx"]), 4),
            "corr_vy": round(_corr(A[f"{tag}_vy"], A["gt_vy"]), 4),
            "median_abs_err": round(float(np.nanmedian(err)), 6),
            # THE LEAK: does the velocity error know about omega?
            "err_vs_omega_corr": round(_corr(err, np.abs(A["omega"])), 4),
            "errx_vs_omega_corr": round(_corr(ex, A["omega"]), 4),
            "erry_vs_omega_corr": round(_corr(ey, A["omega"]), 4),
            "n_valid": int(np.isfinite(err).sum()),
        }
    Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n# Velocity tracker certification on GROUND-TRUTH renders "
          f"({out['n_clips']} clips, marker present)\n")
    print("| tracker | corr vx | corr vy | med|err| | err~|omega| | errx~omega | erry~omega |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in (("cur", "current (darkness)"), ("mi", "marker-invariant")):
        d = out[tag]
        print(f"| {name} | {d['corr_vx']:+.3f} | {d['corr_vy']:+.3f} | {d['median_abs_err']:.5f} "
              f"| {d['err_vs_omega_corr']:+.3f} | {d['errx_vs_omega_corr']:+.3f} "
              f"| {d['erry_vs_omega_corr']:+.3f} |")
    print("\nREAD: `err~|omega|` near 0 means the velocity readout does not know about spin. A large")
    print("magnitude for the current tracker is spin->velocity leak IN THE INSTRUMENT, which would be")
    print("indistinguishable from real crosstalk in the steering result.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
