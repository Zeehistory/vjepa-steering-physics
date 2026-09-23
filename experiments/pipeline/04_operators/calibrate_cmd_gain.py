#!/usr/bin/env python
"""Leakage-free calibration of the cmd_U8 global gain (the one hyperparameter of the command-only operator).

cmd_U8 synthesizes the velocity edit in the global subspace U from the command (no H_b), then applies a
single global gain. Ridge shrinks the predicted U-coordinate magnitude, AND cmd_U8 drops the off-U
component of the true edit, so the decode-optimal gain (~2) must be set against the DECODE objective, not
a latent magnitude match. We pick it on a VALIDATION split of decoded scenes and report on a DISJOINT test
split — standard, leakage-free hyperparameter selection.

Reads a steer2d_summary.json that swept ``cmd_U8_s{gain}`` over scenes, splits scenes val/test by index,
selects the gain minimizing val mean angle error, and reports the held-out test angle error (with the
ceiling/baselines on the same test scenes for context).

    python experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary .../steer_calib/steer2d_summary.json --val_frac 0.5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def angle_err(dec, vb):
    dec = np.asarray(dec, float); vb = np.asarray(vb, float)
    if not np.isfinite(dec).all() or np.linalg.norm(dec) < 1e-9:
        return np.nan
    c = dec @ vb / (np.linalg.norm(dec) * np.linalg.norm(vb) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def mean_ang(ps, scenes, method):
    return float(np.nanmean([angle_err(ps[k][method], ps[k]["v_b"]) for k in scenes]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--val_frac", type=float, default=0.5, help="fraction of scenes used to PICK the gain")
    p.add_argument("--out", default="", help="optional path to write the calibration result json")
    args = p.parse_args()

    d = json.loads(Path(args.summary).read_text())
    ps = d["per_scene"]
    keys = sorted(ps)
    gains = sorted((m for m in d["results"] if re.fullmatch(r"cmd_U8_s[0-9.]+", m)),
                   key=lambda m: float(m.split("_s")[1]))
    if not gains:
        raise SystemExit("no cmd_U8_s{gain} methods found in summary")

    nval = max(1, int(round(len(keys) * args.val_frac)))
    val, test = keys[:nval], keys[nval:]
    val_err = {g: mean_ang(ps, val, g) for g in gains}
    best = min(val_err, key=val_err.get)
    best_gain = float(best.split("_s")[1])

    refs = {m: round(mean_ang(ps, test, m), 2)
            for m in ("full_delta", "ridge_global", "subspace_U8") if m in d["results"]}
    res = {
        "summary": args.summary,
        "n_val": len(val), "n_test": len(test),
        "val_angle_err_by_gain": {g: round(v, 2) for g, v in val_err.items()},
        "selected_gain": best_gain,
        "val_angle_err_at_selected": round(val_err[best], 2),
        "HELDOUT_test_angle_err_at_selected": round(mean_ang(ps, test, best), 2),
        "test_references": refs,
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
