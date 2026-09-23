#!/usr/bin/env python
"""Score MAGNITUDE control, which the published velocity metrics never tested.

`angle_err_deg` is heading-only by construction, and `mag_ratio` is a ratio of MEANS -- an operator
that always emits the band's typical speed scores ~1.0 on it. So both are ~blind to whether the
commanded SPEED is tracked. The honest test is the regression of achieved speed on commanded speed
across held-out scenes:

    corr(|v_commanded|, |v_achieved|)   and   slope   (perfect control = +1.0, +1.0)

Prints every method in each summary, baseline vs standardized side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def score(summary_path: Path):
    d = json.loads(summary_path.read_text())
    ps = d["per_scene"]
    methods = [m for m in d["results"]]
    out = {}
    for m in methods:
        C, A, ang = [], [], []
        for s, rec in ps.items():
            if m not in rec:
                continue
            vb = np.asarray(rec["v_b"], float)
            va = np.asarray(rec[m], float)
            if not np.all(np.isfinite(va)):
                continue
            C.append(np.linalg.norm(vb)); A.append(np.linalg.norm(va))
            cc = np.dot(vb, va) / (np.linalg.norm(vb) * np.linalg.norm(va) + 1e-12)
            ang.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))
        if len(C) < 3:
            continue
        C = np.array(C); A = np.array(A)
        out[m] = dict(n=len(C), corr=float(np.corrcoef(C, A)[0, 1]),
                      slope=float(np.polyfit(C, A, 1)[0]),
                      mag_ratio=float(A.mean() / C.mean()),
                      angle_err=float(np.mean(ang)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="steer2d_summary.json for the BASELINE fit")
    p.add_argument("--std", required=True, help="steer2d_summary.json for the STANDARDIZED fit")
    p.add_argument("--label", default="")
    args = p.parse_args()

    b, s = score(Path(args.base)), score(Path(args.std))
    print(f"\n{'='*104}\n{args.label}   (corr/slope of achieved speed on commanded speed; "
          f"perfect = +1.000/+1.000)\n{'='*104}")
    print(f"{'method':22s} | {'BASELINE ridge':>34s} | {'STANDARDIZED ridge':>34s}")
    print(f"{'':22s} | {'corr':>8s} {'slope':>7s} {'magrat':>7s} {'ang':>8s} | "
          f"{'corr':>8s} {'slope':>7s} {'magrat':>7s} {'ang':>8s}")
    print("-" * 104)
    for m in sorted(set(b) | set(s)):
        rb, rs = b.get(m), s.get(m)
        f = lambda r: (f"{r['corr']:+8.3f} {r['slope']:+7.3f} {r['mag_ratio']:7.3f} "
                       f"{r['angle_err']:7.1f}d") if r else f"{'-':>8s} {'-':>7s} {'-':>7s} {'-':>8s}"
        print(f"{m:22s} | {f(rb)} | {f(rs)}")


if __name__ == "__main__":
    main()
