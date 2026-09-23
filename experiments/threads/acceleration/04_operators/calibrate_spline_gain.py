#!/usr/bin/env python
"""Leakage-free gain selection for the spline acceleration operator.

Each spline arm has one hyperparameter — a global gain — and the decode-optimal value differs by knot
count (a constant-in-time edit needs a much larger gain than a shaped one). Picking that gain on the
same scenes the number is reported on would be selection leakage, and the K-vs-K comparison is exactly
the kind that leakage would distort.

So: split the scenes in half by index, pick each arm's gain on the VALIDATION half, and report it on the
DISJOINT test half. Baselines that have no gain (noop, full_delta, prof_full, proj_K*) are reported on
the same test half so every number in the table describes the same scenes.

Merges any number of steer2d_summary.json files, so a truncated sweep plus its extension are treated as
one gain grid — necessary here, because the first sweep stopped at gain 3.0 while the K=1 arm's optimum
is at 5.0, which would have understated the constant-in-time baseline.

    python experiments/threads/acceleration/04_operators/calibrate_spline_gain.py --summaries A/steer2d_summary.json B/steer2d_summary.json \
        --val_frac 0.5 --out .../calib_spline_gain.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Any arm name of the form <family>_s<gain> is a gain family. The pattern used to enumerate only
# spline_K*/shufT_K*, which silently dropped every other swept arm into the ungained "plain" bucket --
# where each gain would be printed on the TEST half with no val selection, i.e. an invitation to read
# the best gain off the reported column. The oracle arms (full_delta, prof_full, proj_K*) are swept now,
# so the pattern has to be general.
ARM_RE = re.compile(r"^(?P<fam>.+)_s(?P<gain>\d+(?:\.\d+)?)$")


def angle_err(dec, tgt) -> float:
    d = np.asarray(dec, float); t = np.asarray(tgt, float)
    if not np.isfinite(d).all() or np.linalg.norm(d) < 1e-12:
        return np.nan
    c = float(d @ t / (np.linalg.norm(d) * np.linalg.norm(t) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--val_frac", type=float, default=0.5)
    p.add_argument("--baseline", default="spline_K8_s2.5",
                   help="arm each calibrated family is paired against, scene-level bootstrap. Default is "
                        "the unconstrained per-token profile at its calibrated gain -- the operator the "
                        "spline arm must beat, since K=8 IS that operator by construction")
    p.add_argument("--out", default="")
    args = p.parse_args()

    per_scene: dict[str, dict] = defaultdict(dict)
    for s in args.summaries:
        d = json.loads(Path(s).read_text())
        for scene, row in (d.get("per_scene") or {}).items():
            per_scene[scene].update(row)

    scenes = sorted(per_scene)
    # Split by SCENE, never by row. With --pairs all a scene contributes 7 rows that share an anchor
    # clip; splitting rows would put the same anchor on both sides of the split.
    def scene_of(key: str) -> str:
        row = per_scene[key]
        return f"scene{int(row['scene']):05d}" if "scene" in row else key

    groups: dict[str, list[str]] = defaultdict(list)
    for k in scenes:
        groups[scene_of(k)].append(k)
    scene_names = sorted(groups)
    n_val_sc = int(round(args.val_frac * len(scene_names)))
    val_sc, test_sc = scene_names[:n_val_sc], scene_names[n_val_sc:]
    val = [k for s in val_sc for k in groups[s]]
    test = [k for s in test_sc for k in groups[s]]
    if not val or not test:
        raise SystemExit("need a non-empty val and test split")

    methods = sorted({m for r in per_scene.values() for m in r
                      if m not in ("v_b", "a_a", "scene", "rank")})

    # Common finite mask: a row counts only if EVERY arm tracked on it. Otherwise an arm that destroys
    # the ball on the hard rows is scored on the easy ones and looks better than its rivals.
    def errs(key, method):
        return angle_err(per_scene[key][method], per_scene[key]["v_b"])

    usable = [k for k in scenes
              if all(m in per_scene[k] and np.isfinite(errs(k, m)) for m in methods)]
    dropped = len(scenes) - len(usable)
    keep = set(usable)
    val = [k for k in val if k in keep]
    test = [k for k in test if k in keep]

    err_cache = {m: {k: errs(k, m) for k in usable} for m in methods}

    def mean_err(split, method):
        vals = [err_cache[method][k] for k in split]
        return float(np.mean(vals)) if vals else float("nan")

    def paired_ci(split, method, base, n_boot=20000, seed=0):
        """Paired bootstrap CI of (method - base) over the split, resampled at SCENE level.

        Scene-level resampling because the 7 rows of a scene share an anchor clip and are correlated;
        resampling rows would understate the CI by treating them as independent.
        """
        if base not in err_cache:
            return None
        by_scene: dict[str, list[float]] = defaultdict(list)
        for k in split:
            by_scene[scene_of(k)].append(err_cache[method][k] - err_cache[base][k])
        units = [float(np.mean(v)) for v in by_scene.values()]
        if len(units) < 2:
            return None
        u = np.asarray(units)
        rng = np.random.default_rng(seed)
        bs = rng.choice(u, size=(n_boot, len(u)), replace=True).mean(axis=1)
        return {"delta_deg": round(float(u.mean()), 3),
                "ci95": [round(float(np.percentile(bs, 2.5)), 2),
                         round(float(np.percentile(bs, 97.5)), 2)],
                "p_better": round(float((bs < 0).mean()), 3),
                "n_scenes": len(u)}
    families: dict[str, list[tuple[float, str]]] = defaultdict(list)
    plain = []
    for m in methods:
        mt = ARM_RE.match(m)
        if mt:
            families[mt.group("fam")].append((float(mt.group("gain")), m))
        else:
            plain.append(m)

    out = {"n_rows": len(scenes), "n_usable_rows": len(usable), "n_rows_dropped": dropped,
           "n_scenes": len(scene_names), "n_val_scenes": len(val_sc), "n_test_scenes": len(test_sc),
           "n_val_rows": len(val), "n_test_rows": len(test),
           "val_frac": args.val_frac, "summaries": args.summaries, "baseline": args.baseline,
           "protocol": ("gain chosen on the val half, reported on the disjoint test half; split by "
                        "SCENE; all arms scored on the common rows where every arm tracked"),
           "calibrated": {}, "ungained": {}}

    for fam, arms in sorted(families.items()):
        arms.sort()
        scored = [(mean_err(val, m), g, m) for g, m in arms]
        scored = [x for x in scored if np.isfinite(x[0])]
        if not scored:
            continue
        _, best_gain, best_m = min(scored)
        rec = {
            "chosen_gain": best_gain,
            "val_angle_err_deg": round(mean_err(val, best_m), 2),
            "test_angle_err_deg": round(mean_err(test, best_m), 2),
            "test_median_deg": round(float(np.median([err_cache[best_m][k] for k in test])), 2),
            "test_win_rate_lt20deg": round(
                float(np.mean([err_cache[best_m][k] < 20.0 for k in test])), 3),
            "gains_available": [g for g, _ in arms],
        }
        ci = paired_ci(test, best_m, args.baseline)
        if ci is not None and best_m != args.baseline:
            rec["vs_baseline_paired"] = ci
        # Stratify by target rank. Under --pairs all the 7 pairs of a scene span a 45deg step up to a
        # near-reversal, so a single mean hides an arm that only wins on the easy end.
        by_rank: dict[str, list[float]] = defaultdict(list)
        for k in test:
            r = per_scene[k].get("rank")
            if r is not None:
                by_rank[str(int(r))].append(err_cache[best_m][k])
        if len(by_rank) > 1:
            rec["test_by_target_rank"] = {r: round(float(np.mean(v)), 2)
                                          for r, v in sorted(by_rank.items(), key=lambda kv: int(kv[0]))}
        out["calibrated"][fam] = rec
    for m in plain:
        e = mean_err(test, m)
        if np.isfinite(e):
            out["ungained"][m] = {"test_angle_err_deg": round(e, 2)}

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"[calib] {len(scene_names)} scenes / {len(usable)} usable rows "
          f"({dropped} dropped for a non-finite arm) -> val {len(val_sc)} sc / {len(val)} rows, "
          f"test {len(test_sc)} sc / {len(test)} rows (disjoint by scene)")
    print(f"  paired baseline: {args.baseline}")
    print("\n  arm                 gain   val°    TEST°   med°   win%   vs baseline (95% CI)")
    for fam, r in sorted(out["calibrated"].items(),
                         key=lambda kv: kv[1]["test_angle_err_deg"]):
        ci = r.get("vs_baseline_paired")
        tail = (f"  {ci['delta_deg']:+6.2f} [{ci['ci95'][0]:+.2f},{ci['ci95'][1]:+.2f}] "
                f"p={ci['p_better']}") if ci else ""
        print(f"  {fam:18s} {r['chosen_gain']:>5}  {r['val_angle_err_deg']:>6}  "
              f"{r['test_angle_err_deg']:>6} {r['test_median_deg']:>6} "
              f"{r['test_win_rate_lt20deg']:>6}{tail}")
    print("\n  no-gain references (same test half)")
    for m, r in sorted(out["ungained"].items(), key=lambda kv: kv[1]["test_angle_err_deg"]):
        print(f"  {m:18s}              {r['test_angle_err_deg']:>6}")
    if args.out:
        print(f"\n[calib] -> {args.out}")


if __name__ == "__main__":
    main()
