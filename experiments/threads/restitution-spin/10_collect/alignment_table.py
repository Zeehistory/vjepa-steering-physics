#!/usr/bin/env python
"""Collect every fitted operator's ALIGNMENT and put it next to the ceiling. The central table.

Alignment, not gain, is the quantity that decides whether a steering bar is reachable. Because the
operators are linear, rescaling an edit multiplies gain and leak together, so:

    max gain achievable by ANY rescaling of an operator == that operator's alignment (cosine with the
    true displacement)

An 80-90% steering bar therefore needs alignment >= 0.8, and no amount of gain calibration can
manufacture it. (Gain calibration is what took the robotics loop 54.8% -> 94.7%; it cannot work here,
and that is a statement about this latent rather than about the calibration.)

**The normalizer trap this script exists to avoid.** ``latent_crosstalk.json`` reports, per edit,
``gain`` (in units of ``|D_vel|``), ``leak`` (in units of ``|D_spin_perp|``) and ``norm_rel_vel``
(in units of ``|D_vel|``). For the VELOCITY operator, gain and norm share a denominator, so
``gain / norm_rel_vel`` is a genuine cosine. For the SPIN operator it is not: leak and norm are
normalized by DIFFERENT quantities, and dividing them directly overstates spin alignment by
``|D_vel| / |D_spin_perp| ~ 2.5x``. It has to be rescaled first, using

    |D_spin_perp| / |D_vel| = norm_rel_vel(gt_spin) * sqrt(1 - cos(D_vel, D_spin)^2)

both factors of which are already recorded per scene. Doing this by hand once produced a spin alignment
of 0.30 that appeared to beat a ceiling of ~0.00 -- an impossibility that is the tell for a units bug,
the same class of error that invalidated the first two versions of the ceiling metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _alignments(path: Path) -> tuple[float, float, int]:
    rows = json.loads(path.read_text())["rows"]
    av, as_ = [], []
    for x in rows:
        c = float(x["cos_Dvel_Dspin"])
        # |D_spin_perp| expressed in |D_vel| units, so the spin numbers can be put on one scale.
        n_perp = float(x["gt_spin"]["norm_rel_vel"]) * np.sqrt(max(1.0 - c * c, 0.0))
        av.append(x["V"]["gain"] / max(x["V"]["norm_rel_vel"], 1e-12))
        as_.append((x["S"]["leak"] * n_perp) / max(x["S"]["norm_rel_vel"], 1e-12))
    return float(np.median(av)), float(np.median(as_)), len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis_dir", required=True)
    p.add_argument("--ceiling", default="operator_ceiling.json")
    p.add_argument("--layer", type=int, default=18)
    args = p.parse_args()

    ana = Path(args.analysis_dir)
    rows = []
    # One glob, deduped: `latent_crosstalk*_L18.json` already matches the bare `latent_crosstalk_L18`,
    # so unioning it with a second pattern listed the global fit twice.
    for f in sorted(set(ana.glob(f"latent_crosstalk*_L{args.layer}.json"))):
        tag = f.stem.replace("latent_crosstalk", "").replace(f"_L{args.layer}", "").strip("_") or "global"
        try:
            v, s, n = _alignments(f)
        except (KeyError, ValueError) as e:
            print(f"[skip] {f.name}: {e}")
            continue
        rows.append((tag, v, s, n))

    ceil_p = ana / args.ceiling
    ceil = json.loads(ceil_p.read_text()).get(f"L{args.layer}", {}) if ceil_p.exists() else {}
    cv = ceil.get("ceiling_nn_sqrt_vel")
    cs = ceil.get("ceiling_nn_sqrt_spin")

    print(f"\n# Operator alignment vs ceiling (layer {args.layer})\n")
    print("| operator | V alignment | S alignment | n scenes | max gain reachable (=V align) |")
    print("|---|---|---|---|---|")
    for tag, v, s, n in sorted(rows, key=lambda r: -r[1]):
        print(f"| `{tag}` | {v:+.3f} | {s:+.3f} | {n} | {v:.0%} |")
    if cv is not None:
        print(f"| **ceiling (COMMAND-ONLY maps only)** | **{cv:.3f}** | **{cs:.3f}** | — | **{cv:.0%}** |")
        # The ceiling bounds maps that see ONLY the command. `cond*` operators also read the base clip's
        # latent, which is a strictly larger model class, so they are NOT bound by it and exceeding it is
        # expected rather than impossible. Reporting them against it produced a "104% of ceiling" line
        # that reads as a metric bug -- the same signature that genuinely was one twice before. Split the
        # comparison so the bound is only ever applied to the class it actually covers.
        cmd_only = [r for r in rows if r[0] in ("global", "canon")]
        cond = [r for r in rows if r[0].startswith("cond")]
        if cmd_only:
            b = max(r[1] for r in cmd_only)
            print(f"\nCommand-only best: {b:.3f} = {b / max(cv, 1e-9):.0%} of its {cv:.2f} ceiling. "
                  f"An 80-90% bar needs 0.80, so that bar is out of reach for this class.")
        if cond:
            b = max(r[1] for r in cond)
            print(f"Scene-conditioned best: {b:.3f} -- ABOVE the command-only ceiling, which is legal: "
                  f"these read the base latent too. This class is not bounded by {cv:.2f}, and whether "
                  f"it reaches 0.80 is an empirical question about capacity, not a settled impossibility.")
    print("\nNOTE: latent alignment is a CONSERVATIVE proxy. Displacement the operator misses may be "
          "nuisance the decoder ignores; the pixel measurement decides whether steering works.")


if __name__ == "__main__":
    main()
