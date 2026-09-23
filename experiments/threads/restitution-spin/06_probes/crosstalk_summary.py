#!/usr/bin/env python
"""Turn ``crosstalk_raw.json`` into the answer: steering gains, crosstalk, and the commutator.

Every effect is a difference from the DECODED ``base`` condition, so a constant bias in the decoder or
in either tracker cancels rather than being charged to the steer.

Definitions (per commutation square, then aggregated by median + IQR across squares):

  velocity gain      ``g_v = <dv, dv_cmd> / |dv_cmd|^2``    -- 1.0 = the commanded change was achieved in
                     full, along the commanded direction. The projection, not ``|dv|/|dv_cmd|``, so that
                     motion in the wrong direction cannot be scored as success.
  spin gain          ``g_s = d_omega / d_omega_cmd``
  spin leakage       ``|d_omega| / |omega_base|``  -- "the spin changed by this fraction of what it was",
                     which is the form the acceptance bar was stated in. Also reported against the
                     commanded spin step (``leak_vs_cmd``), which says how much of a real spin steer the
                     leak amounts to.
  velocity leakage   ``|dv| / |v_base|`` and ``|dv| / |dv_cmd|``, symmetrically.
  commutator         ``|(v,w)_VtS - (v,w)_StV|`` with each axis normalized by its commanded step, so the
                     two incommensurable units combine into one dimensionless number. 0 = the two orders
                     land on the same physical state.

The ``rand_V`` / ``rand_S`` rows are the interpretation key for every leakage number: they are
norm-matched random edits, so they say what an arbitrary perturbation of the same size does to the other
quantity. A leakage figure is only evidence of selectivity if it sits well below its random control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _iqr(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


def _stat(x: list[float]) -> dict:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    q25, q75 = _iqr(a)
    return {"n": int(len(a)), "median": round(float(np.median(a)), 4),
            "q25": round(q25, 4), "q75": round(q75, 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = json.loads(Path(args.raw).read_text())
    acc: dict[str, dict[str, list[float]]] = {}
    comm: dict[str, list[float]] = {"commutator": [], "d_gv": [], "d_gs": []}

    def push(cond: str, key: str, val: float) -> None:
        acc.setdefault(cond, {}).setdefault(key, []).append(val)

    for r in rows:
        c = r["cond"]
        b = c["base"]
        dv_cmd = np.array(r["gt_vb"]) - np.array(r["gt_va"])
        dw_cmd = r["gt_wb"] - r["gt_wa"]
        nv, nw = float(np.linalg.norm(dv_cmd)), abs(dw_cmd)
        if nv < 1e-9 or nw < 1e-9 or not np.isfinite(b["omega"]):
            continue
        vb_base = np.array([b["vel_x"], b["vel_y"]])

        def eff(name: str) -> tuple[float, float] | None:
            if name not in c or not np.isfinite(c[name]["omega"]):
                return None
            dv = np.array([c[name]["vel_x"], c[name]["vel_y"]]) - vb_base
            dw = c[name]["omega"] - b["omega"]
            push(name, "g_v", float(dv @ dv_cmd / (nv ** 2)))
            push(name, "g_s", float(dw / dw_cmd))
            push(name, "spin_leak_rel_base", float(abs(dw) / max(abs(b["omega"]), 1e-9)))
            push(name, "spin_leak_vs_cmd", float(abs(dw) / nw))
            push(name, "vel_leak_rel_base", float(np.linalg.norm(dv) / max(np.linalg.norm(vb_base), 1e-9)))
            push(name, "vel_leak_vs_cmd", float(np.linalg.norm(dv) / nv))
            return float(dv @ dv_cmd / (nv ** 2)), float(dw / dw_cmd)

        for name in c:
            if name != "base":
                eff(name)

        if "V_then_S" in c and "S_then_V" in c:
            a1, a2 = c["V_then_S"], c["S_then_V"]
            if np.isfinite(a1["omega"]) and np.isfinite(a2["omega"]):
                dvv = (np.array([a1["vel_x"], a1["vel_y"]]) - np.array([a2["vel_x"], a2["vel_y"]])) / nv
                dww = (a1["omega"] - a2["omega"]) / nw
                comm["commutator"].append(float(np.sqrt(float(dvv @ dvv) + dww ** 2)))
                comm["d_gv"].append(float(np.linalg.norm(dvv)))
                comm["d_gs"].append(float(abs(dww)))

    summary = {cond: {k: _stat(v) for k, v in d.items()} for cond, d in acc.items()}
    summary["_commutator"] = {k: _stat(v) for k, v in comm.items()}
    summary["_n_squares"] = len(rows)

    def med(cond: str, key: str) -> float:
        return summary.get(cond, {}).get(key, {}).get("median", float("nan"))

    print(f"\n# Crosstalk summary ({len(rows)} commutation squares)\n")
    print("| condition | vel gain g_v | spin gain g_s | spin leak (/base) | vel leak (/base) |")
    print("|---|---|---|---|---|")
    fixed = ["V", "S", "V+S", "joint", "rand_V", "rand_S",
             "gt_vel", "gt_spin", "gt_both", "V_then_S", "S_then_V"]
    gain_conds = sorted([k for k in summary if k.startswith(("V_g", "S_g"))],
                        key=lambda k: (k[0], float(k.split("_g")[1])))
    for cond in fixed + gain_conds:
        if cond not in summary:
            continue
        print(f"| `{cond}` | {med(cond,'g_v'):+.3f} | {med(cond,'g_s'):+.3f} | "
              f"{med(cond,'spin_leak_rel_base')*100:5.1f}% | {med(cond,'vel_leak_rel_base')*100:5.1f}% |")
    # SCALE-INVARIANT SELECTIVITY. The operators are ridge-shrunk, so absolute gains understate the
    # steering and absolute leaks understate the crosstalk by the same factor. For a linear operator
    # both scale together, so the RATIO is the invariant that survives any gain calibration -- it is
    # the number that answers "if I steer velocity all the way, how much does spin move?".
    sel = {}
    for cond in ["V"] + [k for k in summary if k.startswith("V_g")]:
        g, l = med(cond, "g_v"), med(cond, "g_s")
        if np.isfinite(g) and abs(g) > 1e-9:
            sel[cond] = round(float(l / g), 4)
    for cond in ["S"] + [k for k in summary if k.startswith("S_g")]:
        g, l = med(cond, "g_s"), med(cond, "g_v")
        if np.isfinite(g) and abs(g) > 1e-9:
            sel[cond] = round(float(l / g), 4)
    summary["_selectivity_ratio"] = sel
    if sel:
        print("\nscale-invariant leak/gain ratio (0 = perfectly selective):")
        for k in sorted(sel):
            print(f"  {k:10s} {sel[k]:+.3f}")

    # Written last, so the derived selectivity ratios above are included rather than computed after
    # the file is already on disk.
    Path(args.out).write_text(json.dumps(summary, indent=2))

    cm = summary["_commutator"]["commutator"]
    print(f"\ncommutator |SV - VS| (normalized): median {cm['median']:.3f} "
          f"[IQR {cm['q25']:.3f}, {cm['q75']:.3f}], n={cm['n']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
