#!/usr/bin/env python
"""Paired bootstrap on the decoder-free gate: is the model-size difference real, or noise?

The pilot reports a mean gate lift per encoder. A difference of +0.05 between two encoders means
nothing without an interval, and the sample is not 350 independent pairs -- it is 50 scenes, each
contributing ~7 correlated pairs. So resample SCENES, not pairs.

The comparison is PAIRED: every encoder sees the identical seeded scenes in sorted order, so
gate_pairs_*_L*.npy are aligned element-wise across encoders and the same scene resample can be
applied to all of them. That removes scene difficulty as a source of variance and is much more
sensitive than comparing two independent intervals.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DEPTH = {"vjepa2_large": {"mid": 12, "late": 23},
         "vjepa2_huge": {"mid": 16, "late": 31},
         "vjepa2_giant": {"mid": 20, "late": 38}}
ORDER = ["vjepa2_large", "vjepa2_huge", "vjepa2_giant"]
SHORT = {"vjepa2_large": "ViT-L", "vjepa2_huge": "ViT-H", "vjepa2_giant": "ViT-g"}


def load(cell: Path, layer: int):
    try:
        return (np.load(cell / f"gate_pairs_cmdU_L{layer}.npy").astype(np.float64),
                np.load(cell / f"gate_pairs_cmdU_shuf_L{layer}.npy").astype(np.float64),
                np.load(cell / f"gate_pairs_scene_L{layer}.npy").astype(int))
    except FileNotFoundError:
        return None


def scene_boot(lift: np.ndarray, scene: np.ndarray, idx_by_scene, boots, rng):
    """Mean lift under `boots` scene-level resamples."""
    scenes = np.arange(len(idx_by_scene))
    out = np.empty(boots)
    for b in range(boots):
        pick = rng.choice(scenes, size=len(scenes), replace=True)
        sel = np.concatenate([idx_by_scene[s] for s in pick])
        out[b] = lift[sel].mean()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/vjepa_sweep/pilot")
    ap.add_argument("--quantity", default="accel2d_mixed")
    ap.add_argument("--boots", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root) / args.quantity
    res = {"quantity": args.quantity, "boots": args.boots, "per_depth": {}}
    rng = np.random.default_rng(0)

    print("=" * 74)
    print(f"PAIRED SCENE-LEVEL BOOTSTRAP ON GATE LIFT -- {args.quantity}  ({args.boots} resamples)")
    print("resampling SCENES (n=50), not pairs: pairs within a scene are correlated")
    print("=" * 74)

    for pos in ("mid", "late"):
        data = {}
        for enc in ORDER:
            got = load(root / enc / "artifacts", DEPTH[enc][pos])
            if got is not None:
                data[enc] = got
        if len(data) < 2:
            print(f"\n-- {pos}: not enough cells with per-pair arrays yet "
                  f"({sorted(SHORT[e] for e in data)}) --")
            continue

        n = min(len(d[0]) for d in data.values())
        scene = next(iter(data.values()))[2][:n]
        # Verify alignment: all encoders must agree on which scene each pair belongs to.
        aligned = all(np.array_equal(d[2][:n], scene) for d in data.values())
        idx_by_scene = [np.flatnonzero(scene == s) for s in np.unique(scene)]

        print(f"\n-- {pos} depth --   n_pairs={n}  n_scenes={len(idx_by_scene)}  "
              f"pair-alignment across encoders: {'OK' if aligned else 'MISMATCH (unpaired!)'}")
        lifts = {e: (d[0][:n] - d[1][:n]) for e, d in data.items()}

        res["per_depth"][pos] = {"n_pairs": int(n), "n_scenes": len(idx_by_scene),
                                 "aligned": bool(aligned), "encoders": {}, "deltas": {}}

        for enc in ORDER:
            if enc not in lifts:
                continue
            bs = scene_boot(lifts[enc], scene, idx_by_scene, args.boots, np.random.default_rng(1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            m = float(lifts[enc].mean())
            res["per_depth"][pos]["encoders"][enc] = {"lift": m, "ci": [float(lo), float(hi)]}
            print(f"   {SHORT[enc]:6s} lift = {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

        if not aligned:
            print("   (skipping paired deltas -- arrays are not aligned)")
            continue
        base = "vjepa2_large"
        if base not in lifts:
            continue
        print(f"   PAIRED deltas vs {SHORT[base]}:")
        for enc in ORDER[1:]:
            if enc not in lifts:
                continue
            d = lifts[enc] - lifts[base]           # paired per-pair difference
            bs = scene_boot(d, scene, idx_by_scene, args.boots, np.random.default_rng(2))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI contains 0)"
            res["per_depth"][pos]["deltas"][enc] = {"delta": float(d.mean()),
                                                    "ci": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0)}
            print(f"     {SHORT[enc]:6s} - {SHORT[base]:6s} = {d.mean():+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
        # Is the LARGEST model the best? That is the actual scaling claim.
        if "vjepa2_giant" in lifts and "vjepa2_huge" in lifts:
            d = lifts["vjepa2_giant"] - lifts["vjepa2_huge"]
            bs = scene_boot(d, scene, idx_by_scene, args.boots, np.random.default_rng(3))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s."
            res["per_depth"][pos]["deltas"]["giant_minus_huge"] = {
                "delta": float(d.mean()), "ci": [float(lo), float(hi)],
                "significant": bool(lo > 0 or hi < 0)}
            print(f"     ViT-g  - ViT-H  = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}"
                  f"   <- monotonic-in-size check")

    out = Path(args.out) if args.out else root / "gate_bootstrap.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
