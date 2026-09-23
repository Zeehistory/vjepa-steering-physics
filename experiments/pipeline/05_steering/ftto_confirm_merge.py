#!/usr/bin/env python
"""Merge the ftto_confirm shard JSONs and recompute aggregate stats over all pairs.

Reads bridge/ftto_confirm_shard*.json, pools per_scene records (each holds the target a_b and the
tracked accel 2-vector per init per K), recomputes angle error / magnitude stats over the union,
writes bridge/ftto_confirm_COMBINED.json and prints the K-curve table.
"""
import glob
import json
import sys

import numpy as np

BRIDGE = ("outputs/analysis/"
          "moving_ball_accel2d_mixed/bridge")


def agg(preds, tgts):
    preds, tgts = np.asarray(preds, float), np.asarray(tgts, float)
    pn = np.linalg.norm(preds, axis=1) + 1e-12
    tn = np.linalg.norm(tgts, axis=1) + 1e-12
    cos = np.clip((preds * tgts).sum(1) / (pn * tn), -1, 1)
    ang = np.degrees(np.arccos(cos))
    r = float(np.corrcoef(pn, tn)[0, 1]) if len(pn) > 2 else float("nan")
    return {"n": int(len(ang)),
            "angle_err_deg": round(float(ang.mean()), 3),
            "angle_median_deg": round(float(np.median(ang)), 3),
            "mag_ratio_median": round(float(np.median(pn / tn)), 3),
            "mag_corr_r": round(r, 3)}


def main():
    shards = sorted(glob.glob(f"{BRIDGE}/ftto_confirm_shard*.json"))
    if not shards:
        sys.exit("no shard JSONs found")
    per_scene, params = {}, []
    for p in shards:
        d = json.load(open(p))
        per_scene.update(d["per_scene"])
        params.append({k: d["params"][k] for k in
                       ("n_test_scenes", "scene_offset", "pairs_per_scene", "n_pairs_total")})
    inits = ("posefield", "zero")
    ks = sorted({int(k) for rec in per_scene.values()
                 for init in inits if init in rec for k in rec[init]})
    summary = {}
    for init in inits:
        summary[init] = {}
        for k in ks:
            preds, tgts = [], []
            for rec in per_scene.values():
                if init in rec and str(k) in rec[init]:
                    preds.append(rec[init][str(k)])
                    tgts.append(rec["a_b"])
            summary[init][str(k)] = agg(preds, tgts)
    out = {"idea": "fieldinit_tto_confirm",
           "summary": f"CONFIRMATION run: {len(per_scene)} pairs pooled from {len(shards)} shard(s), "
                      "3 pairs/scene over all 100 held-out scenes",
           "shards": params, "results": summary, "per_scene": per_scene}
    json.dump(out, open(f"{BRIDGE}/ftto_confirm_COMBINED.json", "w"), indent=2)
    print(f"pooled pairs: {len(per_scene)} from {len(shards)} shard(s)")
    print("==================== CONFIRMATION K-curve (held-out) ====================")
    for init in inits:
        for k in ks:
            print(f"  init={init:9s} K={k:2d}: {summary[init][str(k)]}")
    print("  original n=20: pf@20=6.7deg/+0.86 | zero@20=13.6deg/+0.77")


if __name__ == "__main__":
    main()
