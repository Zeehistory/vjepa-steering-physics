

#!/usr/bin/env python
"""EXP 2b (user 2026-07-07): is the latent CURVATURE command-predictable, and does integrating it beat dH?

Prior arc: every linear command operator that predicts the raw difference dH_t = Z^b_t - Z^a_t plateaus at
held-out reconstruction cos ~0.28 (the "read != write" wall) because dH_t has a t^2-growing temporal profile
that a GLOBAL command->dH map averages away. This script tests a second-order idea: instead of
predicting dH_t, predict the per-step latent CURVATURE

    A_t = Z_{t+1} - 2 Z_t + Z_{t-1}      (second temporal difference, t = 1..T-2)

change  dA_t = A^b_t - A^a_t  (the 2nd diff of dH_t), then REBUILD the edit by discrete double-integration
with reference initial conditions Z*_0 = Z^a_0, Z*_1 = Z^a_1:

    e_t = sum_{s=1}^{t-1} (t - s) * dA_s          (double cumsum; e_0 = e_1 = 0)

Algebra: since dH_t satisfies the same 2nd-difference recurrence as e_t, e_t - dH_t is linear in t, so with
PERFECT dA one gets  e_t = dH_t - dH_0 - t*(dH_1 - dH_0). Thus integration reconstructs dH_t exactly IFF the
initial slabs match (dH_0 = dH_1 = 0), which should hold within a scene (ranks share pos0, v0, appearance).
Why it could beat 0.28: for constant accel, dA_t is ~CONSTANT in t (= the accel change spread uniformly),
so a command->dA map predicts a t-stable low-variance target and integration restores the t^2 profile for
free -- exactly the temporal-profile information the dH operators threw away.

Metrics (held-out test, per layer L):
  * ||dH_t|| profile + ||dH_0||,||dH_1|| vs ||dH_last||   -- IC-match assumption check
  * dA t-constancy (mean pairwise cos of dA_t across t, PCA participation ratio)
  * recon cos command->dH_t          (reproduces the ~0.28 wall; the baseline to beat)
  * recon cos command->dA_t          (is curvature command-predictable?)
  * recon cos e_t(PREDICTED dA) vs dH_t   <-- THE metric: does integrated curvature beat 0.28?
  * recon cos e_t(TRUE dA) vs dH_t        <-- integration CEILING (IC-offset loss only)

All full-D streaming ridge (13-d command features -> 262144-d slab), no subspace, leakage-free. If the
integrated-curvature recon cos >> 0.28, the decoder steer (integrate + write back + decode) is worth building.

    python experiments/threads/acceleration/11_run/curvature_predictability.py --config configs/train/moving_ball_scene_decoder.yaml \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large --output_dir .../curvature
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

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config


def _evict(ds, keep=2):
    if len(ds._shard_cache) > keep:
        ds._shard_cache.clear()


def second_diff(mat):
    """mat (T, D) -> curvature A_t = m[t+1]-2m[t]+m[t-1] for t=1..T-2, returned as {t: (D,)}."""
    T = mat.shape[0]
    return {t: mat[t + 1] - 2 * mat[t] + mat[t - 1] for t in range(1, T - 1)}


def integrate(dA, T):
    """Discrete double-integral with e_0=e_1=0: e_t = sum_{s=1}^{t-1}(t-s) dA_s, for t=2..T-1.

    dA is {s: (D,)} for s=1..T-2. Returns {t: (D,)} for t=2..T-1.
    """
    e = {}
    for t in range(2, T):
        acc = None
        for s in range(1, t):
            if s in dA:
                term = (t - s) * dA[s]
                acc = term if acc is None else acc + term
        e[t] = acc
    return e


class Ridge13:
    """Streaming ridge: 13-d command features -> D-d target, per (layer, t)."""

    def __init__(self, D, ridge=1.0):
        self.XtX = np.zeros((13, 13)); self.XtY = np.zeros((13, D)); self.ridge = ridge; self.n = 0

    def add(self, x, y):
        self.XtX += np.outer(x, x); self.XtY += np.outer(x, y); self.n += 1

    def solve(self):
        self.B = np.linalg.solve(self.XtX + self.ridge * np.eye(13), self.XtY)  # (13, D)

    def predict(self, x):
        return x @ self.B


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_train_scenes", type=int, default=0)
    p.add_argument("--max_test_scenes", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config, [])
    layers = list(cfg.encoder.layers)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dtr = LatentDataset(args.train_dir, layers=layers)
    dte = LatentDataset(args.test_dir, layers=layers)
    T = int(tuple(int(x) for x in dtr[0]["grid"])[0])
    Dfull = int(dtr[0]["layers"][layers[0]].shape[-1]) * int(tuple(int(x) for x in dtr[0]["grid"])[1]) \
        * int(tuple(int(x) for x in dtr[0]["grid"])[2])
    print(f"[curv] layers={layers} T={T} Dslab={Dfull}", flush=True)

    # ---- PASS 1: fit command->dH_t and command->dA_t (full-D streaming ridge) ----
    op_dH = {L: {t: Ridge13(Dfull, args.ridge) for t in range(T)} for L in layers}
    op_dA = {L: {t: Ridge13(Dfull, args.ridge) for t in range(1, T - 1)} for L in layers}
    scenes = vo.group_scenes(dtr)
    sids = sorted(scenes)[: args.max_train_scenes or None]
    for n, s in enumerate(sids):
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = dtr[ia], dtr[ib]
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        cmd = vo.command_features(aa, ab)
        for L in layers:
            Za = np.asarray(sa["layers"][L], dtype=np.float64).reshape(T, Dfull)
            Zb = np.asarray(sb["layers"][L], dtype=np.float64).reshape(T, Dfull)
            dH = Zb - Za                                   # (T, D)
            for t in range(T):
                op_dH[L][t].add(cmd, dH[t])
            dA = second_diff(dH)
            for t, v in dA.items():
                op_dA[L][t].add(cmd, v)
        if (n + 1) % 64 == 0:
            _evict(dtr); print(f"  [fit] {n+1}/{len(sids)}", flush=True)
    for L in layers:
        for t in range(T):
            op_dH[L][t].solve()
        for t in range(1, T - 1):
            op_dA[L][t].solve()
    print(f"[curv] fit done on {len(sids)} scenes", flush=True)

    # ---- PASS 2: held-out eval ----
    scenes_te = vo.group_scenes(dte)
    sids_te = sorted(scenes_te)[: args.max_test_scenes or None]
    # accumulators
    acc = {L: {"dH_recon": np.zeros(T), "dA_recon": {t: 0.0 for t in range(1, T - 1)},
               "e_pred_recon": {t: 0.0 for t in range(2, T)},
               "e_true_recon": {t: 0.0 for t in range(2, T)},
               "dH_norm": np.zeros(T), "dA_norm": {t: 0.0 for t in range(1, T - 1)},
               "dA_tcos": [], "nfin": 0} for L in layers}
    for n, s in enumerate(sids_te):
        ranks = sorted(scenes_te[s]); ia, ib = scenes_te[s][ranks[0]], scenes_te[s][ranks[-1]]
        sa, sb = dte[ia], dte[ib]
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        cmd = vo.command_features(aa, ab)
        for L in layers:
            Za = np.asarray(sa["layers"][L], dtype=np.float64).reshape(T, Dfull)
            Zb = np.asarray(sb["layers"][L], dtype=np.float64).reshape(T, Dfull)
            dH = Zb - Za
            dA_true = second_diff(dH)
            dA_pred = {t: op_dA[L][t].predict(cmd) for t in range(1, T - 1)}
            e_pred = integrate(dA_pred, T)
            e_true = integrate(dA_true, T)
            a = acc[L]; a["nfin"] += 1
            for t in range(T):
                a["dH_norm"][t] += np.linalg.norm(dH[t])
                a["dH_recon"][t] += cos(op_dH[L][t].predict(cmd), dH[t])
            for t in range(1, T - 1):
                a["dA_recon"][t] += cos(dA_pred[t], dA_true[t])
                a["dA_norm"][t] += np.linalg.norm(dA_true[t])
            for t in range(2, T):
                a["e_pred_recon"][t] += cos(e_pred[t], dH[t])
                a["e_true_recon"][t] += cos(e_true[t], dH[t])
            # dA t-constancy: mean pairwise cos of true dA_t across t
            ts = list(dA_true)
            cc = [cos(dA_true[i], dA_true[j]) for k, i in enumerate(ts) for j in ts[k + 1:]]
            a["dA_tcos"].append(float(np.mean(cc)) if cc else 0.0)
        if (n + 1) % 32 == 0:
            _evict(dte)

    summary = {"layers": layers, "T": T, "n_train": len(sids), "n_test": len(sids_te),
               "wall_reference_dH_recon": 0.28, "per_layer": {}}
    for L in layers:
        a = acc[L]; N = max(1, a["nfin"])
        dHrec = (a["dH_recon"] / N).round(3).tolist()
        dHn = (a["dH_norm"] / N).round(2).tolist()
        dArec = {str(t): round(a["dA_recon"][t] / N, 3) for t in range(1, T - 1)}
        epred = {str(t): round(a["e_pred_recon"][t] / N, 3) for t in range(2, T)}
        etrue = {str(t): round(a["e_true_recon"][t] / N, 3) for t in range(2, T)}
        summary["per_layer"][L] = {
            "dH_recon_per_t": dHrec, "dH_norm_per_t": dHn,
            "dH0_over_dHlast": round(dHn[0] / (dHn[-1] + 1e-9), 3),
            "dH1_over_dHlast": round(dHn[1] / (dHn[-1] + 1e-9), 3),
            "dA_recon_per_t": dArec,
            "dA_recon_mean": round(float(np.mean([a["dA_recon"][t] / N for t in range(1, T - 1)])), 3),
            "dA_tconstancy_cos": round(float(np.mean(a["dA_tcos"])), 3),
            "e_pred_recon_vs_dH": epred, "e_pred_recon_mean": round(float(np.mean([a["e_pred_recon"][t] / N for t in range(2, T)])), 3),
            "e_true_recon_ceiling": etrue, "e_true_recon_mean": round(float(np.mean([a["e_true_recon"][t] / N for t in range(2, T)])), 3),
        }
        pl = summary["per_layer"][L]
        print(f"\n== L{L} ==  dH_recon(t)={dHrec}  (mean {np.mean(dHrec):.3f}; wall 0.28)")
        print(f"   dA_recon mean={pl['dA_recon_mean']}  dA_tconstancy_cos={pl['dA_tconstancy_cos']}")
        print(f"   INTEGRATED e(pred dA) recon vs dH = {pl['e_pred_recon_mean']}  "
              f"[ceiling e(true dA) = {pl['e_true_recon_mean']}]")
        print(f"   ||dH_0||/last={pl['dH0_over_dHlast']} ||dH_1||/last={pl['dH1_over_dHlast']}  dH_norm={dHn}",
              flush=True)
    (out / "curvature_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[curv] -> {out}/curvature_summary.json", flush=True)


if __name__ == "__main__":
    main()
