

#!/usr/bin/env python
"""Latent-space SEARCH for a better command-only acceleration operator (Step 2 "Bravo" debug).

The committed canon operator moves decoded pixels ~14.5deg but the accel PROBE reveals it installs the
WRONG latent representation: probe(steered)->a_b = 36.8deg vs probe(H_b)->a_b = 1.9deg upper bound. So the
operator recovers only cos~0.28 of the true Delta H. This script screens candidate operators PURELY in
latent+probe space (no decoder, CPU, minutes) to find one whose steered latent reads as the target accel.

KEY INSIGHT: the accel probe reads the TEMPORAL-POOLED rep r(H) = spatial-mean per frame (T*D,), which is
spatial-roll-INVARIANT -- so canon/placement is invisible to it. The honest lever is the temporal profile
of the edit. And for the probe, the command determines Delta a = a_b - a_a EXACTLY, so an operator that
targets the probe-read direction reads perfectly regardless of v0/pos0/appearance.

Operators screened (all command-only, reconstruct edit from command + H_a, never H_b):
  base       cmd -> U_canon coords -> roll back  (the committed 14.5deg operator; expect ~37deg on probe)
  cmd_prof   fit cmd(13) -> r(Delta H) temporal profile (T*D); edit = spatially-BROADCAST profile
  v0_prof    fit [cmd(13), v0(2)] -> r(Delta H); tests whether reading v0 from H_a helps the profile
  probe_ax   MIN-NORM edit along the probe axis s.t. probe reads exactly Delta a: r(edit)=Wp (WpᵀWp)⁻¹ Δa,
             broadcast spatially (guaranteed probe-correct by construction -> the DECODE is the real test)

Per operator, per layer, held-out TEST: delta_probe_cos (direction the operator moves the probe reading vs
the target a_b-a_a), probe_angle @ train-calibrated gain, recon_cos (cos(edit, true Delta H) full-D), and
recon_cos_pool (cos in temporal-pool space). Also writes the winning operator's per-layer artifacts
(cmd_prof / v0_prof B matrices, probe Wp/mean) so steer_accel2d can decode it.

    python experiments/threads/acceleration/04_operators/accel_operator_search.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --artifacts_dir .../subspace --layers 6,12,18,23 --output_dir .../accel_opsearch
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
import gc
import json
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

P = vo.COMMAND_FEATURE_DIM  # 13


def temporal_pool(flat: np.ndarray, grid) -> np.ndarray:
    """(T*H*W*D,) -> (T, D) spatial mean per frame."""
    T, H, W = grid
    D = flat.size // (T * H * W)
    return flat.reshape(T, H, W, D).mean(axis=(1, 2))


def broadcast_profile(profile_td: np.ndarray, grid) -> np.ndarray:
    """(T, D) temporal profile -> flat (T*H*W*D,) constant across spatial tokens (spatial-mean = profile)."""
    T, H, W = grid
    D = profile_td.shape[1]
    return np.broadcast_to(profile_td[:, None, None, :], (T, H, W, D)).reshape(-1).copy()


def ridge_fit(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, mx, my


def clear(ds):
    if hasattr(ds, "_shard_cache") and len(ds._shard_cache) > 2:
        ds._shard_cache.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--layers", default="6,12,18,23")
    ap.add_argument("--probe_lam", type=float, default=10.0)
    ap.add_argument("--prof_ridge", type=float, default=1.0)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--gains", default="0.5,1,1.5,2,2.5,3,4,6,8")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    gains = [float(g) for g in args.gains.split(",")]
    art = Path(args.artifacts_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    trs, tes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        trs = {s: trs[s] for s in sorted(trs)[: args.max_scenes]}
        tes = {s: tes[s] for s in sorted(tes)[: args.max_scenes]}
    print(f"[opsearch] train {len(trs)} scenes, test {len(tes)}; layers={layers}", flush=True)

    Ucanon = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
    Wu = {L: np.load(art / f"cmd_Wu_canon_L{L}.npy").astype(np.float64) for L in layers}

    # ---- PASS 1 (train): accumulate for probe (r(H)->a), cmd_prof (cmd->r(dH)), v0_prof ([cmd,v0]->r(dH))
    # probe: fit on ALL train clips (r(H), a). profile ops: fit on pairs (a-rank ref -> b-rank), target
    # r(dH)=r(Hb)-r(Ha). Reference rank = rank 0 (a_a), matching the operator's steer-time contract.
    probe_ls = {L: None for L in layers}     # accumulate via ridge on stacked arrays (small: T*D)
    prof_ls = {L: vo.LinearLS(P, 0) for L in layers}
    v0_ls = {L: vo.LinearLS(P + 2, 0) for L in layers}
    rH_all = {L: [] for L in layers}
    a_all = []
    ntd = {}
    for s in sorted(trs):
        ranks = sorted(trs[s]); ia = trs[s][ranks[0]]
        sa = tr[ia]; grid = tuple(int(x) for x in sa["grid"])
        aa = vo.clip_acceleration(sa); v0 = vo.clip_velocity(sa)
        rHa = {L: temporal_pool(vo.layer_flat(sa["layers"][L]), grid) for L in layers}  # (T,D)
        for L in layers:
            rH_all[L].append(rHa[L].reshape(-1)); ntd[L] = rHa[L].size
        a_all.append(aa)
        for b in ranks[1:]:
            sb = tr[trs[s][b]]; ab = vo.clip_acceleration(sb)
            phi = vo.command_features(aa, ab).reshape(1, P)
            phiv = np.concatenate([phi, v0.reshape(1, 2)], axis=1)
            a_all.append(ab)  # once per clip, aligned with the single rH_all[L] append below
            for L in layers:
                rHb = temporal_pool(vo.layer_flat(sb["layers"][L]), grid)
                rH_all[L].append(rHb.reshape(-1))
                dprof = (rHb - rHa[L]).reshape(1, -1)
                if prof_ls[L].out_dim == 0:  # lazy-init out dim
                    prof_ls[L] = vo.LinearLS(P, dprof.shape[1], args.prof_ridge)
                    v0_ls[L] = vo.LinearLS(P + 2, dprof.shape[1], args.prof_ridge)
                prof_ls[L].add(phi, dprof)
                v0_ls[L].add(phiv, dprof)
        clear(tr)
    a_all = np.asarray(a_all)
    for L in layers:
        X = np.asarray(rH_all[L])
        probe_ls[L] = ridge_fit(X, a_all[: len(X)], args.probe_lam)  # (Wp,(M,2)), mx,my
        del X
    rH_all = None; gc.collect()
    Bprof = {L: prof_ls[L].solve() for L in layers}
    Bv0 = {L: v0_ls[L].solve() for L in layers}
    del prof_ls, v0_ls; gc.collect()
    print("[opsearch] fitted probe + cmd_prof + v0_prof", flush=True)

    # probe-axis pseudo-inverse per layer: r(edit) = Wp (WpᵀWp)⁻¹ Δa
    Wpinv = {}
    for L in layers:
        Wp = probe_ls[L][0]  # (M,2)
        Wpinv[L] = Wp @ np.linalg.inv(Wp.T @ Wp + 1e-6 * np.eye(2))  # (M,2): profile = Wpinv @ Δa

    ops = ["base", "cmd_prof", "v0_prof", "probe_ax"]

    def edit_for(op, L, aa, ab, v0, grid, Ha_flat, sh):
        cmd = vo.command_features(aa, ab)
        da = (ab - aa).reshape(2)
        if op == "base":
            return vo.roll_layer((cmd @ Wu[L]) @ Ucanon[L], grid, (-sh[0], -sh[1]))
        if op == "cmd_prof":
            prof = (cmd @ Bprof[L]).reshape(grid[0], -1)
            return broadcast_profile(prof, grid)
        if op == "v0_prof":
            prof = (np.concatenate([cmd, v0]) @ Bv0[L]).reshape(grid[0], -1)
            return broadcast_profile(prof, grid)
        if op == "probe_ax":
            prof = (Wpinv[L] @ da).reshape(grid[0], -1)
            return broadcast_profile(prof, grid)

    # ---- PASS 2 (test): per op/layer, calibrate probe reading. Metrics gain-free where possible. -----
    # We collect: probe predicted Delta (Wpᵀ r(edit)) direction vs target Δa; recon cos vs true dH.
    bestL = max(layers, key=lambda L: 1)  # placeholder; pick probe metric per layer below
    acc = {op: {L: {"dprobe_cos": [], "recon_cos": [], "recon_pool_cos": [],
                    "edit_pool_dot_Wp": [], "da_norm": []} for L in layers} for op in ops}
    # store per (scene) predicted probe-delta and true target for gain calibration
    store = {op: {L: {"dpred": [], "da": [], "aa": []} for L in layers} for op in ops}
    for s in sorted(tes):
        ranks = sorted(tes[s]); ia = tes[s][ranks[0]]
        sa = te[ia]; grid = tuple(int(x) for x in sa["grid"])
        aa = vo.clip_acceleration(sa); v0 = vo.clip_velocity(sa)
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[tes[s][b]]; ab = vo.clip_acceleration(sb)
            da = ab - aa
            for L in layers:
                Wp, mx, my = probe_ls[L]
                dH_true = vo.layer_flat(sb["layers"][L]) - Ha[L]
                rdH_true = temporal_pool(dH_true, grid).reshape(-1)
                for op in ops:
                    e = edit_for(op, L, aa, ab, v0, grid, Ha[L], sh)
                    re = temporal_pool(e, grid).reshape(-1)  # r(edit)
                    dpred = re @ Wp  # (2,) how much the probe reading moves per unit gain
                    acc[op][L]["dprobe_cos"].append(vo.cosine(dpred, da))
                    acc[op][L]["recon_cos"].append(vo.cosine(e, dH_true))
                    acc[op][L]["recon_pool_cos"].append(vo.cosine(re, rdH_true))
                    store[op][L]["dpred"].append(dpred)
                    store[op][L]["da"].append(da)
                    store[op][L]["aa"].append(aa)
        del Ha; gc.collect(); clear(te)

    # gain calibration: choose single gain g per (op,L) minimizing mean angle of (aa + g*dpred) vs ab=aa+da,
    # picked on first-half scenes, reported on second half (leakage-free within test).
    summary = {"layers": layers, "ops": ops, "per_op": {}}
    for op in ops:
        summary["per_op"][op] = {}
        for L in layers:
            dpred = np.asarray(store[op][L]["dpred"]); da = np.asarray(store[op][L]["da"])
            aa = np.asarray(store[op][L]["aa"])
            n = len(da); half = n // 2
            def ang(g, sl):
                pred = aa[sl] + g * dpred[sl]; tgt = aa[sl] + da[sl]
                c = (pred * tgt).sum(1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-12)
                return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())
            gv = min(gains, key=lambda g: ang(g, slice(0, half)))
            summary["per_op"][op][str(L)] = {
                "dprobe_cos": round(float(np.mean(acc[op][L]["dprobe_cos"])), 3),
                "recon_cos": round(float(np.mean(acc[op][L]["recon_cos"])), 3),
                "recon_pool_cos": round(float(np.mean(acc[op][L]["recon_pool_cos"])), 3),
                "calib_gain": gv,
                "probe_angle_heldout": round(ang(gv, slice(half, n)), 2),
            }
            r = summary["per_op"][op][str(L)]
            print(f"[opsearch] {op:9s} L{L}: dprobe_cos={r['dprobe_cos']:+.3f} recon_cos={r['recon_cos']:+.3f} "
                  f"pool_cos={r['recon_pool_cos']:+.3f} | probe_angle(g={gv})={r['probe_angle_heldout']}deg",
                  flush=True)

    (out / "opsearch_summary.json").write_text(json.dumps(summary, indent=2))
    # save profile-operator artifacts for the eventual decode of the winner
    np.savez(out / "prof_operators.npz",
             **{f"Bprof_L{L}": Bprof[L] for L in layers},
             **{f"Bv0_L{L}": Bv0[L] for L in layers},
             **{f"probeWp_L{L}": probe_ls[L][0] for L in layers},
             **{f"probemx_L{L}": probe_ls[L][1] for L in layers},
             **{f"probemy_L{L}": probe_ls[L][2] for L in layers},
             layers=np.array(layers))
    print(f"[opsearch] wrote {out}/opsearch_summary.json + prof_operators.npz", flush=True)


if __name__ == "__main__":
    main()
