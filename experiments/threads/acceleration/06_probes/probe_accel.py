

#!/usr/bin/env python
"""Acceleration PROBE + probe->steer verification (the velocity-style probe, ported to acceleration).

Answers two questions, latent-only (no decoder):
  (1) DECODABILITY: is acceleration linearly readable from the latent? Train a ridge probe H -> a (2D) on
      TRAIN clips, report held-out R^2 per axis + a MAGNITUDE R^2 (|a|) and angle error, with a
      shuffled-latent control. This localises the pixel finding that per-scene magnitude wasn't achieved:
      if the probe can't read |a| from the TRUE latent either, it's an encoding limit, not a steer limit.
  (2) PROBE->STEER: reconstruct the canon steered latent (H_a + canon edit, the best operator) for each
      TEST scene and read it with the probe. Compare probe(steered) to the target a_b, alongside
      probe(H_a)->a_a and probe(H_b)->a_b (the real-latent upper bound). This is the latent-space analog
      of the pixel-tracked steer metric.

Representations per layer: 'pool' (mean over all tokens) and 'temporal' (spatial-pool per frame, keep the
T time tokens) -- acceleration is second-order so time may matter.

    python experiments/threads/acceleration/06_probes/probe_accel.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --artifacts_dir .../subspace --layers 6,12,18,23 --steer_gain 2.5 --output_dir .../accel_probe
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


QUANTITY_FNS = {          # one harness for every physics quantity in the sweep
    "accel": vo.clip_acceleration,   # also used for `gravity` (a 1-DOF acceleration scenario)
    "velocity": vo.clip_velocity,
    "angvel": vo.clip_angvel,
    "angaccel": vo.clip_angaccel,
}


def reps_from_layer(flat: np.ndarray, grid: tuple[int, int, int]) -> dict:
    """Return {'pool': (D,), 'temporal': (T*D,)} from a flattened (T*H*W*D) layer."""
    T, H, W = grid
    D = flat.size // (T * H * W)
    x = flat.reshape(T, H, W, D)
    return {"pool": x.reshape(-1, D).mean(0), "temporal": x.mean(axis=(1, 2)).reshape(-1)}


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, mx, my


def ridge_pred(X, W, mx, my):
    return (X - mx) @ W + my


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def ang_mag(pred, true):
    cos = (pred * true).sum(1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-12)
    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
    mp, mt = np.linalg.norm(pred, axis=1), np.linalg.norm(true, axis=1)
    magr = float(np.corrcoef(mp, mt)[0, 1])
    return ang, magr, float((mp / (mt + 1e-12)).mean())


def collect(ds, scenes, layers, want_reps, qfn=None):
    """Per clip: reps by layer + accel target. Returns {rep: {L: X}}, Y.

    LatentDataset caches every opened shard (~2GB each) and never evicts, so iterating the whole dataset
    would hold ~130GB of full latents even though we only keep the reduced reps. We evict once the cache
    grows past a couple shards (NOT every scene -- consecutive scenes share a shard, so per-scene clearing
    forces re-reading each 2GB shard ~8x). This bounds memory to ~2 shards while reading each shard once.
    """
    qfn = qfn or vo.clip_acceleration
    X = {r: {L: [] for L in layers} for r in want_reps}
    Y = []
    for s in sorted(scenes):
        for rank, idx in sorted(scenes[s].items()):
            smp = ds[idx]
            grid = tuple(int(x) for x in smp["grid"])
            Y.append(qfn(smp))
            for L in layers:
                rr = reps_from_layer(vo.layer_flat(smp["layers"][L]), grid)
                for r in want_reps:
                    X[r][L].append(rr[r])
        if hasattr(ds, "_shard_cache") and len(ds._shard_cache) > 2:
            ds._shard_cache.clear()
    Y = np.asarray(Y)
    X = {r: {L: np.asarray(X[r][L]) for L in layers} for r in want_reps}
    return X, Y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--steer_gain", type=float, default=2.5, help="canon operator gain for probe->steer")
    p.add_argument("--quantity", choices=sorted(QUANTITY_FNS), default="accel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_probe_steer", action="store_true",
                   help="report probe R^2 only; skip the canon probe->steer section")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    reps = ["pool", "temporal"]

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    print(f"[probe-a] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}", flush=True)

    qfn = QUANTITY_FNS[args.quantity]
    Xtr, Ytr = collect(tr, tr_scenes, layers, reps, qfn)
    Xte, Yte = collect(te, te_scenes, layers, reps, qfn)
    print(f"[probe-a] collected {len(Ytr)} train / {len(Yte)} test clips", flush=True)

    rng = np.random.default_rng(args.seed)
    summary = {"quantity": args.quantity, "layers": layers, "lam": args.lam, "decodability": {}}
    probes = {}  # (rep,L) -> fitted probe, for the steer step
    for r in reps:
        summary["decodability"][r] = {}
        for L in layers:
            W, mx, my = ridge_fit(Xtr[r][L], Ytr, args.lam)
            probes[(r, L)] = (W, mx, my)
            pred = ridge_pred(Xte[r][L], W, mx, my)
            r2v = r2(pred, Yte)
            ang, magr, magratio = ang_mag(pred, Yte)
            # shuffled-latent control
            Xs = Xte[r][L][rng.permutation(len(Xte[r][L]))]
            r2s = r2(ridge_pred(Xs, W, mx, my), Yte)
            mag_r2 = float(r2(np.linalg.norm(pred, axis=1, keepdims=True),
                              np.linalg.norm(Yte, axis=1, keepdims=True))[0])
            row = {"r2_ax": round(float(r2v[0]), 3), "r2_ay": round(float(r2v[1]), 3),
                   "r2_mag": round(mag_r2, 3), "angle_err_deg": round(ang, 2),
                   "mag_corr_r": round(magr, 3), "ctrl_shuffled_r2_ax": round(float(r2s[0]), 3)}
            summary["decodability"][r][str(L)] = row
            print(f"[probe-a] {r:8s} L{L}: R2 a=({row['r2_ax']},{row['r2_ay']}) |a|={row['r2_mag']} "
                  f"ang={row['angle_err_deg']} magr={row['mag_corr_r']} (ctrl {row['ctrl_shuffled_r2_ax']})",
                  flush=True)

    # pick best (rep,L) by mean axis R2 for the probe->steer verification
    best = max(probes, key=lambda k: (summary["decodability"][k[0]][str(k[1])]["r2_ax"]
                                      + summary["decodability"][k[0]][str(k[1])]["r2_ay"]))
    br, bL = best
    print(f"[probe-a] best probe = {br} L{bL} -> using it for probe->steer", flush=True)
    del Xtr, Xte; gc.collect()

    # ---- probe->steer: read the canon steered latent -----------------------------------------------
    # These come from the CANON operator pipeline (fit_command_operators_accel_canon.py), not from
    # accel_subspace.py. When only the probe R^2 is wanted -- e.g. the decoder-free model-size
    # pilot -- skip this section rather than dying after the probe is already computed but before
    # the summary is written.
    canon_needed = [art / f"global_basis_canon_L{L}.npy" for L in layers] + \
                   [art / f"cmd_Wu_canon_L{L}.npy" for L in layers]
    missing = [p.name for p in canon_needed if not p.exists()]
    if args.skip_probe_steer or missing:
        if missing and not args.skip_probe_steer:
            print(f"[probe-a] SKIPPING probe->steer: missing canon artifacts {missing[:3]}"
                  f"{'...' if len(missing) > 3 else ''}. Probe R^2 below is unaffected.", flush=True)
        summary["probe_steer"] = {"skipped": True, "missing": missing}
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "accel_probe_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[probe-a] wrote {out}/accel_probe_summary.json (probe R^2 only)", flush=True)
        return

    Ucanon = {L: np.load(art / f"global_basis_canon_L{L}.npy").astype(np.float64) for L in layers}
    Wu = {L: np.load(art / f"cmd_Wu_canon_L{L}.npy").astype(np.float64) for L in layers}
    W, mx, my = probes[best]
    aa_pred, ab_pred, steer_pred, targ, aa_true_l = [], [], [], [], []
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); ia, ib = te_scenes[s][ranks[0]], te_scenes[s][ranks[-1]]
        sa, sb = te[ia], te[ib]
        grid = tuple(int(x) for x in sa["grid"])
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        sh = vo.canon_shift(vo.clip_start_pos(sa), grid)
        cmd = vo.command_features(aa, ab)
        # reconstruct canon steered latent per layer, then read best-layer rep with the probe
        HaL = vo.layer_flat(sa["layers"][bL])
        edit = vo.roll_layer((cmd @ Wu[bL]) @ Ucanon[bL], grid, (-sh[0], -sh[1])) * args.steer_gain
        steered = HaL + edit
        rp = lambda flat: reps_from_layer(flat, grid)[br].reshape(1, -1)
        aa_pred.append(ridge_pred(rp(HaL), W, mx, my)[0])
        ab_pred.append(ridge_pred(rp(vo.layer_flat(sb["layers"][bL])), W, mx, my)[0])
        steer_pred.append(ridge_pred(rp(steered), W, mx, my)[0])
        targ.append(ab); aa_true_l.append(aa)
        if hasattr(te, "_shard_cache") and len(te._shard_cache) > 2:
            te._shard_cache.clear()
    aa_pred, ab_pred, steer_pred, targ = map(np.asarray, (aa_pred, ab_pred, steer_pred, targ))
    aa_true = np.asarray(aa_true_l)

    def block(name, pred, true):
        ang, magr, magratio = ang_mag(pred, true)
        return {"angle_err_deg": round(ang, 2), "mag_corr_r": round(magr, 3),
                "mag_ratio": round(magratio, 3), "r2_mean": round(float(r2(pred, true).mean()), 3)}

    summary["probe_to_steer"] = {
        "probe": f"{br}_L{bL}", "steer_gain": args.steer_gain,
        "probe_Ha_reads_a_a": block("Ha", aa_pred, aa_true),     # sanity: reads the reference accel
        "probe_Hb_reads_a_b": block("Hb", ab_pred, targ),        # real-latent upper bound
        "probe_STEERED_vs_target_a_b": block("steered", steer_pred, targ),  # the verification
    }
    print("[probe-a] PROBE->STEER (latent-space):", flush=True)
    for k in ("probe_Ha_reads_a_a", "probe_Hb_reads_a_b", "probe_STEERED_vs_target_a_b"):
        v = summary["probe_to_steer"][k]
        print(f"   {k:32s} angle={v['angle_err_deg']:6.2f} magr={v['mag_corr_r']:+.3f} "
              f"magratio={v['mag_ratio']:.3f} R2={v['r2_mean']}", flush=True)

    (out / "accel_probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[probe-a] wrote {out}/accel_probe_summary.json", flush=True)


if __name__ == "__main__":
    main()
