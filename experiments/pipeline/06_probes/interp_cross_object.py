

#!/usr/bin/env python
"""Cross-OBJECT transfer: is the velocity axis object-agnostic (physics) or disk-bound (dataset concept)?

The PI's question — "is the steer a concept tied to the training data, or actual physics?" — is decided by
whether the disk-trained velocity machinery transfers to a NOVEL object identity it never saw. We render the
SAME scene_velocity2d trajectories/velocities with a SQUARE instead of a disk (identical token grid, so the
disk artifacts apply verbatim) and ask three things on the held-out square test set:

  READ transfer  -- does the DISK-trained linear probe read the square's velocity (R^2)? Compare to an
                    oracle probe fit on square-train. Object-agnostic readout => the velocity *encoding*
                    is a physical quantity, not a disk texture.
  WRITE transfer -- do the DISK operators (ridge_global / subspace_U8 / cmd_U8 / per-pair full_delta)
                    install the target velocity into the SQUARE's own latent readout (achieved fraction,
                    angle)? Compare to the disk numbers (from interp_why_steer).
  SUBSPACE align -- principal angle between the disk's shared U8 velocity subspace and the square's own
                    (PCA of square Delta H); and the fraction of square Delta energy captured by the DISK U8
                    vs a random same-rank subspace. Small angle / high retention => the shared, transferable
                    velocity component is the SAME subspace across objects (physics).

All latent-only (no decoder). Self-contained: refits the disk probe from disk-train so it does not depend on
interp_why_steer's run.

    python experiments/pipeline/06_probes/interp_cross_object.py \
        --disk_train .../moving_ball_scene_v2d/train/vjepa2_large \
        --disk_test  .../moving_ball_scene_v2d/test/vjepa2_large \
        --sq_train   .../moving_ball_scene_v2d_square/train/vjepa2_large \
        --sq_test    .../moving_ball_scene_v2d_square/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_v2d/subspace \
        --output_dir outputs/analysis/moving_ball_v2d/interp --layers 12,23
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


def pool_layer(sl) -> np.ndarray:
    return np.asarray(sl, dtype=np.float64).mean(axis=0)


def fit_ridge(X, Y, lam):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_ridge(X, W):
    return np.concatenate([X, np.ones((X.shape[0], 1))], 1) @ W


def r2_cols(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-30
    return [round(float(x), 4) for x in (1 - ss_res / ss_tot)]


def angle_deg(a, b):
    cos = (a * b).sum(1) / (np.linalg.norm(a, 1 if False else None, axis=1) *
                            np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def pooled_and_vel(path, layers):
    """Build a LatentDataset, return {L:(N,1024)} pooled latents + (N,2) GT velocity, then FREE its
    shard cache (the full latents are ~16MB/clip; holding four datasets' caches OOMs even at 256G)."""
    ds = LatentDataset(path, layers=layers)
    n = len(ds)
    X = {L: np.empty((n, 1024)) for L in layers}
    V = np.empty((n, 2))
    for i in range(n):
        s = ds[i]
        V[i] = vo.clip_velocity(s)
        for L in layers:
            X[L][i] = pool_layer(s["layers"][L])
        # clear the shard cache periodically so peak RAM stays ~a few GB (not the whole 64GB cache)
        if hasattr(ds, "_shard_cache") and (i % 256 == 255):
            ds._shard_cache.clear()
    if hasattr(ds, "_shard_cache"):
        ds._shard_cache.clear()
    return X, V


def scene_deltas(ds, layers, scene_ids, scenes):
    """Per-scene anchor->extreme full-flat Delta H for each layer + (va,vb). Lists keyed by layer."""
    dH = {L: [] for L in layers}
    vab = []
    for s in scene_ids:
        ranks = sorted(scenes[s])
        sa, sb = ds[scenes[s][ranks[0]]], ds[scenes[s][ranks[-1]]]
        vab.append((vo.clip_velocity(sa), vo.clip_velocity(sb)))
        for L in layers:
            dH[L].append(vo.layer_flat(sb["layers"][L]) - vo.layer_flat(sa["layers"][L]))
    return {L: np.asarray(dH[L]) for L in layers}, vab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disk_train", required=True)
    p.add_argument("--disk_test", required=True)
    p.add_argument("--sq_train", required=True)
    p.add_argument("--sq_test", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="12,23")
    p.add_argument("--probe_ridge", type=float, default=10.0)
    p.add_argument("--num_scenes", type=int, default=100)
    p.add_argument("--cmd_gain", type=float, default=2.0)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("[xobj] pooling latents one dataset at a time (cache freed after each)...")
    Xdtr, Vdtr = pooled_and_vel(args.disk_train, layers)
    Xdte, Vdte = pooled_and_vel(args.disk_test, layers)
    Xstr, Vstr = pooled_and_vel(args.sq_train, layers)
    Xste, Vste = pooled_and_vel(args.sq_test, layers)

    # only the two TEST sets are kept alive for the per-scene full-flat work (~13GB cache each)
    dte = LatentDataset(args.disk_test, layers=layers)
    ste = LatentDataset(args.sq_test, layers=layers)
    sc_d = vo.group_scenes(dte); ids_d = sorted(sc_d)[: args.num_scenes]
    sc_s = vo.group_scenes(ste); ids_s = sorted(sc_s)[: args.num_scenes]
    print(f"[xobj] disk_test scenes={len(sc_d)} square_test scenes={len(sc_s)}")

    summary = {"layers": layers, "n": {"disk_train": len(Vdtr), "disk_test": len(Vdte),
               "sq_train": len(Vstr), "sq_test": len(Vste)}, "cmd_gain": args.cmd_gain, "layers_detail": {}}

    for L in layers:
        print(f"\n===== L{L} =====")
        Uf = np.load(art / f"global_basis_L{L}.npy").astype(np.float64)[:8]   # disk shared subspace
        Btf = np.load(art / f"ridge_Bt_L{L}.npy").astype(np.float64)
        Wuf = np.load(art / f"cmd_Wu_L{L}.npy").astype(np.float64)
        Dflat = Uf.shape[1]

        # ---- READ transfer ------------------------------------------------------------------------
        Wdisk = fit_ridge(Xdtr[L], Vdtr, args.probe_ridge)         # disk-trained probe
        Wsq = fit_ridge(Xstr[L], Vstr, args.probe_ridge)           # oracle square probe
        read = {
            "disk_probe_on_disk_test_r2": r2_cols(apply_ridge(Xdte[L], Wdisk), Vdte),
            "disk_probe_on_square_test_r2": r2_cols(apply_ridge(Xste[L], Wdisk), Vste),  # TRANSFER
            "oracle_square_probe_on_square_test_r2": r2_cols(apply_ridge(Xste[L], Wsq), Vste),
        }
        print(f"  READ: disk-probe on disk {read['disk_probe_on_disk_test_r2']} | "
              f"on SQUARE {read['disk_probe_on_square_test_r2']} | "
              f"oracle-square {read['oracle_square_probe_on_square_test_r2']}")

        # ---- WRITE transfer: disk operators applied to SQUARE test, read by disk probe ------------
        write = {}
        for obj, ds, scenes, ids in [("disk", dte, sc_d, ids_d), ("square", ste, sc_s, ids_s)]:
            recs = {m: {"vhat": []} for m in ["full_delta", "ridge_global", "subspace_U8", "cmd_U8", "random8"]}
            vb_l, va_l = [], []
            for s in ids:
                ranks = sorted(scenes[s])
                sa, sb = ds[scenes[s][ranks[0]]], ds[scenes[s][ranks[-1]]]
                va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb); dv = vb - va
                Ha = vo.layer_flat(sa["layers"][L]); Hb = vo.layer_flat(sb["layers"][L]); dH = Hb - Ha
                phi = vo.command_features(va, vb)
                edits = {
                    "full_delta": dH,
                    "ridge_global": dv @ Btf,
                    "subspace_U8": vo.project(dH, Uf),
                    "cmd_U8": args.cmd_gain * ((phi @ Wuf) @ Uf),
                    "random8": vo.project(dH, vo.random_basis(Dflat, 8, rng)),
                }
                pa = Ha.reshape(-1, 1024).mean(0); va_l.append(apply_ridge(pa[None], Wdisk)[0])
                vb_l.append(vb)
                for m, e in edits.items():
                    Hs = (Ha + e).reshape(-1, 1024).mean(0)
                    recs[m]["vhat"].append(apply_ridge(Hs[None], Wdisk)[0])
            vb_a = np.asarray(vb_l); va_a = np.asarray(va_l)
            dvec = vb_a - va_a
            mm = {}
            for m in recs:
                vh = np.asarray(recs[m]["vhat"])
                ach = ((vh - va_a) * dvec).sum(1) / ((dvec * dvec).sum(1) + 1e-12)
                ang = angle_deg(vh, vb_a)
                mm[m] = {"readout_angle_deg": round(float(np.mean(ang)), 2),
                         "achieved_frac": round(float(np.median(ach)), 3)}
            write[obj] = mm
            print(f"  WRITE[{obj:6s}] " + " ".join(
                f"{m}:ang={mm[m]['readout_angle_deg']},ach={mm[m]['achieved_frac']}" for m in mm))

        # ---- SUBSPACE alignment: disk U8 vs square's own U8; square Delta energy in disk U8 -------
        dH_sq, _ = scene_deltas(ste, [L], ids_s, sc_s)
        dH_dk, _ = scene_deltas(dte, [L], ids_d, sc_d)
        Usq, _ = vo.pca_gram(dH_sq[L], k=8)        # square's own shared velocity subspace
        Udk_te, _ = vo.pca_gram(dH_dk[L], k=8)     # disk test-derived subspace (sanity vs saved Uf)
        Rrand = vo.random_basis(Dflat, 8, rng)
        def energy_in(deltas, basis):
            num = ((deltas @ basis.T) ** 2).sum(1)
            den = (deltas ** 2).sum(1) + 1e-30
            return float(np.mean(num / den))
        align = {
            "principal_angle_diskU8_vs_squareU8": vo.principal_angles_bases(Uf, Usq),
            "principal_angle_diskU8_vs_disktestU8": vo.principal_angles_bases(Uf, Udk_te),
            "principal_angle_diskU8_vs_random8": vo.principal_angles_bases(Uf, Rrand),
            "square_delta_energy_in_diskU8": round(energy_in(dH_sq[L], Uf), 4),     # cross-object retain
            "square_delta_energy_in_squareU8": round(energy_in(dH_sq[L], Usq), 4),  # within-object ceiling
            "square_delta_energy_in_random8": round(energy_in(dH_sq[L], Rrand), 4),
            "disk_delta_energy_in_diskU8": round(energy_in(dH_dk[L], Uf), 4),
        }
        print(f"  ALIGN: diskU8<->squareU8 mean={align['principal_angle_diskU8_vs_squareU8']['mean_deg']:.1f}deg"
              f" (vs random {align['principal_angle_diskU8_vs_random8']['mean_deg']:.1f}); "
              f"square Delta in diskU8={align['square_delta_energy_in_diskU8']} "
              f"(own {align['square_delta_energy_in_squareU8']}, rand {align['square_delta_energy_in_random8']})")

        summary["layers_detail"][f"L{L}"] = {"read": read, "write": write, "align": align}

    (out / "cross_object_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[xobj] -> {out}/cross_object_summary.json")


if __name__ == "__main__":
    main()
