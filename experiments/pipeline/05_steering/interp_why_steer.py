

#!/usr/bin/env python
"""Interpretability: WHY does the velocity steer work? (PI question, 2026-06-29)

The PI asks: is the steer working because of a concept tied to the training data, or is V-JEPA actually
representing an abstract physical velocity? The pixel results gave us a sharp tension to exploit:

  * velocity is near-perfectly *readable* from frozen latents (linear probe R^2 ~ 0.99), yet
  * velocity is only partially *writable* (a single global linear operator steers at cos 0.39 / ~34deg,
    a single mean vector fails; only the per-pair on-manifold Delta and the U8-synthesis transfer).

Read != write is the mechanistic crux. This script runs that interp *entirely in the latent space*
(no decoder) so we can separate three places the "physics" could live:

  (A) READ          -- a global linear probe latent->v (is velocity linearly encoded at all?).
  (B) WRITE-then-READ -- probe the *steered* latent H_a*. If the probe reads the target velocity off a
                        steered latent, the edit moved V-JEPA's OWN velocity code (not just fooled the
                        decoder). Comparing this latent-readout angle to the known *decoded* angle tells us
                        whether a method fails at the representation, the manifold, or the rendering.
  (C) MANIFOLD      -- off-manifold residual + nearest-real-neighbour distance of each H_a*. Tests the
                        hypothesis that the per-pair steer works because Delta = H_b - H_a is an
                        ON-MANIFOLD chord between two real encoded states, while the global operator
                        lands off the manifold (so the decoder can't render it).
  (D) Delta SPLIT    -- energy of the per-pair Delta inside the shared global U8 subspace (transferable /
                        "abstract") vs its complement (scene-local / "dataset-bound").
  (E) SATURATION    -- sweep the cmd_U8 gain and read the probe's speed off the steered latent: does the
                        latent velocity coordinate stay LINEAR out past the trained speed range, or does it
                        saturate? (Cross-checks the known decode saturation.)

All operations are on POOLED latents for the probe (mean over the 8x16x16 tokens -> D=1024). That velocity
is readable from the *spatially pooled* latent at all is itself evidence the velocity code is GLOBAL /
distributed, not stored in the ball's location tokens -- consistent with the transport-operator negative.

    python experiments/pipeline/05_steering/interp_why_steer.py \
        --train_dir .../moving_ball_scene_v2d/train/vjepa2_large \
        --test_dir  .../moving_ball_scene_v2d/test/vjepa2_large \
        --artifacts_dir outputs/analysis/moving_ball_v2d/subspace \
        --output_dir outputs/analysis/moving_ball_v2d/interp \
        --layers 12,23
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


def pool_layer(sample_layer) -> np.ndarray:
    """(L_tok, D) -> (D,) mean over tokens (the probe's clip_pool representation)."""
    return np.asarray(sample_layer, dtype=np.float64).mean(axis=0)


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with intercept. X (N,p), Y (N,k) -> W ((p+1),k) (last row = bias)."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ Y)


def apply_ridge(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ W


def r2_per_col(pred: np.ndarray, true: np.ndarray) -> list[float]:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-30
    return [float(x) for x in (1 - ss_res / ss_tot)]


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle (deg) between 2D vectors a,b."""
    dot = (a * b).sum(1)
    cos = dot / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="12,23")
    p.add_argument("--probe_ridge", type=float, default=10.0)
    p.add_argument("--pca_k", type=int, default=256, help="manifold PCA rank (pooled latents)")
    p.add_argument("--num_scenes", type=int, default=100)
    p.add_argument("--cmd_gain", type=float, default=2.0, help="default gain for cmd_U8 steered probe")
    p.add_argument("--gain_sweep", default="-0.5,0,0.5,1,1.5,2,2.5,3,4")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    gains = [float(x) for x in args.gain_sweep.split(",")]

    train = LatentDataset(args.train_dir, layers=layers)
    test = LatentDataset(args.test_dir, layers=layers)
    print(f"[interp] train={len(train)} test={len(test)} clips; layers={layers}")

    # ---- TRAIN pass: pooled latents + GT velocity (single pass, all layers) -------------------------
    n_tr = len(train)
    Xtr = {L: np.empty((n_tr, 1024)) for L in layers}
    Vtr = np.empty((n_tr, 2))
    for i in range(n_tr):
        s = train[i]
        Vtr[i] = vo.clip_velocity(s)
        for L in layers:
            Xtr[L][i] = pool_layer(s["layers"][L])
        if i % 500 == 0:
            print(f"  [train] {i}/{n_tr}")

    # ---- TEST pass: pooled latents + GT velocity ---------------------------------------------------
    n_te = len(test)
    Xte = {L: np.empty((n_te, 1024)) for L in layers}
    Vte = np.empty((n_te, 2))
    for i in range(n_te):
        s = test[i]
        Vte[i] = vo.clip_velocity(s)
        for L in layers:
            Xte[L][i] = pool_layer(s["layers"][L])

    scenes = vo.group_scenes(test)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[interp] {len(scenes)} test scenes; using {len(scene_ids)}")

    # ---- artifacts: global U basis + ridge operator (full-flat) ------------------------------------
    Ubasis = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}  # (8,Dflat)
    Bt = {L: np.load(art / f"ridge_Bt_L{L}.npy").astype(np.float64) for L in layers}          # (2,Dflat)
    Wu = {L: np.load(art / f"cmd_Wu_L{L}.npy").astype(np.float64) for L in layers}            # (13,8)
    rng = np.random.default_rng(0)

    summary = {"train_dir": args.train_dir, "test_dir": args.test_dir, "layers": layers,
               "n_train": n_tr, "n_test": n_te, "n_scenes": len(scene_ids),
               "probe_ridge": args.probe_ridge, "pca_k": args.pca_k, "cmd_gain": args.cmd_gain,
               "layers_detail": {}}

    for L in layers:
        print(f"\n===== layer L{L} =====")
        # (A) READ: global linear velocity probe, held-out R^2 ---------------------------------------
        W = fit_ridge(Xtr[L], Vtr, args.probe_ridge)
        r2 = r2_per_col(apply_ridge(Xte[L], W), Vte)
        print(f"  (A) READ probe R^2 (vx,vy) held-out = {r2}")

        # (C) manifold model: PCA on pooled TRAIN latents -------------------------------------------
        mu = Xtr[L].mean(0)
        Xc = Xtr[L] - mu
        # economy SVD on (N,1024); top pca_k right singular vectors
        _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(args.pca_k, Vt.shape[0])
        P = Vt[:k]  # (k,1024) orthonormal rows = manifold basis
        def resid_frac(Z):  # off-manifold residual fraction of mean-centred pooled latent rows
            Zc = Z - mu
            recon = (Zc @ P.T) @ P
            return np.linalg.norm(Zc - recon, axis=1) / (np.linalg.norm(Zc, axis=1) + 1e-12)
        # baselines: residual of REAL test latents (the on-manifold floor)
        real_resid = float(np.mean(resid_frac(Xte[L])))

        Uf = Ubasis[L][:8]      # (8, Dflat)  shared/global velocity subspace (full-flat)
        Btf = Bt[L]             # (2, Dflat)
        Wuf = Wu[L]             # (13, 8)

        # ---- per-scene: build edits (full-flat), pool, probe, residual, U8 energy ------------------
        methods = ["full_delta", "ridge_global", "subspace_U8", "cmd_U8", "random8"]
        rec = {m: {"vhat": [], "resid": []} for m in methods}
        vb_list, vhat_anchor, anchor_resid = [], [], []
        dH_u8_frac = []  # (D) energy of per-pair Delta inside U8 vs total
        gain_curve = {g: [] for g in gains}  # cmd_U8 readout speed vs gain

        Dflat = Uf.shape[1]
        for s in scene_ids:
            ranks = sorted(scenes[s])
            ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
            sa, sb = test[ia], test[ib]
            va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb)
            dv = vb - va
            Ha_f = vo.layer_flat(sa["layers"][L])   # (Dflat,)
            Hb_f = vo.layer_flat(sb["layers"][L])
            dH = Hb_f - Ha_f

            # Delta split: fraction of per-pair Delta energy inside the shared U8 subspace
            c = Uf @ dH
            dH_u8_frac.append(float((c @ c) / (dH @ dH + 1e-30)))

            phi = vo.command_features(va, vb)        # (13,)
            cU8 = (phi @ Wuf) @ Uf                    # (Dflat,)  command-synth edit (gain 1)
            edits = {
                "full_delta": dH,
                "ridge_global": dv @ Btf,
                "subspace_U8": vo.project(dH, Uf),
                "cmd_U8": args.cmd_gain * cU8,
                "random8": vo.project(dH, vo.random_basis(Dflat, 8, rng)),
            }
            # anchor (unsteered) pooled readout
            pa = Ha_f.reshape(-1, 1024).mean(0)
            vhat_a = apply_ridge(pa[None], W)[0]
            vhat_anchor.append(vhat_a)
            anchor_resid.append(float(resid_frac(pa[None])[0]))
            vb_list.append(vb)
            for m, e in edits.items():
                Hstar = (Ha_f + e).reshape(-1, 1024).mean(0)   # pool the steered full latent
                rec[m]["vhat"].append(apply_ridge(Hstar[None], W)[0])
                rec[m]["resid"].append(float(resid_frac(Hstar[None])[0]))
            # (E) gain sweep on cmd_U8: probe speed off the steered latent vs gain
            for g in gains:
                Hg = (Ha_f + g * cU8).reshape(-1, 1024).mean(0)
                gain_curve[g].append(float(np.linalg.norm(apply_ridge(Hg[None], W)[0])))

        vb_arr = np.asarray(vb_list)
        va_arr = np.asarray(vhat_anchor)
        det = {"read_probe_r2": r2, "real_latent_resid_frac": real_resid,
               "anchor_resid_frac": float(np.mean(anchor_resid)),
               "delta_U8_energy_frac_mean": float(np.mean(dH_u8_frac)),
               "delta_U8_energy_frac_median": float(np.median(dH_u8_frac)),
               "methods": {}}
        for m in methods:
            vhat = np.asarray(rec[m]["vhat"])
            ang = angle_deg(vhat, vb_arr)                       # latent-readout angle vs target
            # achieved fraction of the commanded velocity change along dv (readout space)
            dvec = vb_arr - va_arr
            achieved = ((vhat - va_arr) * dvec).sum(1) / ((dvec * dvec).sum(1) + 1e-12)
            sr = np.linalg.norm(vhat, axis=1) / (np.linalg.norm(vb_arr, axis=1) + 1e-12)
            det["methods"][m] = {
                "latent_readout_angle_deg": round(float(np.mean(ang)), 2),
                "latent_readout_speed_ratio": round(float(np.median(sr)), 3),
                "achieved_frac_of_command": round(float(np.median(achieved)), 3),
                "offmanifold_resid_frac": round(float(np.mean(rec[m]["resid"])), 4),
            }
            mm = det["methods"][m]
            print(f"  (B/C) {m:13s} readout_ang={mm['latent_readout_angle_deg']:5}deg "
                  f"achieved={mm['achieved_frac_of_command']:5} "
                  f"resid={mm['offmanifold_resid_frac']:6} (real {real_resid:.4f})")
        det["cmd_U8_gain_curve_meanspeed"] = {f"{g:g}": round(float(np.mean(gain_curve[g])), 5)
                                              for g in gains}
        det["target_speed_mean"] = round(float(np.mean(np.linalg.norm(vb_arr, axis=1))), 5)
        print(f"  (D) per-pair Delta energy in shared U8 = {det['delta_U8_energy_frac_mean']:.3f} "
              f"(median {det['delta_U8_energy_frac_median']:.3f})")
        print(f"  (E) cmd_U8 gain->probe speed: {det['cmd_U8_gain_curve_meanspeed']} "
              f"(target {det['target_speed_mean']})")
        summary["layers_detail"][f"L{L}"] = det

    (out / "interp_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[interp] -> {out}/interp_summary.json")


if __name__ == "__main__":
    main()
