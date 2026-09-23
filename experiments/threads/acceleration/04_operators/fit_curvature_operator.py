

#!/usr/bin/env python
"""EXP 3 fit (user 2026-07-07): command -> latent CURVATURE-CHANGE subspace operator.

Fits the operator the second-order integration steer needs. Ranks in a scene share pos0, v0, appearance
and differ only in acceleration, so the target-vs-reference latent difference dH_t = Z^b_t - Z^a_t is a
pure acceleration edit whose second temporal difference

    dA_s = dH_{s+1} - 2 dH_s + dH_{s-1}        (s = 1..T-2, the per-step CURVATURE CHANGE)

is ~constant in s (= the acceleration change spread uniformly over time). We build a low-rank subspace
U_curv from PCA of all train dA_s (pooled over scenes AND s -- the constancy is what lets us pool), then
regress the 13-d command features -> U_curv coordinates with a single SHARED map (all s pooled, high SNR).

At steer time the operator predicts one curvature direction dA_pred(command) and the integration
    e_t = sum_{s=1}^{t-1}(t-s) dA_pred = dA_pred * (t-1)t/2
rebuilds the edit with the physically-correct t^2 temporal profile imposed for free -- the profile every
prior per-t / global operator averaged away. Saves global_basis_curv_L*.npy (U) + cmd_Wu_curv_L*.npy (map)
and also cmd_Brich_L*.npy is REUSED from the existing subspace dir (not written here).

    python experiments/threads/acceleration/04_operators/fit_curvature_operator.py --config configs/train/moving_ball_scene_decoder.yaml \
        --train_dir .../train/vjepa2_large --artifacts_dir .../subspace --save_k 32
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
from pathlib import Path

import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config


def _evict(ds, keep=2):
    if len(ds._shard_cache) > keep:
        ds._shard_cache.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--save_k", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config, [])
    layers = list(cfg.encoder.layers)
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    ds = LatentDataset(args.train_dir, layers=layers)
    grid = tuple(int(x) for x in ds[0]["grid"])
    T = grid[0]
    Dslab = int(ds[0]["layers"][layers[0]].shape[-1]) * grid[1] * grid[2]
    print(f"[fit-curv] layers={layers} T={T} Dslab={Dslab} save_k={args.save_k}", flush=True)

    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.max_scenes or None]
    # collect dA_s (float32) per layer + command features
    dA_store = {L: [] for L in layers}   # each entry (D,)
    cmd_store = []                        # per (scene, s) -> repeated command
    s_index = list(range(1, T - 1))
    for n, s in enumerate(sids):
        ranks = sorted(scenes[s]); ia, ib = scenes[s][ranks[0]], scenes[s][ranks[-1]]
        sa, sb = ds[ia], ds[ib]
        aa, ab = vo.clip_acceleration(sa), vo.clip_acceleration(sb)
        cmd = vo.command_features(aa, ab).astype(np.float32)
        for L in layers:
            Za = np.asarray(sa["layers"][L], dtype=np.float32).reshape(T, Dslab)
            Zb = np.asarray(sb["layers"][L], dtype=np.float32).reshape(T, Dslab)
            dH = Zb - Za
            for si in s_index:
                dA_store[L].append(dH[si + 1] - 2 * dH[si] + dH[si - 1])
        for _ in s_index:
            cmd_store.append(cmd)
        if (n + 1) % 64 == 0:
            _evict(ds); print(f"  [collect] {n+1}/{len(sids)}", flush=True)

    Cmd = np.stack(cmd_store).astype(np.float64)          # (M, 13)
    print(f"[fit-curv] collected M={Cmd.shape[0]} (scene x s) rows; building subspaces", flush=True)
    for L in layers:
        X = np.stack(dA_store[L]).astype(np.float64)      # (M, D)
        dA_store[L] = None
        # PCA subspace of the curvature change (Gram trick), top save_k
        U, _ = vo.pca_gram(X, k=args.save_k)              # (k, D)
        U = vo.orthonormalize(U)
        # shared command -> U coords ridge
        Coords = X @ U.T                                  # (M, k)
        A = Cmd.T @ Cmd + args.ridge * np.eye(Cmd.shape[1])
        Wu = np.linalg.solve(A, Cmd.T @ Coords)           # (13, k)
        # report train fit quality: coord R2 and reconstruction cos
        Chat = Cmd @ Wu
        ss_res = ((Chat - Coords) ** 2).sum(); ss_tot = ((Coords - Coords.mean(0)) ** 2).sum() + 1e-30
        recon = (Chat @ U)                                # (M, D) predicted dA
        cos = ((recon * X).sum(1) / (np.linalg.norm(recon, axis=1) * np.linalg.norm(X, axis=1) + 1e-30))
        np.save(art / f"global_basis_curv_L{L}.npy", U.astype(np.float32))
        np.save(art / f"cmd_Wu_curv_L{L}.npy", Wu.astype(np.float32))
        print(f"  L{L}: U {U.shape} | train coord_R2={1 - ss_res/ss_tot:.3f} "
              f"dA_recon_cos(train)={cos.mean():.3f}", flush=True)
        del X, Coords, recon
    print(f"[fit-curv] saved global_basis_curv_L*/cmd_Wu_curv_L* -> {art}", flush=True)


if __name__ == "__main__":
    main()
