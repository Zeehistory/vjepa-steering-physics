#!/usr/bin/env python
"""READ probe: is angular velocity omega linearly readable from the VJEPA latent (independent of any decoder)?

The decoder renders the rotating bar as a faint smear (decode(true H_b) doesn't track GT omega). This probe
asks the upstream question: does the LATENT itself carry omega? Mean-pool each clip's tokens per layer ->
feature; ridge-regress the signed omega on TRAIN, evaluate on the held-out TEST split. High R²/rho => omega
is readable (ceiling is the decoder's rendering); low => omega is barely encoded (a deeper cap). This is the
'readable vs writable' question the paper hinges on, now for angular velocity.
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
import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config


def pooled(ds, scenes, layers, max_clips=None):
    X = {L: [] for L in layers}
    y = []
    ids = sorted(scenes)
    count = 0
    for s in ids:
        for rank, idx in scenes[s].items():
            smp = ds[idx]
            y.append(float(vo.clip_angvel(smp)[0]))
            for L in layers:
                f = vo.layer_flat(smp["layers"][L])          # (Ltok*D,) or (Ltok,D)?
                arr = np.asarray(smp["layers"][L])
                # mean-pool over token axis -> (D,)
                if arr.ndim == 2:
                    X[L].append(arr.mean(axis=0))
                else:
                    X[L].append(arr.reshape(arr.shape[0], -1).mean(axis=0))
            count += 1
        if max_clips and count >= max_clips:
            break
    return {L: np.stack(X[L]) for L in layers}, np.asarray(y)


def ridge_fit(Xtr, ytr, lam=1.0):
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--max_train_clips", type=int, default=2000)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    sc_tr, sc_te = vo.group_scenes(tr), vo.group_scenes(te)
    print(f"[read] layers={layers} train_scenes={len(sc_tr)} test_scenes={len(sc_te)}")

    Xtr, ytr = pooled(tr, sc_tr, layers, max_clips=args.max_train_clips)
    Xte, yte = pooled(te, sc_te, layers)
    print(f"[read] pooled train={len(ytr)} clips, test={len(yte)} clips")

    # standardize per-feature on train
    for L in layers:
        mu, sd = Xtr[L].mean(0), Xtr[L].std(0) + 1e-6
        Xtr[L] = (Xtr[L] - mu) / sd
        Xte[L] = (Xte[L] - mu) / sd
        w = ridge_fit(Xtr[L], ytr, lam=10.0)
        pred = Xte[L] @ w
        ss_res = float(((yte - pred) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        rho = float(np.corrcoef(yte, pred)[0, 1])
        sign = float(np.mean(np.sign(pred) == np.sign(yte)))
        print(f"  L{L:2d}: test R²={r2:+.3f} rho={rho:+.3f} sign_acc={sign:.3f}")

    # all-layers concat
    Xtr_c = np.concatenate([Xtr[L] for L in layers], 1)
    Xte_c = np.concatenate([Xte[L] for L in layers], 1)
    w = ridge_fit(Xtr_c, ytr, lam=10.0)
    pred = Xte_c @ w
    r2 = 1 - float(((yte - pred) ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum())
    rho = float(np.corrcoef(yte, pred)[0, 1]); sign = float(np.mean(np.sign(pred) == np.sign(yte)))
    print(f"  ALL: test R²={r2:+.3f} rho={rho:+.3f} sign_acc={sign:.3f}")


if __name__ == "__main__":
    main()
