

#!/usr/bin/env python
"""Is SPIN in the V-JEPA latent at all? A read-only probe, and the question everything else rests on.

Three independent measurements have now come back saying the same thing about spin, and none of them
can distinguish "spin is present but hard to use" from "spin was never encoded":

  * the spin OPERATOR reaches alignment ~0.10 with the true spin displacement;
  * the CEILING for any command-only map is ~0.00-0.03, i.e. two scenes given near-identical spin
    commands produce near-orthogonal latent displacements;
  * the DECODER renders velocity at 1.4 deg median heading error but smears the marker to
    ``mass = 0.000``, so decoded omega correlates -0.006 with truth.

Those are three symptoms of one possible cause. If V-JEPA simply does not encode this marker's phase,
then the operator cannot steer it, the ceiling must be zero, and the decoder cannot render it -- all
three follow, and no amount of refitting or retraining changes any of them. **You cannot steer what is
not represented.** The alternative -- spin IS encoded and three separate methods each failed to exploit
it -- is a very different situation calling for very different work.

A linear probe settles it, and it is the cleanest possible test because it asks nothing of an operator
or a decoder: fit ``omega ~ H`` by ridge on train clips, score correlation and R^2 on held-out clips.

Controls that make the answer interpretable:

  * ``speed`` probed the same way. Velocity is known to be strongly encoded (the decoder renders it at
    1.4 deg), so a high speed score confirms the probe itself works on this data and this layer. Without
    it, a low omega score could just mean the probe is broken.
  * ``|omega|`` as well as signed ``omega``. The dataset balances CW against CCW within every scene, so
    a representation that encodes rotation RATE but discards handedness would score ~0 on signed omega
    while scoring well on the magnitude. That is a materially different finding from "no spin at all"
    and the two must not be conflated.
  * a SHUFFLED-label run, which is the honest floor for R^2 at this sample size and dimensionality.
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

import sys
from pathlib import Path as _P

import numpy as np

from src.analysis import spin_ops as so
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def _ridge_score(X: np.ndarray, y: np.ndarray, n_train: int, alphas, rng) -> dict:
    """Held-out correlation and R^2 for ridge ``y ~ X``, with the best alpha chosen on the TRAIN half.

    Alpha is selected inside the training split only. Picking it on the test split would let the probe
    tune itself to the answer, which on a high-dimensional latent is enough to manufacture signal.
    """
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()

    # Inner split of the TRAIN half for alpha selection.
    k = max(1, int(0.75 * len(Xtr)))
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        G = Xtr[:k].T @ Xtr[:k] + a * np.eye(Xtr.shape[1])
        w = np.linalg.solve(G, Xtr[:k].T @ (ytr[:k] - ytr[:k].mean()))
        p = Xtr[k:] @ w + ytr[:k].mean()
        if len(p) < 2:
            continue
        c = float(np.corrcoef(p, ytr[k:])[0, 1]) if np.std(p) > 0 else 0.0
        if np.isfinite(c) and c > best:
            best, best_a = c, a

    G = Xtr.T @ Xtr + best_a * np.eye(Xtr.shape[1])
    w = np.linalg.solve(G, Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    corr = float(np.corrcoef(pred, yte)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"corr": corr if np.isfinite(corr) else 0.0,
            "r2": 1.0 - ss_res / max(ss_tot, 1e-12), "alpha": float(best_a)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--num_clips", type=int, default=512)
    p.add_argument("--pca_dim", type=int, default=256)
    p.add_argument("--max_cached_shards", type=int, default=1)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    rng = np.random.default_rng(0)
    out = {}

    for L in layers:
        ds = LatentDataset(args.test_dir, layers=[L], max_cached_shards=args.max_cached_shards)
        n = min(args.num_clips, len(ds))
        feats, om, sp = [], [], []
        for i in range(n):
            s = ds[i]
            # Mean-pool over tokens. The probe asks whether the quantity is LINEARLY present in the
            # representation, not whether some elaborate readout can dig it out, so the simplest
            # permutation-invariant summary is the right feature -- and it keeps the ridge well posed
            # at 512 clips instead of fitting 500k dimensions from 512 samples.
            x = np.asarray(s["layers"][L], dtype=np.float64)
            feats.append(x.reshape(x.shape[0], -1).mean(axis=0) if x.ndim > 1 else x.ravel())
            om.append(float(so.clip_spin(s)))
            sp.append(float(np.linalg.norm(vo.clip_velocity(s))))
        X = np.stack(feats)
        om, sp = np.asarray(om), np.asarray(sp)

        # Random projection to keep the ridge conditioned; preserves linear structure (JL).
        if X.shape[1] > args.pca_dim:
            R = rng.standard_normal((X.shape[1], args.pca_dim)) / np.sqrt(X.shape[1])
            X = X @ R

        idx = rng.permutation(len(X))
        X, om, sp = X[idx], om[idx], sp[idx]
        n_tr = int(0.7 * len(X))
        alphas = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

        res = {
            "n_clips": int(len(X)), "feat_dim": int(X.shape[1]),
            "omega_signed": _ridge_score(X, om, n_tr, alphas, rng),
            "omega_abs": _ridge_score(X, np.abs(om), n_tr, alphas, rng),
            "speed_control": _ridge_score(X, sp, n_tr, alphas, rng),
            "omega_shuffled_floor": _ridge_score(X, rng.permutation(om), n_tr, alphas, rng),
        }
        out[f"L{L}"] = res
        print(f"[probe] L{L} (n={res['n_clips']}, d={res['feat_dim']})", flush=True)
        for k in ("omega_signed", "omega_abs", "speed_control", "omega_shuffled_floor"):
            print(f"    {k:22s} corr {res[k]['corr']:+.3f}  R2 {res[k]['r2']:+.3f}", flush=True)
        del ds

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("READ: if speed_control is high and BOTH omega rows sit at the shuffled floor, spin is not "
          "linearly encoded -- and no operator or decoder can recover what is not there.")


if __name__ == "__main__":
    main()
