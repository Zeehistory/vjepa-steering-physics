

#!/usr/bin/env python
"""Is per-frame marker PHASE theta(t) in the latent? The question that decides whether spin is renderable.

``spin_probe.py`` established that OMEGA -- a clip-level rotation RATE -- is linearly decodable (signed
omega corr 0.734 at L23, far above a shuffled floor). That is necessary but NOT sufficient for what the
decoder has to do. To draw a rotating marker the decoder needs theta at EACH frame; a clip constant
tells it how fast to spin, not where the marker is right now. A representation could encode rate
perfectly while discarding phase -- temporally pooled features are exactly the kind of thing that would
-- and every decoder attempt would then fail no matter how it was trained.

That distinction is not academic here. Six decoder runs now render the marker as a STATIC smudge
(readable in 100% of clips, omega_corr ~0), and adding correctly-projected phase supervision degrades
velocity without moving omega_corr at all -- including when warm-started from a decoder that already
draws a marker with the detector's rotation centre on the ball. Those are all symptoms consistent with
one cause: theta is not there to be rendered.

So this probes theta directly, per TEMPORAL TOKEN rather than per clip:
  * features: layer tokens reshaped to (T,H,W,D), pooled over space -> (T,D), one row per temporal token;
  * target: ``(cos theta, sin theta)`` averaged over the frames that token covers (theta is circular, so
    it is regressed as a unit vector and scored as an angle, never as a raw scalar);
  * scored by MEDIAN ANGULAR ERROR on held-out SCENES, against two floors: a shuffled-target control,
    and predicting the per-clip mean direction (which any rate-only representation could achieve).

The second floor is the important one. Beating "shuffled" only shows the tokens know something; beating
"clip mean" is what shows they know PHASE as opposed to a constant offset per clip.

    PYTHONPATH=. python experiments/threads/restitution-spin/06_probes/theta_probe.py --test_dir ... --out ...
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

from src.encoders.feature_extractor import LatentDataset


def _ang_err_deg(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Angle between predicted and target unit vectors, degrees."""
    p = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    t = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip((p * t).sum(axis=1), -1.0, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="6,12,18,23")
    ap.add_argument("--num_clips", type=int, default=384)
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--alphas", default="1e-2,1e0,1e2,1e4")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    ds = LatentDataset(args.test_dir, layers=layers, max_cached_shards=1)
    keys = list(ds[0]["state_keys"])
    if "obj0_theta" not in keys:
        raise SystemExit(f"obj0_theta not in state_keys: {keys}")
    ti = keys.index("obj0_theta")
    n = min(args.num_clips, len(ds))
    alphas = [float(a) for a in args.alphas.split(",")]

    # ---- collect per-temporal-token features and targets ------------------------------------------
    feats = {L: [] for L in layers}
    tgts, clip_id = [], []
    for i in range(n):
        s = ds[i]
        st = np.asarray(s["state"], dtype=np.float64)
        if st.ndim != 2:
            continue
        theta = st[:, ti]                                   # (F,)
        F = theta.shape[0]
        T, H, W = (int(x) for x in s["grid"])
        if F % T != 0:
            continue
        tub = F // T                                        # frames per temporal token
        th = theta.reshape(T, tub)
        cs = np.stack([np.cos(th).mean(axis=1), np.sin(th).mean(axis=1)], axis=1)   # (T,2)
        cs = cs / (np.linalg.norm(cs, axis=1, keepdims=True) + 1e-12)
        tgts.append(cs)
        clip_id.append(np.full(T, i))
        for L in layers:
            x = np.asarray(s["layers"][L], dtype=np.float32).reshape(T, H * W, -1).mean(axis=1)  # (T,D)
            feats[L].append(x)
        if (i + 1) % 64 == 0:
            print(f"[theta] {i + 1}/{n}", flush=True)

    Y = np.concatenate(tgts, axis=0)                        # (N,2)
    cid = np.concatenate(clip_id, axis=0)
    clips = np.unique(cid)
    rng = np.random.default_rng(0)
    rng.shuffle(clips)
    tr_clips = set(clips[: int(0.7 * len(clips))].tolist())
    tr = np.array([c in tr_clips for c in cid])
    te = ~tr
    print(f"[theta] {Y.shape[0]} tokens from {len(clips)} clips; train {tr.sum()} / test {te.sum()}",
          flush=True)

    out = {"n_tokens": int(Y.shape[0]), "n_clips": int(len(clips)), "layers": {}}

    # Floor 1: predict each clip's MEAN direction -> what a rate-only / phase-free code can do.
    clip_mean = {}
    for c in clips:
        m = (cid == c) & tr
        clip_mean[c] = Y[m].mean(axis=0) if m.sum() else np.array([1.0, 0.0])
    base_pred = np.stack([clip_mean[c] for c in cid[te]], axis=0)
    out["floor_clip_mean_median_deg"] = round(float(np.median(_ang_err_deg(base_pred, Y[te]))), 2)

    for L in layers:
        X = np.concatenate(feats[L], axis=0).astype(np.float64)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xs = (X - mu) / sd
        # PCA on train tokens for conditioning
        U, S, Vt = np.linalg.svd(Xs[tr] - Xs[tr].mean(0), full_matrices=False)
        P = Vt[: args.pca_dim].T
        Z = Xs @ P
        Ztr, Ytr = Z[tr], Y[tr]
        G = Ztr.T @ Ztr
        best, bestv = None, 1e9
        for a in alphas:
            Wt = np.linalg.solve(G + a * np.eye(G.shape[0]), Ztr.T @ Ytr)
            e = float(np.median(_ang_err_deg(Z[te] @ Wt, Y[te])))
            if e < bestv:
                best, bestv = a, e
        Wt = np.linalg.solve(G + best * np.eye(G.shape[0]), Ztr.T @ Ytr)
        err = _ang_err_deg(Z[te] @ Wt, Y[te])
        # Floor 2: shuffled targets
        Ysh = Ytr[rng.permutation(Ytr.shape[0])]
        Wsh = np.linalg.solve(G + best * np.eye(G.shape[0]), Ztr.T @ Ysh)
        esh = _ang_err_deg(Z[te] @ Wsh, Y[te])
        out["layers"][str(L)] = {
            "median_ang_err_deg": round(float(np.median(err)), 2),
            "shuffled_floor_deg": round(float(np.median(esh)), 2),
            "alpha": best,
        }
        print(f"[theta] L{L}: median {np.median(err):.1f} deg (shuffled {np.median(esh):.1f}, "
              f"clip-mean {out['floor_clip_mean_median_deg']:.1f})", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\n# Per-frame marker phase theta from latents (held-out scenes)\n")
    print("| layer | median angular err | shuffled floor |")
    print("|---|---|---|")
    for L in layers:
        d = out["layers"][str(L)]
        print(f"| L{L} | {d['median_ang_err_deg']:.1f} deg | {d['shuffled_floor_deg']:.1f} deg |")
    print(f"\nclip-mean floor (phase-free): {out['floor_clip_mean_median_deg']:.1f} deg; chance = 90 deg")
    print("READ: to render a rotating marker the decoder needs theta PER FRAME. Beating the shuffled")
    print("floor only shows the tokens carry something; beating the CLIP-MEAN floor is what shows they")
    print("carry phase rather than a per-clip constant. If theta sits at the clip-mean floor, no decoder")
    print("can draw a marker in the right place and the spin half is not renderable -- a property of the")
    print("representation, not of any training run.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
