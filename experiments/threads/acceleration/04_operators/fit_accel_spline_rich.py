

#!/usr/bin/env python
"""Spline-in-time accel operators with RICHER CONDITIONING -- attacking the synthesis gap, not the basis.

The 2026-07-27 K-sweep settled which half of the problem is binding. Its held-out latent gate
(``spline/spline_operator_meta.json``) reads, at L18:

    K:      1      2      3      4      6      8
    pred_cos   0.462  0.630  0.642  0.646  0.655  0.660     <- what the operator ACHIEVES
    proj_cos   0.729  0.892  0.924  0.941  0.971  1.000     <- what K knots can EXPRESS

By K=8 the basis is exact (proj_cos == 1 by construction) and the operator still only reaches 0.66.
The missing 0.34 is not a representational limit -- it is that a 13-dim function of (a_a, a_b) cannot
predict the edit. Adding knots cannot fix that, which is exactly why K>=2 is flat in decode. So this
script holds the spline basis fixed and widens the MAP's conditioning instead:

    cmd        phi(a_a, a_b)                       (13)   the existing operator, refit here as the control
    quad       command_features_quad               (P_q)  second-order in the command alone
    appc       [phi || appc(H_a)]                   (13+Ka) anchor-appearance conditioning
    bilinear   [phi || appc(H_a) || phi (x) appc]   (13+Ka+13*Ka) lets the edit's DIRECTION depend on
                                                    which scene it is applied to, not just on the command

Every variant is legal command-only steering: ``appc`` reads the ANCHOR latent H_a, which is available
at inference, and never H_b. ``appc`` reuses the appearance basis fit by
``fit_command_operators_accel_appc.py`` (``appc_mean_L*.npy`` / ``appc_basis_L*.npy``), so the appearance
summary is byte-identical to the one the appc operator family already used -- any difference is
attributable to putting it in the spline basis, not to a new feature definition.

Same streaming ridge, same 7-pairs-per-scene enrichment, same held-out gate as
``fit_accel_spline_operator.py``; only the feature map changes. Writes
``rich_{tag}_W_K{K}_L{L}.npy`` + ``rich_operator_meta.json``, consumed by ``steer_accel_spline.py``.

    python experiments/threads/acceleration/04_operators/fit_accel_spline_rich.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --appc_dir .../subspace --layers 6,12,18,23 --knots 2,3,8 --features appc \
        --output_dir .../spline_rich
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

from src.analysis import spline_ops as sp
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset


def pooled(sample_layer) -> np.ndarray:
    """Appearance summary of one layer's latent = mean over tokens -> (D,). Matches appc's definition."""
    return np.asarray(sample_layer, dtype=np.float64).mean(axis=0)


class FeatureMap:
    """Per-layer feature builder. ``appc``/``bilinear`` are layer-dependent; ``cmd``/``quad`` are not."""

    def __init__(self, kind: str, appc_dir: Path | None, layers: list[int], ka: int):
        self.kind = kind
        self.ka = ka
        self.mean = self.basis = None
        if kind in ("appc", "bilinear"):
            if appc_dir is None:
                raise SystemExit(f"--features {kind} needs --appc_dir (appc_mean_L*/appc_basis_L*)")
            self.mean = {L: np.load(appc_dir / f"appc_mean_L{L}.npy").astype(np.float64) for L in layers}
            self.basis = {L: np.load(appc_dir / f"appc_basis_L{L}.npy").astype(np.float64)[:ka]
                          for L in layers}

    def base(self, aa: np.ndarray, ab: np.ndarray) -> np.ndarray:
        return vo.command_features_quad(aa, ab) if self.kind == "quad" else vo.command_features(aa, ab)

    def __call__(self, aa, ab, sa_layer, L: int) -> np.ndarray:
        phi = self.base(aa, ab)
        if self.kind in ("cmd", "quad"):
            return phi
        ca = self.basis[L] @ (pooled(sa_layer) - self.mean[L])          # (Ka,)
        if self.kind == "appc":
            return np.concatenate([phi, ca])
        return np.concatenate([phi, ca, np.outer(phi, ca).reshape(-1)])  # bilinear

    def dim(self, L: int) -> int:
        p = vo.COMMAND_FEATURE_DIM_QUAD if self.kind == "quad" else vo.COMMAND_FEATURE_DIM
        if self.kind == "cmd" or self.kind == "quad":
            return p
        return p + self.ka + (p * self.ka if self.kind == "bilinear" else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--appc_dir", default="", help="dir holding appc_mean_L*.npy / appc_basis_L*.npy")
    p.add_argument("--features", choices=["cmd", "quad", "appc", "bilinear"], default="appc")
    p.add_argument("--ka", type=int, default=16, help="appearance PCs read from pooled H_a")
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--knots", default="1,2,3,8")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0, help="0 = all; smoke-test knob")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    Ks = [int(x) for x in args.knots.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    fm = FeatureMap(args.features, Path(args.appc_dir) if args.appc_dir else None, layers, args.ka)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}

    grid = tuple(int(x) for x in tr[tr_scenes[sorted(tr_scenes)[0]][0]]["grid"])
    T, D = grid[0], int(tr.records[0]["hidden_dim"])
    B = {K: sp.spline_basis(T, K, args.degree) for K in Ks}
    P = {L: fm.dim(L) for L in layers}
    print(f"[spline-rich] features={args.features} P={P[layers[0]]} ka={args.ka}; "
          f"train {len(tr_scenes)} scenes, test {len(te_scenes)}; K={Ks}; grid={grid} D={D}", flush=True)

    ls = {K: {L: vo.LinearLS(P[L], K * D, args.ridge) for L in layers} for K in Ks}
    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s])
        sa = tr[tr_scenes[s][ranks[0]]]
        aa = vo.clip_acceleration(sa)
        Ra = {L: sp.temporal_profile(vo.layer_flat(sa["layers"][L]), grid) for L in layers}
        feats = {L: fm(aa, aa, sa["layers"][L], L) for L in layers}   # anchor part is target-independent
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            ab = vo.clip_acceleration(sb)
            for L in layers:
                phi = fm(aa, ab, sa["layers"][L], L).reshape(1, P[L])
                dR = sp.temporal_profile(vo.layer_flat(sb["layers"][L]), grid) - Ra[L]
                for K in Ks:
                    ls[K][L].add(phi, sp.project_profile(dR, B[K]).reshape(1, -1))
            del sb
        del Ra, sa, feats
        gc.collect()
        n += 1
        if n % 50 == 0:
            print(f"[spline-rich]   {n}/{len(tr_scenes)} scenes", flush=True)

    W = {K: {L: ls[K][L].solve() for L in layers} for K in Ks}
    del ls; gc.collect()
    tag = args.features
    for K in Ks:
        for L in layers:
            np.save(out / f"rich_{tag}_W_K{K}_L{L}.npy", W[K][L].astype(np.float32))

    # ------------------------------------------------------------------ held-out latent gate
    gate = {K: {L: [] for L in layers} for K in Ks}
    m = 0
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s])
        sa = te[te_scenes[s][ranks[0]]]
        aa = vo.clip_acceleration(sa)
        Ra = {L: sp.temporal_profile(vo.layer_flat(sa["layers"][L]), grid) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]
            ab = vo.clip_acceleration(sb)
            for L in layers:
                phi = fm(aa, ab, sa["layers"][L], L)
                dR = sp.temporal_profile(vo.layer_flat(sb["layers"][L]), grid) - Ra[L]
                for K in Ks:
                    pred = (phi @ W[K][L]).reshape(K, D)
                    gate[K][L].append(vo.cosine(sp.reconstruct_profile(pred, B[K]).reshape(-1),
                                                dR.reshape(-1)))
            del sb
        del Ra, sa
        gc.collect()
        m += 1
        if m % 25 == 0:
            print(f"[spline-rich]   gated {m}/{len(te_scenes)} scenes", flush=True)

    summary = {
        "train_dir": args.train_dir, "test_dir": args.test_dir, "features": args.features,
        "ka": args.ka, "layers": layers, "knots": Ks, "degree": args.degree, "ridge": args.ridge,
        "grid": list(grid), "T": T, "D": D, "feature_dim": {str(L): P[L] for L in layers},
        "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes), "pairs_per_scene": 7,
        "per_knot": {str(K): {str(L): {"pred_cos": round(float(np.mean(gate[K][L])), 4)}
                              for L in layers} for K in Ks},
        "artifacts": f"rich_{tag}_W_K{{K}}_L{{L}}.npy",
        "note": "compare pred_cos against spline_operator_meta.json's cmd-only pred_cos at the same K. "
                "The latent gate is diagnostic ONLY -- prior arms improved it while decode did not move.",
    }
    (out / f"rich_{tag}_operator_meta.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[spline-rich] held-out pred_cos ({args.features}):")
    print(f"  {'K':>3}  " + "  ".join(f"L{L}" for L in layers))
    for K in Ks:
        print(f"  {K:>3}  " + "  ".join(f"{np.mean(gate[K][L]):.3f}" for L in layers))
    print(f"[spline-rich] -> {out}/rich_{tag}_operator_meta.json")


if __name__ == "__main__":
    main()
