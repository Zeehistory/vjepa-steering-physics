

#!/usr/bin/env python
"""Fit the 2nd-order TRANSLATION-FIELD accel operator (Step 2 "Bravo", transfield push).

Physics (ranks share pos0 AND v0): target-B vs reference-A trajectory difference is the exact closed form
``Dx(t) = x_b(t)-x_a(t) = 1/2 (a_b-a_a) t^2``. So the accel edit is a per-frame ball TRANSLATION by Dx(t)
whose latent footprint depends on the ball position ``x_a(t)`` (readable from the reference clip). We fit a
single shared per-token linear map
    dH(t)_slab  ~=  [ Dx(t) || Dx(t) (x) posbasis(x_a(t)) ] @ B     (feature dim 14, target = H*W*D slab)
that expresses a smoothly position-varying translation Jacobian. NO rolling (canon's discrete roll was
lossy). At steer, x_a(t) comes from the reference clip and Dx(t) from the command via the fixed temporal
profile ``gvec[t]`` (Dx(t) = (a_b-a_a) * gvec[t]), so the whole operator is H_b-free.

Saves (into --artifacts_dir): ``cmd_Btransfield_L{L}.npy`` (per-layer map), ``transfield_gvec.npy`` (T,),
``cmd_operator_meta_transfield.json`` (held-out recon cos of the assembled dH). Consumed by
``steer_accel2d.py --features transfield``.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_transfield.py --train_dir .../train/vjepa2_large \
        --test_dir .../test/vjepa2_large --layers 6,12,18,23 --ridge 1.0 --artifacts_dir .../subspace
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

PB = vo.TRANSFIELD_POSBASIS_DIM     # 6
F_POS = 2 + 2 * PB                  # [Dx || Dx (x) posbasis] = 14


def token_feat(dx: np.ndarray, xa: np.ndarray) -> np.ndarray:
    pb = vo.transfield_posbasis(xa)
    return np.concatenate([dx, np.outer(dx, pb).reshape(-1)])  # (14,)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    trs, tes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        trs = {s: trs[s] for s in sorted(trs)[: args.max_scenes]}
        tes = {s: tes[s] for s in sorted(tes)[: args.max_scenes]}
    order = sorted(trs)
    print(f"[transfit] train {len(trs)} scenes, test {len(tes)}; layers={layers}; F_POS={F_POS}", flush=True)

    # ---- pass 0: calibrate temporal profile gvec[t] s.t. Dx(t) ~= (a_b-a_a) * gvec[t] -----------------
    Tguess = None
    gnum = None; gden = 0.0
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]; aa = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"]); T = grid[0]
        if Tguess is None:
            Tguess = T; gnum = np.zeros(T)
        pa = vo.clip_frame_positions(sa, T)
        for b in ranks[1:]:
            sb = tr[trs[s][b]]; ab = vo.clip_acceleration(sb)
            da = ab - aa
            dx = vo.clip_frame_positions(sb, T) - pa      # (T,2) = da * gvec[t]
            gnum += dx @ da                               # sum_t-wise dot with da
            gden += float(da @ da)
    gvec = gnum / (gden + 1e-12)                          # (T,)
    np.save(art / "transfield_gvec.npy", gvec.astype(np.float32))
    print(f"[transfit] gvec (1/2 t^2 profile) = {np.round(gvec, 5).tolist()}", flush=True)

    # ---- pass 1: fit per-token translation map B_pos[L] (feature uses GT Dx = pos_b - pos_a) ----------
    m = {L: None for L in layers}
    n = 0
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]
        grid = tuple(int(x) for x in sa["grid"]); T, H, W = grid
        pa = vo.clip_frame_positions(sa, T)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[trs[s][b]]
            dx_t = vo.clip_frame_positions(sb, T) - pa
            for L in layers:
                D = Ha[L].size // (T * H * W); slab_dim = H * W * D
                if m[L] is None:
                    m[L] = vo.LinearLS(F_POS, slab_dim, args.ridge)
                slabs = (vo.layer_flat(sb["layers"][L]) - Ha[L]).reshape(T, slab_dim)
                for t in range(T):
                    m[L].add(token_feat(dx_t[t], pa[t]).reshape(1, F_POS), slabs[t].reshape(1, slab_dim))
        del Ha; gc.collect()
        n += 1
        if n % 50 == 0:
            print(f"[transfit]   fit {n}/{len(trs)}", flush=True)
    B = {}
    for L in layers:
        B[L] = m[L].solve().astype(np.float32)
        np.save(art / f"cmd_Btransfield_L{L}.npy", B[L])
    del m; gc.collect()
    print("[transfit] B_transfield saved", flush=True)

    # ---- held-out gate: recon cos of assembled dH, using STEER-FAITHFUL Dx = (ab-aa)*gvec -------------
    gate = {L: [] for L in layers}
    for s in sorted(tes):
        ranks = sorted(tes[s]); sa = te[tes[s][ranks[0]]]; aa = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"]); T, H, W = grid
        pa = vo.clip_frame_positions(sa, T)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[tes[s][b]]; ab = vo.clip_acceleration(sb)
            da = ab - aa
            dx_t = np.outer(gvec, da)                     # (T,2) steer-faithful displacement
            for L in layers:
                D = Ha[L].size // (T * H * W); slab_dim = H * W * D
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                pred = np.empty((T, slab_dim), dtype=np.float64)
                for t in range(T):
                    pred[t] = token_feat(dx_t[t], pa[t]) @ B[L]
                gate[L].append(vo.cosine(pred.reshape(-1), dH))
        del Ha; gc.collect()

    summary = {"layers": layers, "ridge": args.ridge, "F_POS": F_POS, "quantity": "accel",
               "features": "transfield", "n_train": len(trs), "n_test": len(tes),
               "gvec": [round(float(x), 6) for x in gvec],
               "heldout_recon_cos": {str(L): round(float(np.mean(v)), 4) for L, v in gate.items()}}
    (art / "cmd_operator_meta_transfield.json").write_text(json.dumps(summary, indent=2))
    print("[transfit] HELD-OUT recon cos (steer-faithful Dx=(ab-aa)*gvec):", flush=True)
    for L in layers:
        print(f"  L{L}: {summary['heldout_recon_cos'][str(L)]:.3f}", flush=True)
    print(f"[transfit] saved -> {art}", flush=True)


if __name__ == "__main__":
    main()
