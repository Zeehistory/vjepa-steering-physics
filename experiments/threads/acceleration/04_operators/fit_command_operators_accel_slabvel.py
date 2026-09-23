

#!/usr/bin/env python
"""TEMPORAL-COMPOSITION accel operator: acceleration as a sequence of per-time-token VELOCITY edits.

Motivation: acceleration is second order = velocity + a temporal dimension. Between two ranks that share
pos0 AND v0, the instantaneous velocity difference at latent time token ``t`` is exactly

    Delta v(t) = v_b(t) - v_a(t) = (a_b - a_a) * tau_t        (tau_t = mean frame index of the tubelet)

so an acceleration edit is a *temporally composed* velocity edit: apply the velocity operator per token,
driven by the linearly-growing ``Delta v(t)``. This script LEARNS that per-token velocity operator on the
accel latents (self-contained -- the velocity latents were reclaimed, and the velocity decoder differs, so
we refit the same machinery here rather than transfer a foreign-decoder operator):

  * Build a per-SLAB acceleration/velocity subspace ``U_slab`` (PCA of per-(layer,token) latent differences
    ``dH[t] = H_b[slab t] - H_a[slab t]``, slab dim = Hp*Wp*D). One subspace shared across all T tokens.
  * Fit ONE shared streaming ridge ``W_slab: command_features(v_a(t), v_b(t)) (13) -> U_slab coords`` over
    every (scene-pair, token) example. The per-token instantaneous velocities are read from the packed
    ``obj0_vel`` columns (``velocity_ops.clip_frame_velocities``), so at steer time we reconstruct them from
    the reference clip's v_a(t) + the command via ``v_b(t) = v_a(t) + (a_b - a_a) * tau_t`` -- NO H_b.

At steer (``steer_accel2d.py --features slabvel``) the full edit is assembled token-by-token:
``edit[slab t] = (command_features(v_a(t), v_b(t)) @ W_slab) @ U_slab``. Because the command grows with t,
late tokens (where the ball has moved farther, curvature strongest) get the larger edit automatically --
the physics profile is explicit, not averaged into a single global blob like the other operators.

Saves ``slabvel_basis_L*.npy`` (KU, slabdim) + ``cmd_Wslabvel_L*.npy`` (13, KU) + meta into the accel
artifacts dir. Held-out TEST latent gate: cos(reconstructed slab edit, true slab dH). Decode is decisive.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_slabvel.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --layers 6,12,18,23 --artifacts_dir .../subspace --ku 32 --max_slab 8000
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


def slab_deltas(sa_layer, sb_layer, grid):
    """Per-token slab differences: returns (T, slabdim) with slabdim = Hp*Wp*D."""
    T, H, W = grid
    a = vo.layer_flat(sa_layer); b = vo.layer_flat(sb_layer)
    slab = H * W * (a.size // (T * H * W))
    return (b - a).reshape(T, slab)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ku", type=int, default=32, help="per-slab subspace rank")
    p.add_argument("--max_slab", type=int, default=8000, help="cap on slab-delta samples used for PCA + fit")
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    KU = int(args.ku)
    P = vo.COMMAND_FEATURE_DIM
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
    print(f"[slabvel] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; "
          f"KU={KU} max_slab={args.max_slab}", flush=True)

    # ---- single pass: collect capped (slab-dH, phi) examples per layer -------------------------------
    slab_buf = {L: [] for L in layers}
    phi_buf = []                       # shared across layers (same token order)
    full = False
    for s in sorted(tr_scenes):
        if full:
            break
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]
        aa = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"])
        T = grid[0]
        va_t = vo.clip_frame_velocities(sa, T)                 # (T,2) reference per-token velocity
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]
            ab = vo.clip_acceleration(sb)
            vb_t = vo.clip_frame_velocities(sb, T)             # (T,2) target per-token velocity (from GT)
            phis = np.stack([vo.command_features(va_t[t], vb_t[t]) for t in range(T)], 0)  # (T,P)
            for L in layers:
                sd = slab_deltas(sa["layers"][L], sb["layers"][L], grid).astype(np.float32)  # (T,slab)
                slab_buf[L].append(sd)
            phi_buf.append(phis)
            if sum(x.shape[0] for x in slab_buf[layers[0]]) >= args.max_slab:
                full = True
                break
        del sa
        if hasattr(tr, "_shard_cache") and len(tr._shard_cache) > 2:
            tr._shard_cache.clear()
        gc.collect()

    phi_all = np.concatenate(phi_buf, 0).astype(np.float64)     # (N, P)
    N = phi_all.shape[0]
    print(f"[slabvel] collected {N} slab examples (~{N // 8} pairs)", flush=True)

    # ---- per layer: PCA -> U_slab, then ridge phi -> U_slab coords -----------------------------------
    U_slab, W_slab, gate = {}, {}, {}
    for L in layers:
        X = np.concatenate(slab_buf[L], 0).astype(np.float64)  # (N, slabdim)
        slab_buf[L] = None; gc.collect()
        basis, ev = vo.pca_gram(X, k=KU)                        # (ku, slabdim)
        U_slab[L] = basis
        coords = X @ basis.T                                   # (N, ku)
        del X; gc.collect()
        ls = vo.LinearLS(P, basis.shape[0], args.ridge)
        ls.add(phi_all, coords)
        W_slab[L] = ls.solve().astype(np.float32)
        gate[L] = {"pr": round(vo.participation_ratio(ev), 2), "ku": int(basis.shape[0])}
        print(f"[slabvel] L{L}: U_slab rows={basis.shape[0]} slab_PR={gate[L]['pr']}", flush=True)
        del coords, ls; gc.collect()

    # ---- held-out gate: reconstruct slab edits from command, cos vs true slab dH ---------------------
    for L in layers:
        gate[L]["cmdU_cos"], gate[L]["coord_cos"] = [], []
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]
        grid = tuple(int(x) for x in sa["grid"]); T = grid[0]
        va_t = vo.clip_frame_velocities(sa, T)
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]
            vb_t = vo.clip_frame_velocities(sb, T)
            phis = np.stack([vo.command_features(va_t[t], vb_t[t]) for t in range(T)], 0)
            for L in layers:
                sd = slab_deltas(sa["layers"][L], sb["layers"][L], grid)   # (T,slab) true
                ctrue = sd @ U_slab[L].T                                    # (T,ku)
                cpred = phis @ W_slab[L].astype(np.float64)                 # (T,ku)
                rec = cpred @ U_slab[L]                                     # (T,slab)
                gate[L]["coord_cos"].append(vo.cosine(cpred.reshape(-1), ctrue.reshape(-1)))
                gate[L]["cmdU_cos"].append(vo.cosine(rec.reshape(-1), sd.reshape(-1)))
        del sa
        if hasattr(te, "_shard_cache") and len(te._shard_cache) > 2:
            te._shard_cache.clear()
        gc.collect()

    summary = {"quantity": "accel", "operator": "slabvel (temporal-composition velocity operator)",
               "layers": layers, "P": P, "KU": KU, "ridge": args.ridge,
               "n_slab_examples": int(N), "n_test_scenes": len(te_scenes), "per_layer": {}}
    for L in layers:
        row = {"slab_participation_ratio": gate[L]["pr"], "ku": gate[L]["ku"],
               "cmdU_cos": round(float(np.mean(gate[L]["cmdU_cos"])), 4),
               "coord_cos": round(float(np.mean(gate[L]["coord_cos"])), 4)}
        summary["per_layer"][str(L)] = row
        np.save(art / f"slabvel_basis_L{L}.npy", U_slab[L].astype(np.float32))
        np.save(art / f"cmd_Wslabvel_L{L}.npy", W_slab[L])
        print(f"[slabvel] L{L}: cmd_U reconstr cos={row['cmdU_cos']:.3f} "
              f"(coord cos={row['coord_cos']:.3f})", flush=True)
    summary["artifacts"] = {"U_slab": "slabvel_basis_L*.npy (ku, Hp*Wp*D)",
                            "W_slab": "cmd_Wslabvel_L*.npy (13, ku)",
                            "steer": "command_features(v_a(t), v_b(t)) @ W_slab @ U_slab per token"}
    (art / "cmd_operator_meta_slabvel.json").write_text(json.dumps(summary, indent=2))
    print(f"[slabvel] saved slabvel_basis + cmd_Wslabvel + meta -> {art}", flush=True)


if __name__ == "__main__":
    main()
