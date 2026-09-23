

#!/usr/bin/env python
"""Fit a COMMAND-ONLY model of the acceleration edit's SPATIAL RESIDUAL.

Why this exists. At n=700 the only arm that ever went below the oracle was ``oman``:

    edit = [ dH - broadcast(dR) ]  +  broadcast( manifold-predicted profile )
           \\________ true spatial residual ________/    \\____ command-only ____/

10.46deg against full_delta's 11.23. Everything on the right of that ``+`` is command-only already; the
left half is not, and it is the whole reason every command-only arm has been stuck at ~14deg while the
oracle sits at 11.2. The pooled profile ``dR`` throws away WHERE the edit lands, and ``prof_full`` (the
true profile, spatially uniform) decodes at 64deg ungained -- placement is most of the signal.

So: can the residual be synthesized from the command?

    S_i = dH_i - broadcast(dR_i)   in R^{T*H*W*D},  ~2.1M dims, 3500 train pairs

Two streaming passes, because S is far too large to hold:

  pass 1  randomized range finding -- accumulate ``Y = sum_i g_i (x) S_i`` for a fixed Gaussian sketch
          ``g_i in R^r``, then orthonormalize to get a basis ``Q (r, 2.1M)``.
  pass 2  project every training residual onto ``Q`` and ridge-regress its coordinates on placement-aware
          command features (the base 13-dim command vector plus the anchor's own ``pos0`` and ``v0`` and
          their interactions with ``da`` -- the trajectory IS determined by those, and the previous
          spatial attempts in this repo all conditioned on the command alone).

The held-out gate reports two numbers that decide whether a decode is worth running at all:

  capture_cos   cos(Q Q^T S, S) on TEST -- can a rank-r subspace even EXPRESS the residual?
  pred_cos      cos(predicted S, true S) on TEST -- can the command reach it?

``capture_cos`` is the ceiling. This repo has four dead spatial-placement arms (transfield, posefield,
trajcanon, slabvel) and a saved 128-dim delta basis that reconstructs dH badly (oprojU128 decoded at
20.7 against full_delta's 11.2), so a low ceiling here is the expected outcome and is worth reporting
as such rather than decoding into noise.

    python experiments/threads/acceleration/04_operators/fit_accel_residual.py --train_dir ... --test_dir ... --rank 192 \
        --output_dir outputs/analysis/moving_ball_accel2d_mixed/residual
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


def placement_features(aa, ab, pos0, v0) -> np.ndarray:
    """Command features that can actually locate the edit: base command + anchor pose + interactions.

    The anchor's start position and initial velocity are observables of the clip being steered, not of
    the target -- reading them keeps the operator command-only in exactly the sense every other arm in
    this thread uses. Under constant acceleration the target trajectory is
    ``pos0 + v0 f + a_b f(f-1)/2``, so ``pos0``, ``v0``, ``a_b`` and their products are the minimal set
    that determines WHERE the edit has to land.
    """
    base = vo.command_features(aa, ab)                       # (13,)
    p = np.asarray(pos0, dtype=np.float64).reshape(2)
    v = np.asarray(v0, dtype=np.float64).reshape(2)
    da = np.asarray(ab, dtype=np.float64).reshape(2) - np.asarray(aa, dtype=np.float64).reshape(2)
    inter = np.concatenate([np.outer(p, da).reshape(-1), np.outer(v, da).reshape(-1),
                            np.outer(p, v).reshape(-1)])     # (12,)
    return np.concatenate([base, p, v, inter, [float(p @ p), float(v @ v)]])   # 13+2+2+12+2 = 31


FEATURE_DIM = 31


def anchor_pose(sample):
    keys = list(sample["state_keys"]); st = np.asarray(sample["state"])
    return (np.array([float(st[0, keys.index("obj0_pos_x")]), float(st[0, keys.index("obj0_pos_y")])]),
            np.array([float(st[0, keys.index("obj0_vel_x")]), float(st[0, keys.index("obj0_vel_y")])]))


def residual(flat_b, flat_a, grid):
    """``S = dH - broadcast(pooled(dH))`` -- the part of the edit that the profile arms cannot see."""
    dH = flat_b - flat_a
    return dH - sp.broadcast_profile(sp.temporal_profile(dH, grid), grid)


def iter_pairs(ds, scenes, scene_ids, layers, grid, qfn):
    """Yield ``(phi, {L: S})`` for every rank0 -> rank_r pair of every scene, one scene resident."""
    for s in scene_ids:
        ranks = sorted(scenes[s])
        sa = ds[scenes[s][ranks[0]]]
        aa = qfn(sa)
        pos0, v0 = anchor_pose(sa)
        flat_a = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = ds[scenes[s][b]]
            phi = placement_features(aa, qfn(sb), pos0, v0)
            yield phi, {L: residual(vo.layer_flat(sb["layers"][L]), flat_a[L], grid) for L in layers}
            del sb
        del sa, flat_a
        gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--rank", type=int, default=192, help="dims of the streamed residual subspace")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--quantity", choices=["accel", "angvel"], default="accel")
    args = p.parse_args()

    qfn = vo.clip_angvel if args.quantity == "angvel" else vo.clip_acceleration
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    r = int(args.rank)

    tr = LatentDataset(args.train_dir, layers=layers, max_cached_shards=2)
    te = LatentDataset(args.test_dir, layers=layers, max_cached_shards=2)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    tr_ids, te_ids = sorted(tr_scenes), sorted(te_scenes)
    if args.max_scenes:
        tr_ids, te_ids = tr_ids[: args.max_scenes], te_ids[: args.max_scenes]
    grid = tuple(int(x) for x in tr[tr_scenes[tr_ids[0]][0]]["grid"])
    Dfull = int(np.prod(grid)) * int(tr.records[0]["hidden_dim"])
    print(f"[res-fit] train {len(tr_ids)} scenes, test {len(te_ids)}; grid={grid}; "
          f"residual dim={Dfull}; rank={r}", flush=True)

    # ------------------------------------------------------------------ pass 1: randomized range finding
    rng = np.random.default_rng(0)
    Y = {L: np.zeros((r, Dfull), dtype=np.float32) for L in layers}
    n = 0
    for phi, S in iter_pairs(tr, tr_scenes, tr_ids, layers, grid, qfn):
        g = rng.standard_normal(r).astype(np.float32)
        for L in layers:
            Y[L] += np.outer(g, S[L].astype(np.float32))
        n += 1
        if n % 350 == 0:
            print(f"[res-fit]   pass1 {n} pairs", flush=True)
    print(f"[res-fit] pass1 done ({n} pairs); orthonormalizing", flush=True)

    Q = {}
    for L in layers:
        Qm, _ = np.linalg.qr(Y[L].T)             # (Dfull, r)
        Q[L] = np.ascontiguousarray(Qm.T)        # (r, Dfull)
        del Qm
        Y[L] = None
        gc.collect()
    del Y
    gc.collect()

    # ------------------------------------------------------------------ pass 2: ridge on the coordinates
    ls = {L: vo.LinearLS(FEATURE_DIM, r, args.ridge) for L in layers}
    n = 0
    for phi, S in iter_pairs(tr, tr_scenes, tr_ids, layers, grid, qfn):
        f = phi.reshape(1, FEATURE_DIM)
        for L in layers:
            ls[L].add(f, (Q[L] @ S[L].astype(np.float32)).reshape(1, r).astype(np.float64))
        n += 1
        if n % 350 == 0:
            print(f"[res-fit]   pass2 {n} pairs", flush=True)
    W = {L: ls[L].solve() for L in layers}       # (FEATURE_DIM, r)
    del ls
    gc.collect()

    for L in layers:
        np.save(out / f"resid_Q_L{L}.npy", Q[L].astype(np.float32))
        np.save(out / f"resid_W_L{L}.npy", W[L].astype(np.float32))

    # ------------------------------------------------------------------ held-out gate
    tr._shard_cache.clear(); gc.collect()
    cap = {L: [] for L in layers}
    prd = {L: [] for L in layers}
    m = 0
    for phi, S in iter_pairs(te, te_scenes, te_ids, layers, grid, qfn):
        for L in layers:
            s32 = S[L].astype(np.float32)
            c = Q[L] @ s32
            cap[L].append(vo.cosine((c @ Q[L]).astype(np.float64), S[L]))
            prd[L].append(vo.cosine(((phi @ W[L]).astype(np.float32) @ Q[L]).astype(np.float64), S[L]))
        m += 1
        if m % 175 == 0:
            print(f"[res-fit]   gated {m} pairs", flush=True)

    gate = {str(L): {"capture_cos": round(float(np.mean(cap[L])), 4),
                     "pred_cos": round(float(np.mean(prd[L])), 4)} for L in layers}
    summary = {
        "what": "command-only model of the acceleration edit's SPATIAL RESIDUAL dH - broadcast(dR)",
        "train_dir": args.train_dir, "test_dir": args.test_dir, "layers": layers, "grid": list(grid),
        "residual_dim": Dfull, "rank": r, "ridge": args.ridge, "feature_dim": FEATURE_DIM,
        "n_train_pairs": n, "n_test_pairs": m,
        "heldout_gate": gate,
        "artifacts": "resid_Q_L{L}.npy (r, Dfull) + resid_W_L{L}.npy (31, r)",
        "read_this_first": ("capture_cos is the CEILING -- what a rank-r subspace can express at all. "
                            "pred_cos is what the command reaches. A low ceiling means the residual is "
                            "high-rank and no amount of conditioning will synthesize it; report that "
                            "rather than decoding into noise."),
    }
    (out / "residual_meta.json").write_text(json.dumps(summary, indent=2))
    print("\n[res-fit] held-out gate (mean cosine vs the true residual):")
    for L in layers:
        print(f"  L{L:<3} capture(ceiling)={gate[str(L)]['capture_cos']:.3f}  "
              f"pred={gate[str(L)]['pred_cos']:.3f}")
    print(f"[res-fit] -> {out}/residual_meta.json")


if __name__ == "__main__":
    main()
