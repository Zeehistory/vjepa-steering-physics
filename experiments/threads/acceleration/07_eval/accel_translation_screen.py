

#!/usr/bin/env python
"""2nd-order screen: is the accel latent edit dH a POSITION-CONDITIONED per-frame TRANSLATION by 1/2*Da*t^2?

Physics: in a scene all ranks share pos0 AND v0, so the trajectory difference between target B and reference
A is EXACT and closed-form:  Dx(t) = x_b(t) - x_a(t) = 1/2 (a_b - a_a) t^2.  pos0/v0 cancel identically, so
the ONLY difference between the two clips at temporal token t is that the ball is TRANSLATED by Dx(t) (dir =
Da, magnitude grows as t^2, zero at t=0). Hence dH(t)_slab = translation-response of moving the ball from
x_a(t) to x_a(t)+Dx(t). This is a function of (x_a(t), Dx(t)) ONLY -- both available at steer time (x_a from
the reference clip, Dx from the command via the t^2 law), NO H_b.

Every operator so far (base..canon..trajcanon) fit one GLOBAL command -> one dH blob, averaging the t^2
temporal profile and blind to placement; canon tried to place by discrete ROLLING (lossy -> trajcanon hurt).
This screen instead drives EACH temporal-token slab by its exact displacement and conditions on ball
position -- a translation FIELD, no rolling. It fits shared per-token linear maps and reports held-out recon
cosine of the ASSEMBLED dH vs truth. Arms:
    base_global : command_features(13) -> full dH               (reproduces the ~0.28 command-only floor)
    disp        : Dx(t)(2) -> slab(t), shared over t            (t^2 profile + dir=Da, NO position)
    disp_pos    : [Dx || Dx (x) posbasis(x_a(t))] -> slab(t)     (position-conditioned translation, THE test)
DECISIVE: disp_pos recon cos >> 0.28 -> a translation-field operator can beat 14.5deg; build the decode.
Flat -> the edit is not a low-order position-conditioned translation; 14.46 stands.
Dx uses GT state positions (obj0_pos, NOT the H_b latent) and equals 1/2*Da*t^2, so it is command-derivable.

    python experiments/threads/acceleration/07_eval/accel_translation_screen.py --train_dir .../train/vjepa2_large \
        --test_dir .../test/vjepa2_large --layers 6,12,18,23 --ridge 1.0 --out .../translation_screen.json
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

P = vo.COMMAND_FEATURE_DIM  # 13


def posbasis(x: np.ndarray) -> np.ndarray:
    """Smooth 2D position basis for a normalized [0,1] ball centre -> [1,px,py,px^2,py^2,px*py] (6)."""
    px, py = float(x[0]), float(x[1])
    return np.array([1.0, px, py, px * px, py * py, px * py])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--out", required=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    trs, tes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        trs = {s: trs[s] for s in sorted(trs)[: args.max_scenes]}
        tes = {s: tes[s] for s in sorted(tes)[: args.max_scenes]}
    order = sorted(trs)
    print(f"[trans] train {len(trs)} scenes, test {len(tes)}; layers={layers}", flush=True)

    PB = 6                       # posbasis dim
    F_disp = 2                   # Dx
    F_pos = 2 + 2 * PB           # [Dx || Dx (x) posbasis] = 14
    # per-token slab maps (shared over tokens): built lazily once slab dim is known
    m_disp = {L: None for L in layers}
    m_pos = {L: None for L in layers}
    m_base = {L: vo.LinearLS(P, 1, args.ridge) for L in layers}  # placeholder; real base built lazily too
    base_built = {"done": False}

    def token_feats(dx, xa):
        pb = posbasis(xa)
        return dx.astype(np.float64), np.concatenate([dx, np.outer(dx, pb).reshape(-1)])

    # ---- fit ---------------------------------------------------------------------------------------
    n = 0
    for s in order:
        ranks = sorted(trs[s]); sa = tr[trs[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"]); T, H, W = grid
        pa = vo.clip_frame_positions(sa, T)                       # (T,2) reference ball path
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        Da = {L: Ha[L].size // (T * H * W) for L in layers}       # feature dim D per token
        for b in ranks[1:]:
            sb = tr[trs[s][b]]; vb = vo.clip_acceleration(sb)
            pb_pos = vo.clip_frame_positions(sb, T)               # (T,2) target ball path
            dx_t = pb_pos - pa                                    # (T,2) = 1/2 Da t^2 per token
            phi = vo.command_features(va, vb).reshape(1, P)
            for L in layers:
                D = Da[L]; slab_dim = H * W * D
                if m_disp[L] is None:
                    m_disp[L] = vo.LinearLS(F_disp, slab_dim, args.ridge)
                    m_pos[L] = vo.LinearLS(F_pos, slab_dim, args.ridge)
                    m_base[L] = vo.LinearLS(P, Ha[L].size, args.ridge)
                dH = (vo.layer_flat(sb["layers"][L]) - Ha[L])
                m_base[L].add(phi, dH.reshape(1, -1))
                slabs = dH.reshape(T, H * W * D)
                for t in range(T):
                    fd, fp = token_feats(dx_t[t], pa[t])
                    m_disp[L].add(fd.reshape(1, F_disp), slabs[t].reshape(1, slab_dim))
                    m_pos[L].add(fp.reshape(1, F_pos), slabs[t].reshape(1, slab_dim))
        del Ha; gc.collect()
        n += 1
        if n % 50 == 0:
            print(f"[trans]   fit {n}/{len(trs)}", flush=True)
    B_disp = {L: m_disp[L].solve() for L in layers}
    B_pos = {L: m_pos[L].solve() for L in layers}
    B_base = {L: m_base[L].solve() for L in layers}
    del m_disp, m_pos, m_base; gc.collect()
    print("[trans] fit done", flush=True)

    # ---- held-out recon cosine of assembled dH -----------------------------------------------------
    acc = {a: {L: [] for L in layers} for a in ("base_global", "disp", "disp_pos")}
    for s in sorted(tes):
        ranks = sorted(tes[s]); sa = te[tes[s][ranks[0]]]; va = vo.clip_acceleration(sa)
        grid = tuple(int(x) for x in sa["grid"]); T, H, W = grid
        pa = vo.clip_frame_positions(sa, T)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[tes[s][b]]; vb = vo.clip_acceleration(sb)
            pb_pos = vo.clip_frame_positions(sb, T)
            dx_t = pb_pos - pa
            phi = vo.command_features(va, vb)
            for L in layers:
                D = Ha[L].size // (T * H * W); slab_dim = H * W * D
                dH = (vo.layer_flat(sb["layers"][L]) - Ha[L])
                pred_base = phi @ B_base[L]
                acc["base_global"][L].append(vo.cosine(pred_base, dH))
                pd = np.empty((T, slab_dim)); pp = np.empty((T, slab_dim))
                for t in range(T):
                    fd, fp = token_feats(dx_t[t], pa[t])
                    pd[t] = fd @ B_disp[L]
                    pp[t] = fp @ B_pos[L]
                acc["disp"][L].append(vo.cosine(pd.reshape(-1), dH))
                acc["disp_pos"][L].append(vo.cosine(pp.reshape(-1), dH))
        del Ha; gc.collect()

    summary = {"layers": layers, "ridge": args.ridge, "n_train": len(trs), "n_test": len(tes),
               "metric": "held-out recon cos of assembled dH vs true dH (full latent)",
               "note": "base_global command-only ~ the 0.28 floor; disp/disp_pos = 2nd-order translation field",
               "recon_cos": {}}
    for a in acc:
        summary["recon_cos"][a] = {str(L): round(float(np.mean(v)), 4) for L, v in acc[a].items()}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("[trans] RECON COS (held-out, full-latent):", flush=True)
    for L in layers:
        print(f"  L{L}: base={summary['recon_cos']['base_global'][str(L)]:.3f}  "
              f"disp={summary['recon_cos']['disp'][str(L)]:.3f}  "
              f"disp_pos={summary['recon_cos']['disp_pos'][str(L)]:.3f}", flush=True)
    print(f"[trans] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
