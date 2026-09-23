

#!/usr/bin/env python
"""APPEARANCE-CONDITIONED acceleration operator (Step 2 "Bravo", disentanglement push).

Diagnosis behind this script: on the appearance-MIXED accel dataset the command-only cmd_U operator tops
out at ~16.9deg held-out vs a 10.1deg full_delta ceiling, while the HOMOGENEOUS-appearance case reached
~10.3deg. Quadratic command features did NOT help (identical 16.88deg) -> the command->U map is not the
bottleneck; the residual is per-scene APPEARANCE modulation of the accel edit that the command (blind to
colour/background) cannot express. dH = H_b - H_a already cancels the STATIC appearance (colour/bg is
fixed within a scene), but the SAME Delta-accel produces a different dH depending on the scene's
appearance, so a single shared subspace / command map averages over appearance variants.

Fix: condition the operator on the reference appearance, which is READ FROM H_a (the clip we steer FROM,
available at steer time -- NOT the target H_b, so this stays a legitimate "no H_b" operator). Appearance
is summarised by mean-pooling H_a over tokens (bg dominates spatially -> captures colour/bg) and projecting
onto the top-Ka PCs of that pooled vector across TRAIN scenes. The operator is then a single ridge

    [ command_features(a_a,a_b) (13)  ||  appc(H_a) (Ka) ]  ->  dH   (full flattened per-layer edit)

i.e. the appearance-augmented analog of the rich (full-D) command ridge. Full-D (not U-restricted) so the
edit can leave the shared accel subspace and reach the per-scene, appearance-modulated target. Held-out
TEST appearance is freshly sampled but from the same colour/bg ranges, so the low-DOF appearance map
generalises. Consumed by steer_accel2d.py --features appc.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel_appc.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --layers 6,12,18,23 --artifacts_dir .../subspace --ka 16 --ridge 1.0
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

P = vo.COMMAND_FEATURE_DIM  # 13 base command features


def pooled(sample_layer) -> np.ndarray:
    """Appearance summary of one layer's latent = mean over tokens -> (D_hidden,)."""
    return np.asarray(sample_layer, dtype=np.float64).mean(axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True, help="dir where appc_* and cmd_Bappc_* artifacts land")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ka", type=int, default=16, help="number of appearance PCs read from pooled H_a")
    p.add_argument("--max_scenes", type=int, default=0)
    args = p.parse_args()

    Ka = int(args.ka)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    print(f"[a-appc] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; "
          f"P={P} Ka={Ka}", flush=True)

    # ---- pass 0: appearance basis from pooled H_a (rank0) across TRAIN scenes -----------------------
    pool_buf = {L: [] for L in layers}
    order = sorted(tr_scenes)
    for s in order:
        sa = tr[tr_scenes[s][sorted(tr_scenes[s])[0]]]
        for L in layers:
            pool_buf[L].append(pooled(sa["layers"][L]))
    appc_mean, appc_basis = {}, {}
    for L in layers:
        X = np.stack(pool_buf[L], 0)  # (N_scenes, D_hidden)
        appc_mean[L] = X.mean(0)
        appc_basis[L], _ = vo.pca_gram(X, k=Ka)  # (<=Ka, D_hidden); < Ka only if rank-deficient (few scenes)
    del pool_buf; gc.collect()
    # PCA can return fewer than Ka components if the pooled data is rank-deficient (few scenes); the true
    # appearance rank is whatever pca_gram gave. Assume the (shared, low-DOF) appearance rank matches across
    # layers -> one Ka for all; truncate to the common min so a per-layer mismatch can't corrupt the operator.
    ka_actual = {L: appc_basis[L].shape[0] for L in layers}
    Ka = min(ka_actual.values())
    for L in layers:
        appc_basis[L] = appc_basis[L][:Ka]
        np.save(art / f"appc_mean_L{L}.npy", appc_mean[L].astype(np.float32))
        np.save(art / f"appc_basis_L{L}.npy", appc_basis[L].astype(np.float32))
    print(f"[a-appc] appearance basis built (Ka requested={args.ka}, actual={Ka}, "
          f"per-layer={ka_actual}) + saved appc_mean/appc_basis", flush=True)

    def appc_feat(sa) -> dict:
        out = {}
        for L in layers:
            out[L] = (pooled(sa["layers"][L]) - appc_mean[L]) @ appc_basis[L].T  # (Ka,)
        return out

    # ---- pass 1: fit [command || appc(H_a)] -> dH (full D) -----------------------------------------
    Pa = P + Ka
    fit = {L: None for L in layers}
    n = 0
    for s in order:
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]; va = vo.clip_acceleration(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = appc_feat(sa)
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]; vb = vo.clip_acceleration(sb)
            cmd = vo.command_features(va, vb)
            for L in layers:
                phi = np.concatenate([cmd, ca[L]]).reshape(1, Pa)
                dH = (vo.layer_flat(sb["layers"][L]) - Ha[L])
                if fit[L] is None:
                    fit[L] = vo.LinearLS(Pa, dH.size, args.ridge)
                fit[L].add(phi, dH.reshape(1, -1))
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[a-appc]   trained {n}/{len(tr_scenes)} scenes", flush=True)

    B_appc = {L: fit[L].solve().astype(np.float32) for L in layers}
    del fit; gc.collect()

    # ---- held-out gate: cos(appc-pred dH, true dH) vs command-only rich ----------------------------
    gate = {L: {"appc_cos": []} for L in layers}
    for s in sorted(te_scenes):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = vo.clip_acceleration(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        ca = appc_feat(sa)
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; vb = vo.clip_acceleration(sb)
            cmd = vo.command_features(va, vb)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                phi = np.concatenate([cmd, ca[L]])
                gate[L]["appc_cos"].append(vo.cosine(phi @ B_appc[L].astype(np.float64), dH))
        del Ha; gc.collect()

    summary = {"layers": layers, "P": P, "Ka": Ka, "Pa": Pa, "ridge": args.ridge, "quantity": "accel",
               "features": "appc", "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes),
               "per_layer": {}}
    for L in layers:
        row = {"appc_cos": round(float(np.mean(gate[L]["appc_cos"])), 4)}
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Bappc_L{L}.npy", B_appc[L])
        print(f"[a-appc] L{L}: appc rich cos(dH)={row['appc_cos']:.3f} "
              f"(command-only rich was ~0.27-0.31)", flush=True)
    summary["artifacts"] = {"B_appc": "cmd_Bappc_L*.npy (P+Ka, D); edit = [cmd||appc(H_a)] @ B_appc",
                            "appc_mean": "appc_mean_L*.npy (D_hidden,)",
                            "appc_basis": "appc_basis_L*.npy (Ka, D_hidden)"}
    (art / "cmd_operator_meta_appc.json").write_text(json.dumps(summary, indent=2))
    print(f"[a-appc] saved B_appc + appc basis + meta -> {art} (Ka={Ka})", flush=True)


if __name__ == "__main__":
    main()
