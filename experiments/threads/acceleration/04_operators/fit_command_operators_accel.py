

#!/usr/bin/env python
"""Fit COMMAND-ONLY ACCELERATION operators that synthesize the edit inside the global accel subspace U.

The acceleration analog of ``experiments/pipeline/04_operators/fit_command_operators.py`` (Step 2 "Bravo"). Identical streaming
ridge math; the only change is the per-clip command is the constant 2D ACCELERATION
(``velocity_ops.clip_acceleration``), so ``command_features(a_a, a_b)`` -> U-coordinates predicts the
acceleration edit without H_b. Reuses the U basis saved by ``accel_subspace.py`` (``global_basis_L*.npy``)
in the accel artifacts dir, and writes the SAME artifact names (``cmd_Wu*_L*.npy``, ``cmd_Brich_L*.npy``)
there so ``steer_accel2d.py`` consumes them unchanged.

  * ``W_U`` : command_features (13) -> U_KU coordinates (KU).  Reconstruct edit = c @ U[:KU]  (cmd_U8).
  * ``B_rich``: command_features (13) -> full flattened Delta H (D).  Richer-feature global ridge.

Held-out TEST latent gate: cos(pred, true Delta H) for each, plus the cos of the U-coordinate prediction
vs the true U coordinates. Decode (experiments/threads/acceleration/05_steering/steer_accel2d.py) is the decisive test.

    python experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
        --train_dir .../train/vjepa2_large --test_dir .../test/vjepa2_large \
        --layers 6,12,18,23 --artifacts_dir outputs/analysis/moving_ball_accel2d/subspace --ridge 1.0
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


QUANTITY_FNS = {          # one harness for every physics quantity in the sweep
    "accel": vo.clip_acceleration,   # also used for `gravity` (a 1-DOF acceleration scenario)
    "velocity": vo.clip_velocity,
    "angvel": vo.clip_angvel,
    "angaccel": vo.clip_angaccel,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--layers", default="6,12,18,23")
    p.add_argument("--artifacts_dir", required=True, help="dir with global_basis_L*.npy; outputs land here")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--quantity", choices=sorted(QUANTITY_FNS), default="accel",
                   help="accel -> clip_acceleration; angvel -> clip_angvel ([omega,0], rotational).")
    p.add_argument("--basis_tag", default="",
                   help="Load global_basis_{tag}_L*.npy (e.g. 'spd' = U8 + supervised |a| axes from "
                        "speed_axis.py --quantity accel); --ku 0 uses every row; tag appended to artifacts.")
    p.add_argument("--ku", type=int, default=8,
                   help="U-subspace rank to synthesize into (needs global_basis with >= ku rows). "
                        "Saved artifacts are tagged cmd_Wu_ku{ku}_L*.npy when ku != 8 so U8/U16 coexist.")
    p.add_argument("--features", choices=["base", "quad"], default="base",
                   help="base -> command_features (13); quad -> command_features_quad (26, adds 2nd-order "
                        "terms). quad artifacts are tagged cmd_Wu_quad*_L*.npy so both coexist.")
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--standardize", action="store_true",
                   help="scale-free ridge (LinearLS.solve(standardize=True)): rescale every command "
                        "feature column to unit RMS before the penalty. Without it the ~0.002-scale "
                        "acceleration columns are shrunk ~1e5x harder than the unit-heading columns, "
                        "so the operator steers direction and drops magnitude. Artifacts tagged cmd_Wu_std*.")
    args = p.parse_args()

    feat_fn = vo.command_features_quad if args.features == "quad" else vo.command_features
    P = vo.COMMAND_FEATURE_DIM_QUAD if args.features == "quad" else vo.COMMAND_FEATURE_DIM
    feat_tag = "_quad" if args.features == "quad" else ""
    qfn = QUANTITY_FNS[args.quantity]

    KU = int(args.ku)
    layers = [int(x) for x in args.layers.split(",")]
    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    btag = f"_{args.basis_tag}" if args.basis_tag else ""
    U = {L: np.load(art / f"global_basis{btag}_L{L}.npy").astype(np.float64) for L in layers}
    if KU <= 0:
        KU = min(U[L].shape[0] for L in layers)
    U = {L: U[L][:KU] for L in layers}
    for L in layers:
        if U[L].shape[0] < KU:
            raise SystemExit(f"global_basis{btag}_L{L}.npy has {U[L].shape[0]} rows < ku={KU}; "
                             f"re-run accel_subspace.py with --save_k {KU}")

    tr = LatentDataset(args.train_dir, layers=layers)
    te = LatentDataset(args.test_dir, layers=layers)
    tr_scenes, te_scenes = vo.group_scenes(tr), vo.group_scenes(te)
    if args.max_scenes:
        tr_scenes = {s: tr_scenes[s] for s in sorted(tr_scenes)[: args.max_scenes]}
        te_scenes = {s: te_scenes[s] for s in sorted(te_scenes)[: args.max_scenes]}
    print(f"[a-cmd] train {len(tr_scenes)} scenes, test {len(te_scenes)}; layers={layers}; "
          f"features={args.features} P={P} KU={KU}", flush=True)

    rich = {L: None for L in layers}
    wu = {L: vo.LinearLS(P, KU, args.ridge) for L in layers}

    n = 0
    for s in sorted(tr_scenes):
        ranks = sorted(tr_scenes[s]); a = ranks[0]
        sa = tr[tr_scenes[s][a]]; va = qfn(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = tr[tr_scenes[s][b]]; vb = qfn(sb)
            phi = feat_fn(va, vb).reshape(1, P)
            for L in layers:
                dH = (vo.layer_flat(sb["layers"][L]) - Ha[L])
                if rich[L] is None:
                    rich[L] = vo.LinearLS(P, dH.size, args.ridge)
                rich[L].add(phi, dH.reshape(1, -1))
                wu[L].add(phi, (U[L] @ dH).reshape(1, KU))
        del Ha; gc.collect()
        n += 1
        if n % 100 == 0:
            print(f"[a-cmd]   trained {n}/{len(tr_scenes)} scenes", flush=True)

    B_rich = {L: rich[L].solve(args.standardize).astype(np.float32) for L in layers}
    W_U = {L: wu[L].solve(args.standardize).astype(np.float32) for L in layers}
    del rich, wu; gc.collect()

    gate = {L: {"rich_cos": [], "cmdU_cos": [], "coord_cos": []} for L in layers}
    # Cache just enough to recompute the gate under a SHUFFLED command afterwards. A raw cos is
    # not interpretable on its own -- dH is not isotropic, so a meaningless operator still scores
    # well above zero. The control is the same operator driven by another pair's command: matched
    # command statistics, wrong semantics. Because U is an orthonormal PCA basis,
    #   cos(cpred @ U, dH) = (cpred . (U @ dH)) / (|cpred| * |dH|)
    # so caching ctrue = U@dH (KU-dim) and |dH| (scalar) is enough -- no need to hold any dH.
    phis: list[np.ndarray] = []
    pair_scene: list[int] = []   # scene index per pair, so the bootstrap can resample by SCENE
    ctrue_by_layer = {L: [] for L in layers}
    dHnorm_by_layer = {L: [] for L in layers}
    for si, s in enumerate(sorted(te_scenes)):
        ranks = sorted(te_scenes[s]); a = ranks[0]
        sa = te[te_scenes[s][a]]; va = qfn(sa)
        Ha = {L: vo.layer_flat(sa["layers"][L]) for L in layers}
        for b in ranks[1:]:
            sb = te[te_scenes[s][b]]; vb = qfn(sb)
            phi = feat_fn(va, vb)
            phis.append(np.asarray(phi, dtype=np.float64))
            pair_scene.append(si)
            for L in layers:
                dH = vo.layer_flat(sb["layers"][L]) - Ha[L]
                ctrue = U[L] @ dH
                cpred = phi @ W_U[L].astype(np.float64)
                gate[L]["coord_cos"].append(vo.cosine(cpred, ctrue))
                gate[L]["cmdU_cos"].append(vo.cosine(cpred @ U[L], dH))
                gate[L]["rich_cos"].append(vo.cosine(phi @ B_rich[L].astype(np.float64), dH))
                ctrue_by_layer[L].append(ctrue)
                dHnorm_by_layer[L].append(float(np.linalg.norm(dH)))
        del Ha; gc.collect()

    # Shuffled-command control: derangement so no pair keeps its own command.
    rng = np.random.default_rng(0)
    n_pairs = len(phis)
    perm = rng.permutation(n_pairs)
    for i in range(n_pairs):                      # fix any accidental fixed points
        if perm[i] == i:
            perm[i], perm[(i + 1) % n_pairs] = perm[(i + 1) % n_pairs], perm[i]
    shuf = {L: {"cmdU_cos": [], "coord_cos": []} for L in layers}
    for L in layers:
        WuL = W_U[L].astype(np.float64)
        for i in range(n_pairs):
            cpred_s = phis[perm[i]] @ WuL
            ctrue = ctrue_by_layer[L][i]
            nd = dHnorm_by_layer[L][i]
            denom = np.linalg.norm(cpred_s) * nd
            shuf[L]["cmdU_cos"].append(float(cpred_s @ ctrue / denom) if denom > 0 else 0.0)
            shuf[L]["coord_cos"].append(vo.cosine(cpred_s, ctrue))

    # tag = features (_quad) then rank (_ku16); base+ku8 stays untagged for back-compat with existing runs.
    wu_tag = feat_tag + ("" if KU == 8 else f"_ku{KU}")
    if args.standardize:                 # keep both fits on disk; never clobber the published one
        wu_tag = "_std" + wu_tag
    wu_tag = wu_tag + btag
    summary = {"layers": layers, "P": P, "KU": KU, "ridge": args.ridge, "quantity": args.quantity,
               "basis_tag": args.basis_tag,
               "standardize": bool(args.standardize),
               "features": args.features, "n_train_scenes": len(tr_scenes), "n_test_scenes": len(te_scenes),
               "per_layer": {}}
    for L in layers:
        row = {k: round(float(np.mean(v)), 4) for k, v in gate[L].items()}
        row["cmdU_cos_shuffled"] = round(float(np.mean(shuf[L]["cmdU_cos"])), 4)
        row["coord_cos_shuffled"] = round(float(np.mean(shuf[L]["coord_cos"])), 4)
        # The headline: how far the real command beats a mismatched one at the same statistics.
        row["cmdU_cos_lift"] = round(row["cmdU_cos"] - row["cmdU_cos_shuffled"], 4)
        row["n_pairs"] = int(n_pairs)
        summary["per_layer"][str(L)] = row
        np.save(art / f"cmd_Wu{wu_tag}_L{L}.npy", W_U[L])
        if not args.standardize:         # ridge_rich reference stays the published (unstandardized) fit
            np.save(art / f"cmd_Brich{feat_tag}_L{L}.npy", B_rich[L])
        # Per-PAIR cos values, not just their mean: a mean lift is uninterpretable without a
        # confidence interval, and because every encoder sees the identical seeded scenes in
        # sorted order these arrays are aligned across encoders and support a PAIRED bootstrap.
        np.save(art / f"gate_pairs_cmdU_L{L}.npy", np.asarray(gate[L]["cmdU_cos"], dtype=np.float32))
        np.save(art / f"gate_pairs_cmdU_shuf_L{L}.npy",
                np.asarray(shuf[L]["cmdU_cos"], dtype=np.float32))
        np.save(art / f"gate_pairs_scene_L{L}.npy", np.asarray(pair_scene, dtype=np.int32))
        print(f"[a-cmd] L{L}: cmd_U reconstr cos={row['cmdU_cos']:.3f} (U-coord cos={row['coord_cos']:.3f}) "
              f"| ridge_rich cos={row['rich_cos']:.3f} "
              f"| SHUFFLED cmd={row['cmdU_cos_shuffled']:.3f} -> lift={row['cmdU_cos_lift']:+.3f}", flush=True)
    summary["artifacts"] = {"W_U": f"cmd_Wu{wu_tag}_L*.npy (P,KU) -> coords; edit = coords @ U[:KU]",
                            "B_rich": f"cmd_Brich{feat_tag}_L*.npy (P,D)",
                            "command_features": f"velocity_ops.command_features"
                                                f"{'_quad' if args.features == 'quad' else ''}(a_a,a_b) ({P},)"}
    meta_tag = ("_std" if args.standardize else "") + feat_tag
    (art / f"cmd_operator_meta{meta_tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"[a-cmd] saved W_U + B_rich + cmd_operator_meta -> {art} (features={args.features})", flush=True)


if __name__ == "__main__":
    main()
