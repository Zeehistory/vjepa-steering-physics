

#!/usr/bin/env python
"""Ablation grid for the FOURIER-IN-ORIENTATION operator, measured by the held-out latent gate.

Runs the whole grid inside ONE job against a single in-memory latent cache: loading the latents dominates
the cost (~8 GB per layer for 1000 clips), while each ablation is only a small per-token ridge solve, so
one-job-per-config would pay the I/O over and over.

The metric is the held-out latent gate -- cos(synthesized dH, TRUE H_b - H_a) -- against a shuffled-command
control, which needs no decoder. It is the honest cheap proxy: it asks whether the operator reconstructs the
real latent edit, and it was the gate that correctly predicted the angvel decode (gate 0.588 -> rho 0.94).
Decode-verification of the decisive configs is a separate GPU run; the gate cannot replace it, only rank
candidates for it.

AXES (each isolated, everything else at the operator's default order=4 / all-harmonics / canon-on /
orientation-basis / ridge=10):
  * order        -- 0 (DC only, must be null: a constant cannot express a rotation) through 8.
  * harmonics    -- all / even / odd. The MECHANISTIC test: the bar has pi-symmetry so its content should
                    live in the EVEN harmonics, while the red marker breaks the ambiguity at 2pi and should
                    live in the FUNDAMENTAL (odd). This predicts specific, falsifiable failures.
  * canon        -- center-canonicalization on/off. Predicted to matter a lot: without it, scenes rotating
                    about different centres place the same orientation change on different cells, so no
                    shared operator exists (the original "no shared axis" verdict was largely this).
  * basis        -- orientation vs orientation_rate (rate-interacted harmonics). Asks how much of the
                    residual is the rate/motion content a pose-only model ignores.
  * ridge        -- regularization sensitivity.
  * n_train      -- data efficiency: scenes needed to identify the operator.
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

import argparse, json, time
from pathlib import Path
import numpy as np

from src.analysis import fourier_orientation as fo
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config
from steer_fourier import clip_command, target_of


def load_split(ds, sids, scenes, layers, grid):
    """Cache raw (un-rolled) latents + commands for every clip of these scenes."""
    out = []
    for s in sids:
        for rk in sorted(scenes[s]):
            smp = ds[scenes[s][rk]]
            out.append(dict(s=s, rk=rk, cmd=clip_command(smp),
                            lat={L: fo.to_grid(smp["layers"][L], grid) for L in layers}))
    return out


def rows_for(cmd, tau, order, harm, basis, alpha=None):
    cen, th0, om0, al = cmd
    al = al if alpha is None else alpha
    th = fo.theta_of(th0, om0, al, tau)
    om = fo.omega_of(om0, al, tau)
    return fo.design_matrix(th, om, order=order, harmonics=harm, basis=basis)


def run_config(train, test, layers, grid, tau, order, harm, basis, canon, ridge, n_train, quantity):
    """Fit the operator under one ablation setting and return its held-out latent gate."""
    tr = train[: n_train * 8] if n_train else train
    T = grid[0]
    PHI = np.stack([rows_for(c["cmd"], tau, order, harm, basis) for c in tr]).astype(np.float64)
    Ys = {}
    for L in layers:
        Ys[L] = np.stack([(c["lat"][L] if not canon else fo.canon_roll(c["lat"][L], c["cmd"][0], grid)
                           ).reshape(T, -1) for c in tr]).astype(np.float32)
    C = fo.fit_operator(PHI, Ys, layers, ridge)
    del Ys

    by_scene = {}
    for c in test:
        by_scene.setdefault(c["s"], []).append(c)
    gate = {L: dict(cos=[], shuf=[], magr=[]) for L in layers}
    for s, clips in by_scene.items():
        key = (lambda c: abs(c["cmd"][3] if quantity == "angaccel" else c["cmd"][2]))
        base = min(clips, key=key)
        cen = base["cmd"][0]
        Ha = {L: (fo.canon_roll(base["lat"][L], cen, grid) if canon else base["lat"][L]) for L in layers}
        row_a = rows_for(base["cmd"], tau, order, harm, basis)
        for c in clips:
            if c is base:
                continue
            Hb = {L: (fo.canon_roll(c["lat"][L], cen, grid) if canon else c["lat"][L]) for L in layers}
            row_b = rows_for(c["cmd"], tau, order, harm, basis)
            pred = fo.predict_dH(C, row_a, row_b, layers, grid)
            # shuffled control: sign-flipped command (a genuinely different trajectory)
            _, th0, om0, al = c["cmd"]
            if quantity == "angaccel":
                w_cmd = (base["cmd"][0], th0, om0, -al if abs(al) > 1e-9 else 0.01)
            else:
                w_cmd = (base["cmd"][0], th0, -om0 if abs(om0) > 1e-9 else 0.15, al)
            pred_w = fo.predict_dH(C, row_a, rows_for(w_cmd, tau, order, harm, basis), layers, grid)
            for L in layers:
                true = Hb[L] - Ha[L]
                gate[L]["cos"].append(fo.cosine(pred[L], true))
                gate[L]["shuf"].append(fo.cosine(pred_w[L], true))
                gate[L]["magr"].append(float(np.linalg.norm(pred[L]) / (np.linalg.norm(true) + 1e-30)))
    return {L: dict(cos=float(np.mean(gate[L]["cos"])), cos_shuffled=float(np.mean(gate[L]["shuf"])),
                    mag_ratio=float(np.mean(gate[L]["magr"])), n=len(gate[L]["cos"])) for L in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--quantity", choices=("angvel", "angaccel"), default="angaccel")
    ap.add_argument("--n_train_scenes", type=int, default=125)
    ap.add_argument("--n_test_scenes", type=int, default=30)
    ap.add_argument("--axes", default="order,harmonics,canon,basis,ridge,n_train")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    Q = args.quantity

    tr_ds = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te_ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr_ds[0]["layers"].keys())
    grid = tuple(int(x) for x in tr_ds[0]["grid"])
    F = int(np.asarray(tr_ds[0]["state"]).shape[0])
    tau = vo.frame_token_times(F, grid[0])
    tr_scenes = vo.group_scenes(tr_ds); tr_ids = sorted(tr_scenes)[: args.n_train_scenes]
    te_scenes = vo.group_scenes(te_ds); te_ids = sorted(te_scenes)[: args.n_test_scenes]
    print(f"[ablate:{Q}] caching {len(tr_ids)} train + {len(te_ids)} test scenes, layers={layers} ...", flush=True)
    t0 = time.time()
    train = load_split(tr_ds, tr_ids, tr_scenes, layers, grid)
    test = load_split(te_ds, te_ids, te_scenes, layers, grid)
    print(f"[ablate:{Q}] cached {len(train)}+{len(test)} clips in {time.time()-t0:.0f}s", flush=True)

    D = dict(order=4, harm="all", basis="orientation", canon=True, ridge=10.0, n_train=args.n_train_scenes)
    configs = [dict(axis="default", **D)]
    axes = args.axes.split(",")
    if "order" in axes:
        configs += [dict(D, axis="order", order=o) for o in (0, 1, 2, 3, 6, 8)]
    if "harmonics" in axes:
        configs += [dict(D, axis="harmonics", harm=h) for h in ("even", "odd")]
    if "canon" in axes:
        configs += [dict(D, axis="canon", canon=False)]
    if "basis" in axes:
        configs += [dict(D, axis="basis", basis="orientation_rate")]
    if "ridge" in axes:
        configs += [dict(D, axis="ridge", ridge=r) for r in (0.1, 1.0, 100.0, 1000.0)]
    if "n_train" in axes:
        configs += [dict(D, axis="n_train", n_train=n) for n in (8, 16, 32, 64)
                    if n < args.n_train_scenes]

    results = []
    for i, cf in enumerate(configs):
        t0 = time.time()
        g = run_config(train, test, layers, grid, tau, cf["order"], cf["harm"], cf["basis"],
                       cf["canon"], cf["ridge"], cf["n_train"], Q)
        best = max(g.values(), key=lambda d: d["cos"])
        row = dict(cf); row["gate"] = g
        row["best_cos"] = best["cos"]; row["best_shuffled"] = best["cos_shuffled"]
        results.append(row)
        print(f"  [{i+1}/{len(configs)}] axis={cf['axis']:<10} order={cf['order']} harm={cf['harm']:<4} "
              f"basis={cf['basis']:<17} canon={str(cf['canon']):<5} ridge={cf['ridge']:<6} "
              f"n_train={cf['n_train']:<4} -> L6 cos={g[layers[0]]['cos']:+.3f} "
              f"best cos={best['cos']:+.3f} (shuf {best['cos_shuffled']:+.3f}) [{time.time()-t0:.0f}s]",
              flush=True)
        json.dump(dict(quantity=Q, layers=layers, results=results), open(args.out, "w"), indent=2)

    print(f"\n==================== ABLATION SUMMARY ({Q}, gate = held-out cos vs true dH) ====================")
    print(f"  {'axis':<10} {'setting':<26} {'L6 cos':>8} {'best cos':>9} {'shuffled':>9}")
    for r in results:
        setting = (f"order={r['order']}" if r["axis"] == "order" else
                   f"harmonics={r['harm']}" if r["axis"] == "harmonics" else
                   f"canon={r['canon']}" if r["axis"] == "canon" else
                   f"basis={r['basis']}" if r["axis"] == "basis" else
                   f"ridge={r['ridge']}" if r["axis"] == "ridge" else
                   f"n_train_scenes={r['n_train']}" if r["axis"] == "n_train" else "(operator default)")
        print(f"  {r['axis']:<10} {setting:<26} {r['gate'][layers[0]]['cos']:>+8.3f} "
              f"{r['best_cos']:>+9.3f} {r['best_shuffled']:>+9.3f}")
    print(f"\n[ablate:{Q}] wrote {args.out}")


if __name__ == "__main__":
    main()
