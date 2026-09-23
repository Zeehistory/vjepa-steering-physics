"""Ablations for the inverse action predictor, every one scored by EXECUTION.

Each row of the output table is a full closed loop -- fit the steering map, fit the policy, command
outcomes on held-out TEST scenes, run the strike, measure what came out. Nothing here is an
open-loop proxy, because on this project a good action MAE with a bad realised ``v_out`` is a known
trap.

What is cheap and what is not, which is why the axes are the ones they are:

* **Layer subset and time pooling are FREE.** The cached descriptor is the concatenation over the
  four layers of a (T=8, 4, 4, D=1024) block -- 4 x 131072 = 524288, exactly the cached dimension --
  so a layer subset is a column slice and time-averaging is a reshape-and-mean. No shard re-read.
* **Spatial pool is NOT free.** Changing ``pool`` changes ``pooled_features`` itself and costs a
  full 173 GB shard pass, so it is deliberately out of scope here and called out as such rather
  than quietly omitted.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/07_eval/ablate_action_predictor.py \
        --latent_root .../latents/paddle --output_dir .../action_predictor/paddle_ablations \
        --feat_cache .../action_loop/featcache --lam 1.0
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
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.control.action_predictor import ActionPredictor
from src.data.paddle_strike import build_striker

_CAL_PATH = _REPO_ROOT / "experiments/threads/paddle-robotics/11_run/close_action_loop.py"
_spec = importlib.util.spec_from_file_location("close_action_loop", _CAL_PATH)
cal = importlib.util.module_from_spec(_spec)
sys.modules["close_action_loop"] = cal
assert _spec.loader is not None
_spec.loader.exec_module(cal)

_TAP = _REPO_ROOT / "experiments/threads/paddle-robotics/03_train/train_action_predictor.py"
_spec2 = importlib.util.spec_from_file_location("train_action_predictor", _TAP)
tap = importlib.util.module_from_spec(_spec2)
assert _spec2.loader is not None
_spec2.loader.exec_module(tap)

LAYERS = (6, 12, 18, 23)
TOL = 0.05


def slice_layers(X: np.ndarray, keep: tuple[int, ...], pool: int, t: int = 8,
                 d: int = 1024) -> np.ndarray:
    """Column-slice the pooled descriptor down to a subset of layers."""
    block = t * pool * pool * d
    if X.shape[1] != len(LAYERS) * block:
        raise ValueError(f"descriptor dim {X.shape[1]} is not {len(LAYERS)} x {block}")
    idx = [LAYERS.index(l) for l in keep]
    return np.concatenate([X[:, i * block:(i + 1) * block] for i in idx], axis=1)


def mean_over_time(X: np.ndarray, n_layers: int, pool: int, t: int = 8,
                   d: int = 1024) -> np.ndarray:
    """Collapse the 8 tubelet tokens to their mean, keeping layers and space."""
    return X.reshape(X.shape[0], n_layers, t, pool * pool * d).mean(axis=2).reshape(X.shape[0], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--feat_cache", default=None)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--max_train_post", type=int, default=4000)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--test_seed", type=int, default=2)
    ap.add_argument("--ratios", default="0.75,1.0,1.35,1.7,2.0,2.3,2.6,2.9")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="steering-map ridge penalty, frozen at the main run's selected value so "
                         "every ablation differs in exactly one thing")
    ap.add_argument("--family", default="ridge",
                    help="family for the non-family ablations; ridge is ~100x cheaper and the main "
                         "run reports whether mlp changes the conclusion")
    ap.add_argument("--n_members", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.latent_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_tr = build_striker(args.embodiment, seed=args.train_seed)
    gen_te = build_striker(args.embodiment, seed=args.test_seed)
    ratios = [float(r) for r in args.ratios.split(",")]

    print("[1/3] loading descriptors ...", flush=True)
    fc = Path(args.feat_cache) if args.feat_cache else None
    tr_post = cal.load_split(root / "train_post", gen_tr, "post", args.pool,
                             args.max_train_post, cache_dir=fc)
    tr_pre = cal.load_split(root / "train_pre", gen_tr, "pre", args.pool, cache_dir=fc)
    te_post = cal.load_split(root / "test_post", gen_te, "post", args.pool, cache_dir=fc)
    te_pre = cal.load_split(root / "test_pre", gen_te, "pre", args.pool, cache_dir=fc)

    v_out_tr = np.array([p["v_out"] for p in tr_post["phys"]])
    a_tr_all = np.array([p["v_p"] for p in tr_post["phys"]])
    scene_tr_all = np.array([s for (s, _) in tr_post["keys"]])
    scenes_sorted = sorted(set(scene_tr_all.tolist()))
    n_val = max(8, len(scenes_sorted) // 10)
    val_scenes = sorted(scenes_sorted[:n_val])
    fit_mask_all = np.array([s not in set(val_scenes) for s in scene_tr_all])

    _memo: dict[tuple, float] = {}

    def sim_v_out(gen: Any, v_in_s: float, a: float) -> float:
        key = (id(gen), round(v_in_s, 9), round(a, 9))
        if key not in _memo:
            try:
                _memo[key] = float(gen.simulate(v_in_s, a, render=False)["v_out_world"])
            except RuntimeError:
                _memo[key] = float("nan")
        return _memo[key]

    def run_one(tag: str, *, keep_layers: tuple[int, ...] = LAYERS, time_mean: bool = False,
                k: int | None = None, use_pre: bool = True, use_diff: bool = True,
                source: str = "both", family: str | None = None, n_train: int | None = None,
                noise: float = 0.0, ratio_scale: float = 1.0,
                calibrate: bool = True) -> dict[str, Any]:
        """One complete closed loop under one configuration."""
        family = family or args.family
        k = k or args.k
        Xtr, Xpre, Xte, Xtep = (tr_post["X"], tr_pre["X"], te_post["X"], te_pre["X"])
        if keep_layers != LAYERS:
            Xtr, Xpre, Xte, Xtep = (slice_layers(x, keep_layers, args.pool)
                                    for x in (Xtr, Xpre, Xte, Xtep))
        if time_mean:
            nl = len(keep_layers)
            Xtr, Xpre, Xte, Xtep = (mean_over_time(x, nl, args.pool)
                                    for x in (Xtr, Xpre, Xte, Xtep))

        # cap the TRAINING set by SCENE, never by clip -- a clip cap would keep partial scenes and
        # the sample-efficiency curve would then be measuring something else
        fit_mask = fit_mask_all.copy()
        if n_train is not None:
            keep = set(scenes_sorted[n_val:n_val + n_train])
            fit_mask = np.array([s in keep for s in scene_tr_all])

        red = cal.Reducer(Xtr[fit_mask], k)
        pre_tr = {s: red(Xpre[i][None])[0] for i, (s, _) in enumerate(tr_pre["keys"])}
        pre_te = {s: red(Xtep[i][None])[0] for i, (s, _) in enumerate(te_pre["keys"])}
        Z = red(Xtr[fit_mask])
        Hp = np.stack([pre_tr[s] for (s, _) in np.asarray(tr_post["keys"])[fit_mask]])
        vo, a_f, s_f = v_out_tr[fit_mask], a_tr_all[fit_mask], scene_tr_all[fit_mask]

        W_tgt = cal.ridge_fit(np.concatenate([Hp, vo[:, None]], axis=1), Z, args.lam)
        Hs, Zs, ix = tap.build_targets(Hp, Z, vo, W_tgt, source)
        pi = ActionPredictor.fit(Hs, Zs, a_f[ix], scenes=s_f[ix], family=family,
                                 use_pre=use_pre, use_diff=use_diff,
                                 n_members=args.n_members, seed=args.seed)

        def loop(gen: Any, scenes: list[int], pre: dict, corr: tuple[float, float],
                 rs: float) -> tuple[list, list, list]:
            s_c, c_c = corr
            cmd, ach, err = [], [], []
            rng = np.random.default_rng(args.seed)
            for s in scenes:
                hp = pre[s][None]
                v_in_s = float(gen.scene_params(s)["v_in"])
                for rr in ratios:
                    vt = -rr * rs * v_in_s
                    zt = cal.ridge_apply(np.concatenate([hp, [[(vt - c_c) / s_c]]], axis=1), W_tgt)
                    if noise:
                        # noise SCALED TO THE TARGET's own magnitude, so the perturbation means the
                        # same thing regardless of how the PCA basis happens to be scaled
                        zt = zt + noise * np.linalg.norm(zt) / np.sqrt(zt.size) * rng.normal(size=zt.shape)
                    a_s = float(np.asarray(pi.action_for(hp, zt)).ravel()[0])
                    got = sim_v_out(gen, v_in_s, a_s)
                    cmd.append(vt)
                    ach.append(got)
                    err.append(abs(got - vt) / abs(vt) if np.isfinite(got) else float("inf"))
            return cmd, ach, err

        corr = (1.0, 0.0)
        if calibrate:
            c, a_, _ = loop(gen_tr, val_scenes, pre_tr, (1.0, 0.0), 1.0)
            ok = np.isfinite(a_)
            if ok.sum() > 2:
                sl, ic = np.polyfit(np.asarray(c)[ok], np.asarray(a_)[ok], 1)
                corr = (float(sl), float(ic))

        test_scenes = sorted({s for (s, _) in te_post["keys"]} & set(pre_te))
        _, _, err = loop(gen_te, test_scenes, pre_te, corr, ratio_scale)
        e = np.asarray(err)
        fin = e[np.isfinite(e)]
        row = {"tag": tag, "pass_at_5pct": float((e <= TOL).mean()),
               "pass_at_2pct": float((e <= 0.02).mean()),
               "median": float(np.median(fin)) if len(fin) else float("nan"),
               "p90": float(np.quantile(fin, 0.9)) if len(fin) else float("nan"),
               "val_action_mae": float(pi.meta["val_action_mae"]),
               "n_fit_scenes": int(len(set(s_f.tolist()))), "dim": int(Xtr.shape[1]), "k": int(red.k),
               "cal_slope": corr[0], "cal_intercept": corr[1]}
        print(f"      {tag:34s} pass@5%={row['pass_at_5pct']:6.1%}  median={row['median']:7.2%}  "
              f"actionMAE={row['val_action_mae']:.4f}", flush=True)
        return row

    print("[2/3] running ablations ...", flush=True)
    rows: list[dict[str, Any]] = []

    print("   -- reference --", flush=True)
    rows.append(run_one("reference (all layers, both, ridge)"))

    print("   -- model family --", flush=True)
    for fam in ("ridge", "mlp"):
        rows.append(run_one(f"family={fam}", family=fam))

    print("   -- what the policy is shown --", flush=True)
    rows.append(run_one("input: target only", use_pre=False, use_diff=False))
    rows.append(run_one("input: target+pre", use_diff=False))
    rows.append(run_one("input: target+pre+diff", use_pre=True, use_diff=True))

    print("   -- target source (the decisive knob) --", flush=True)
    for src in ("real", "synth", "both"):
        rows.append(run_one(f"target_source={src}", source=src))

    print("   -- representation --", flush=True)
    for keep in ((23,), (18, 23), (12, 18, 23), LAYERS):
        rows.append(run_one(f"layers={list(keep)}", keep_layers=keep))
    rows.append(run_one("time: mean over 8 tubelets", time_mean=True))
    for k in (8, 16, 32, 64):
        rows.append(run_one(f"k={k}", k=k))

    print("   -- sample efficiency (train SCENES) --", flush=True)
    for n in (25, 50, 100, 250, 450):
        rows.append(run_one(f"train_scenes={n}", n_train=n))

    print("   -- calibration --", flush=True)
    rows.append(run_one("calibration OFF", calibrate=False))
    rows.append(run_one("calibration ON", calibrate=True))

    print("   -- robustness --", flush=True)
    for nz in (0.01, 0.05, 0.10, 0.25):
        rows.append(run_one(f"target noise={nz:.0%}", noise=nz))
    for rs in (1.15, 1.30):
        rows.append(run_one(f"commands {rs:.0%} of envelope (extrapolation)", ratio_scale=rs))

    print("[3/3] writing ...", flush=True)
    (out / "ablations.json").write_text(json.dumps({"rows": rows, "config": vars(args)},
                                                   indent=2, default=float))
    L = ["# Action-predictor ablations", "",
         f"Embodiment `{args.embodiment}`, every row a full closed loop on the held-out TEST split "
         f"(seed {args.test_seed}). Steering-map lam frozen at {args.lam:g}; default family "
         f"`{args.family}`.", "",
         "Spatial `pool` is NOT ablated here: it changes `pooled_features` itself and costs a full "
         "173 GB shard pass, unlike layer subsets and time pooling which are slices of the cache.",
         "",
         "| ablation | pass@5% | pass@2% | median err | p90 | action MAE | dim | k |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['tag']} | **{r['pass_at_5pct']:.1%}** | {r['pass_at_2pct']:.1%} | "
                 f"{r['median']:.2%} | {r['p90']:.2%} | {r['val_action_mae']:.4f} | "
                 f"{r['dim']} | {r['k']} |")
    (out / "ABLATIONS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\nwrote {out}/ABLATIONS.md", flush=True)


if __name__ == "__main__":
    main()
