"""Train the inverse action predictor and score it by EXECUTING what it commands.

The question is "what action must the arm take to achieve a commanded velocity change?",
so the deliverable is a pass rate on outcomes the simulator actually produced, not a regression
error on actions. This script therefore does open-loop MAE first (cheap, catches bugs before any
physics runs) and then the number that counts: command an outcome, run the strike, measure what
came out.

Everything is shared with ``close_action_loop.py`` -- the same descriptor cache, the same PCA
reduction, the same steering map ``W_tgt``, the same test scenes, the same memoised simulator. The
only thing that changes is how the action is chosen, which is the point: the comparison against the
existing argmin loop is like-for-like by construction rather than by care.

Arms scored, all on identical test scenes with identical execution:

* ``pred``/``pred_cal``       -- the new inverse policy, uncorrected and with the affine calibration.
* ``argmin``/``argmin_cal``   -- the existing forward-model loop, recomputed here (not quoted from
  its own run) so nothing differs but the action rule.
* ``shuffled``/``shuffled_cal`` -- the policy fed ANOTHER scene's target. The control that matters.
* ``wrong_ratio``             -- own context, target for a different commanded ratio.
* ``analytic``                -- the certified inverse. A ceiling, not a competitor.
* ``constant``                -- the envelope midpoint. What no information scores.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/03_train/train_action_predictor.py \
        --latent_root outputs/paddle_strike/latents/paddle \
        --output_dir  outputs/paddle_strike/action_predictor/paddle \
        --feat_cache  outputs/paddle_strike/action_loop/featcache
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
from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS, build_striker

# close_action_loop lives in scripts/, which is not a package. Import it by path rather than
# duplicating load_split/Reducer/ridge_fit -- a second copy of the descriptor loader is exactly how
# the two would drift apart and stop being comparable.
_CAL_PATH = _REPO_ROOT / "experiments/threads/paddle-robotics/11_run/close_action_loop.py"
_spec = importlib.util.spec_from_file_location("close_action_loop", _CAL_PATH)
cal = importlib.util.module_from_spec(_spec)
sys.modules["close_action_loop"] = cal
assert _spec.loader is not None
_spec.loader.exec_module(cal)

TOL = 0.05
ARMS = ("pred_cal", "pred", "argmin_cal", "argmin", "analytic",
        "wrong_ratio", "shuffled_cal", "shuffled", "constant")


def build_targets(H_pre: np.ndarray, Z_real: np.ndarray, v_out: np.ndarray,
                  W_tgt: np.ndarray, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training targets under one ``target_source`` policy. Returns (H, Z, index-into-rows).

    ``real`` trains on the post clip that actually happened; ``synth`` trains on the steering map's
    reconstruction of it, which is the object test time will hand the policy; ``both`` stacks them.
    See the module docstring of ``action_predictor`` for why this is the decisive knob.
    """
    Z_synth = cal.ridge_apply(np.concatenate([H_pre, v_out[:, None]], axis=1), W_tgt)
    idx = np.arange(len(H_pre))
    if source == "real":
        return H_pre, Z_real, idx
    if source == "synth":
        return H_pre, Z_synth, idx
    if source == "both":
        return (np.concatenate([H_pre, H_pre]), np.concatenate([Z_real, Z_synth]),
                np.concatenate([idx, idx]))
    raise ValueError(f"unknown target_source '{source}'")


def paired_bootstrap(rows: list[dict[str, Any]], arm_a: str, arm_b: str,
                     n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Scene-level paired bootstrap on the pass-rate DIFFERENCE ``arm_a - arm_b``.

    Resampling scenes, not clips: the ranks within a scene share an episode, so treating them as
    independent draws would understate the interval by roughly sqrt(n_ranks).
    """
    scenes = sorted({r["scene"] for r in rows})
    by_scene: dict[int, list[dict[str, Any]]] = {s: [] for s in scenes}
    for r in rows:
        by_scene[r["scene"]].append(r)

    def rate(sel: list[int], arm: str) -> float:
        e = [rr[f"err_{arm}"] for s in sel for rr in by_scene[s]]
        return float(np.mean(np.asarray(e, dtype=float) <= TOL)) if e else float("nan")

    obs = rate(scenes, arm_a) - rate(scenes, arm_b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sel = list(rng.choice(scenes, size=len(scenes), replace=True))
        diffs[b] = rate(sel, arm_a) - rate(sel, arm_b)
    return {"diff": obs, "lo95": float(np.quantile(diffs, 0.025)),
            "hi95": float(np.quantile(diffs, 0.975)),
            "p_le_0": float((diffs <= 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--transfer_root", default=None,
                    help="fit on --latent_root, test on this rendering instead")
    ap.add_argument("--exec_embodiment", default=None,
                    help="EXECUTE the chosen actions on a different rig than the one the latents "
                         "were rendered with -- e.g. 'franka_dynamic' to run the actuated arm. "
                         "Perception stays on the kinematic rendering, which is required: the "
                         "dynamic variant swings from frame 4, inside the context window, so using "
                         "it for perception would leak the action.")
    ap.add_argument("--cal_scenes", type=int, default=0,
                    help="cap the train scenes used for on-rig calibration (0 = all val scenes). "
                         "The actuated arm costs 2.9 s/rollout against the kinematic 0.12 s, so the "
                         "full val sweep is not affordable there and does not need to be.")
    ap.add_argument("--feat_cache", default=None)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--max_train_post", type=int, default=4000)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--test_seed", type=int, default=2)
    ap.add_argument("--ratios", default="0.75,1.0,1.35,1.7,2.0,2.3,2.6,2.9")
    ap.add_argument("--n_grid", type=int, default=400)
    ap.add_argument("--lam", type=float, default=None,
                    help="ridge penalty for W_tgt/W_p; default reuses the loop's selected value")
    ap.add_argument("--lam_grid", default="1.0,10.0,100.0,300.0,1000.0,3000.0,10000.0")
    ap.add_argument("--families", default="ridge,mlp")
    ap.add_argument("--target_sources", default="real,synth,both")
    ap.add_argument("--n_members", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--max_test_scenes", type=int, default=0,
                    help="cap the test scenes (0 = all). Only needed for the actuated arm, where "
                         "the full sweep is 9 arms x 8 ratios x 100 scenes = 7200 rollouts at "
                         "2.9 s each. The cap is applied to EVERY arm equally, so it costs "
                         "statistical power and nothing else.")
    args = ap.parse_args()

    root = Path(args.latent_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen_tr = build_striker(args.embodiment, seed=args.train_seed)
    gen_te = build_striker(args.embodiment, seed=args.test_seed)
    # The EXECUTOR is separable from the renderer that produced the latents. When they differ, the
    # policy still perceives and steers in the kinematic rendering it was trained on, and only the
    # rollout that scores it runs on the other rig -- which is what "does this work on the robot"
    # actually means.
    exec_emb = args.exec_embodiment or args.embodiment
    gen_x_tr = build_striker(exec_emb, seed=args.train_seed) if args.exec_embodiment else gen_tr
    gen_x_te = build_striker(exec_emb, seed=args.test_seed) if args.exec_embodiment else gen_te
    if args.exec_embodiment:
        print(f"      EXECUTOR: perception on '{args.embodiment}', rollouts on '{exec_emb}'",
              flush=True)

    print("[1/7] loading latents + reconstructing physics ...", flush=True)
    fc = Path(args.feat_cache) if args.feat_cache else None
    tr_post = cal.load_split(root / "train_post", gen_tr, "post", args.pool,
                             args.max_train_post, cache_dir=fc)
    tr_pre = cal.load_split(root / "train_pre", gen_tr, "pre", args.pool, cache_dir=fc)
    test_root = Path(args.transfer_root) if args.transfer_root else root
    if args.transfer_root:
        te_emb = test_root.name if test_root.name in ("paddle", "franka") else args.embodiment
        gen_te = build_striker(te_emb, seed=args.test_seed)
        print(f"      TRANSFER: fitting on {root.name}, testing on {test_root.name}", flush=True)
    te_post = cal.load_split(test_root / "test_post", gen_te, "post", args.pool, cache_dir=fc)
    te_pre = cal.load_split(test_root / "test_pre", gen_te, "pre", args.pool, cache_dir=fc)
    if args.transfer_root:
        n_cmp = min(len(tr_post["X"]), len(te_post["X"]))
        d_rel = float(np.linalg.norm(tr_post["X"][:n_cmp] - te_post["X"][:n_cmp])
                      / np.linalg.norm(tr_post["X"][:n_cmp]))
        print(f"      TRANSFER sanity: descriptor relative difference={d_rel:.3e}", flush=True)
        if d_rel < 1e-6:
            raise RuntimeError(f"transfer requested but descriptors are identical ({d_rel:.2e}) -- "
                               "feature cache key collision or a wrong --transfer_root.")
    print(f"      train post={tr_post['n']} pre={tr_pre['n']}  "
          f"test post={te_post['n']} pre={te_pre['n']}", flush=True)
    print(f"      cache/regeneration agreement: train "
          f"{tr_post.get('worst_state_mismatch', float('nan')):.2e}, test "
          f"{te_post.get('worst_state_mismatch', float('nan')):.2e}", flush=True)

    print("[2/7] PCA on TRAIN POST only ...", flush=True)
    red = cal.Reducer(tr_post["X"], args.k)
    print(f"      k={red.k}, variance explained={red.explained:.4f}", flush=True)
    pre_tr = {s: red(tr_pre["X"][i][None])[0] for i, (s, _) in enumerate(tr_pre["keys"])}
    pre_te = {s: red(te_pre["X"][i][None])[0] for i, (s, _) in enumerate(te_pre["keys"])}
    Ztr = red.train_scores().astype(np.float64)
    Hpre_tr = np.stack([pre_tr[s] for (s, _) in tr_post["keys"]])
    v_out_tr = np.array([p["v_out"] for p in tr_post["phys"]])
    a_tr = np.array([p["v_p"] for p in tr_post["phys"]])
    scene_tr = np.array([s for (s, _) in tr_post["keys"]])

    ratios = [float(r) for r in args.ratios.split(",")]
    a_grid = np.linspace(0.30, -1.60, args.n_grid)

    # -- the simulator, memoised once for the whole script -----------------------------------------
    _memo: dict[tuple[float, float], float] = {}

    def sim_v_out(gen: Any, v_in_s: float, a: float) -> float:
        key = (id(gen), round(v_in_s, 9), round(a, 9))
        if key not in _memo:
            try:
                _memo[key] = float(gen.simulate(v_in_s, a, render=False)["v_out_world"])
            except RuntimeError:
                _memo[key] = float("nan")
        return _memo[key]

    # -- lam for the steering map, selected on held-out TRAIN scenes -------------------------------
    scenes_tr = sorted(set(scene_tr.tolist()))
    n_val = max(8, len(scenes_tr) // 10)
    val_scenes, fit_scenes = set(scenes_tr[:n_val]), set(scenes_tr[n_val:])
    fit_mask = np.array([s in fit_scenes for s in scene_tr])

    print(f"[3/7] selecting lam for W_tgt on {len(val_scenes)} held-out train scenes ...", flush=True)
    if args.lam is not None:
        best_lam = float(args.lam)
        lam_scores = {}
        print(f"      lam fixed at {best_lam:g} by --lam", flush=True)
    else:
        # Selection criterion is the END-TO-END error of the predictor loop on val scenes, not the
        # latent reconstruction residual. A W_tgt that reconstructs post latents beautifully but
        # compresses the command axis is exactly the failure the argmin loop hit, and a residual-based
        # criterion cannot see it.
        lam_scores = {}
        for lam in [float(x) for x in args.lam_grid.split(",")]:
            Wt = cal.ridge_fit(np.concatenate([Hpre_tr[fit_mask], v_out_tr[fit_mask, None]], axis=1),
                               Ztr[fit_mask], lam)
            Hs, Zs, ix = build_targets(Hpre_tr[fit_mask], Ztr[fit_mask], v_out_tr[fit_mask],
                                       Wt, "both")
            pi = ActionPredictor.fit(Hs, Zs, a_tr[fit_mask][ix], scenes=scene_tr[fit_mask][ix],
                                     family="ridge", seed=args.seed)
            errs = []
            for s in sorted(val_scenes):
                hp = pre_tr[s][None]
                v_in_s = float(gen_tr.scene_params(s)["v_in"])
                for rr in ratios:
                    vt = -rr * v_in_s
                    zt = cal.ridge_apply(np.concatenate([hp, [[vt]]], axis=1), Wt)
                    a_s = float(np.asarray(pi.action_for(hp, zt)).ravel()[0])
                    got = sim_v_out(gen_tr, v_in_s, a_s)
                    errs.append(abs(got - vt) / abs(vt) if np.isfinite(got) else float("inf"))
            e = np.asarray(errs)
            lam_scores[lam] = float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else float("inf")
            print(f"       lam={lam:<8g} val median end-to-end err={lam_scores[lam]:.2%}", flush=True)
        best_lam = min(lam_scores, key=lam_scores.get)
        print(f"      -> lam={best_lam:g}", flush=True)

    # -- the BASELINE gets its own lam, selected the same way -------------------------------------
    # This is not a courtesy, it is the difference between a comparison and a rigged one. lam trades
    # off two different things for the two methods: the predictor wants a steering map that keeps the
    # command axis sharp (lam=1), while the argmin loop needs heavy shrinkage to keep its forward
    # model smooth enough to have a well-conditioned minimum (the original run selected lam=300).
    # Scoring the baseline at the PREDICTOR's lam drops it from ~95% to ~31%, which would have
    # manufactured most of the headline out of a hyperparameter. Each method now gets the lam that
    # its own end-to-end error on held-out train scenes selects.
    print("[3b/7] selecting lam for the argmin BASELINE, same procedure ...", flush=True)
    lam_am_scores = {}
    for lam in [float(x) for x in args.lam_grid.split(",")]:
        Wt = cal.ridge_fit(np.concatenate([Hpre_tr[fit_mask], v_out_tr[fit_mask, None]], axis=1),
                           Ztr[fit_mask], lam)
        Wp = cal.ridge_fit(cal.build_action_design(a_tr[fit_mask], Hpre_tr[fit_mask]),
                           Ztr[fit_mask], lam)
        errs = []
        for s in sorted(val_scenes):
            hp = pre_tr[s][None]
            v_in_s = float(gen_tr.scene_params(s)["v_in"])
            Zh = cal.ridge_apply(cal.build_action_design(a_grid, np.repeat(hp, len(a_grid), axis=0)), Wp)
            for rr in ratios:
                vt = -rr * v_in_s
                zt = cal.ridge_apply(np.concatenate([hp, [[vt]]], axis=1), Wt)[0]
                a_s = float(a_grid[np.argmin(np.linalg.norm(Zh - zt, axis=1))])
                got = sim_v_out(gen_tr, v_in_s, a_s)
                errs.append(abs(got - vt) / abs(vt) if np.isfinite(got) else float("inf"))
        e = np.asarray(errs)
        lam_am_scores[lam] = float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else float("inf")
        print(f"       lam={lam:<8g} argmin val median err={lam_am_scores[lam]:.2%}", flush=True)
    lam_am = min(lam_am_scores, key=lam_am_scores.get)
    print(f"      -> argmin lam={lam_am:g} (predictor selected {best_lam:g})", flush=True)

    print("[4/7] fitting the steering map W_tgt and the forward map W_p on FIT scenes ...",
          flush=True)
    # Both maps are frozen on the fit scenes and never refit, so the ONE steering map the policy is
    # trained against is the same one it is calibrated against and the same one it is tested with,
    # and none of them has seen a val or test scene. Refitting on all of train before the test pass
    # would be slightly stronger and would also mean the frozen calibration was measured against a
    # different map than the one in use -- not worth the 10% of rows.
    F_tgt = np.concatenate([Hpre_tr[fit_mask], v_out_tr[fit_mask, None]], axis=1)
    W_tgt = cal.ridge_fit(F_tgt, Ztr[fit_mask], best_lam)
    # the baseline's own steering + forward maps, at ITS selected lam
    W_tgt_am = cal.ridge_fit(F_tgt, Ztr[fit_mask], lam_am)
    W_p = cal.ridge_fit(cal.build_action_design(a_tr[fit_mask], Hpre_tr[fit_mask]),
                        Ztr[fit_mask], lam_am)

    print("[5/7] fitting the action predictors ...", flush=True)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    sources = [s.strip() for s in args.target_sources.split(",") if s.strip()]
    preds: dict[str, ActionPredictor] = {}
    openloop: dict[str, dict[str, float]] = {}
    # FIT ROWS ONLY -- not all of train. The next step picks the policy and freezes the affine
    # calibration by running the loop on ``val_scenes``, and those scenes have to be unseen for
    # either number to mean anything. Training on all of train and then "validating" on a subset of
    # it would drive the measured calibration toward the identity (the policy has memorised those
    # scenes, so it looks perfectly calibrated) and the constants carried forward to the test split
    # would be the wrong two numbers. The test split is a different seed and disjoint either way, so
    # this does not change whether the headline is honest -- it changes whether the calibration that
    # produced it was fit on anything real.
    for src in sources:
        Hs, Zs, ix = build_targets(Hpre_tr[fit_mask], Ztr[fit_mask], v_out_tr[fit_mask], W_tgt, src)
        a_fit, s_fit = a_tr[fit_mask], scene_tr[fit_mask]
        for fam in families:
            name = f"{fam}_{src}"
            pi = ActionPredictor.fit(Hs, Zs, a_fit[ix], scenes=s_fit[ix], family=fam,
                                     n_members=args.n_members, seed=args.seed)
            preds[name] = pi
            openloop[name] = {"val_action_mae": pi.meta["val_action_mae"]}
            print(f"      {name:16s} held-out-scene action MAE = "
                  f"{pi.meta['val_action_mae']:.4f} m/s", flush=True)
            pi.save(out / "models" / name)

    # Choose the policy on TRAIN-side end-to-end error. The test split is touched once, below.
    print("[6/7] selecting the policy + affine calibration on held-out TRAIN scenes ...", flush=True)

    def run_val(pi: ActionPredictor, corr: tuple[float, float],
                gen_x: Any = None, scenes: list[int] | None = None) -> tuple[list, list, list]:
        s_c, c_c = corr
        gen_x = gen_x if gen_x is not None else gen_tr
        cmd, ach, errs = [], [], []
        for s in (scenes if scenes is not None else sorted(val_scenes)):
            hp = pre_tr[s][None]
            v_in_s = float(gen_tr.scene_params(s)["v_in"])
            for rr in ratios:
                vt = -rr * v_in_s
                zt = cal.ridge_apply(np.concatenate([hp, [[(vt - c_c) / s_c]]], axis=1), W_tgt)
                a_s = float(np.asarray(pi.action_for(hp, zt)).ravel()[0])
                got = sim_v_out(gen_x, v_in_s, a_s)
                cmd.append(vt)
                ach.append(got)
                errs.append(abs(got - vt) / abs(vt) if np.isfinite(got) else float("inf"))
        return cmd, ach, errs

    def med(errs: list[float]) -> float:
        e = np.asarray(errs)
        return float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else float("inf")

    sel: dict[str, dict[str, Any]] = {}
    for name, pi in preds.items():
        cmd, ach, errs = run_val(pi, (1.0, 0.0))
        ok = np.isfinite(ach)
        s_c, c_c = (np.polyfit(np.asarray(cmd)[ok], np.asarray(ach)[ok], 1)
                    if ok.sum() > 2 else (1.0, 0.0))
        raw, calib = med(errs), med(run_val(pi, (float(s_c), float(c_c)))[2])
        sel[name] = {"raw": raw, "calibrated": calib, "slope": float(s_c), "intercept": float(c_c)}
        print(f"      {name:16s} val median err raw={raw:.2%} cal={calib:.2%} "
              f"(achieved={s_c:.4f}*commanded{c_c:+.4f})", flush=True)
    best_name = min(sel, key=lambda n: sel[n]["calibrated"])
    pi_best = preds[best_name]
    cal_slope, cal_intercept = sel[best_name]["slope"], sel[best_name]["intercept"]
    print(f"      -> policy '{best_name}' (val {sel[best_name]['calibrated']:.2%})", flush=True)

    # ON-RIG RE-IDENTIFICATION. The settled finding on this project is that the strike law's FORM
    # transfers across embodiments but its CONSTANTS do not, so a calibration measured on the
    # kinematic rig is the wrong two numbers for the actuated arm. Re-fit them by measuring what the
    # arm actually delivers on held-out TRAIN scenes -- no ground-truth law, only outcomes, so it is
    # a procedure a real rig can run. The policy itself is NOT refit; only these two numbers are.
    onrig = None
    if args.exec_embodiment:
        cal_sc = sorted(val_scenes)[:args.cal_scenes] if args.cal_scenes else sorted(val_scenes)
        print(f"      re-identifying the calibration ON THE EXECUTOR '{exec_emb}' over "
              f"{len(cal_sc)} train scenes ...", flush=True)
        cmd_x, ach_x, _ = run_val(pi_best, (1.0, 0.0), gen_x=gen_x_tr, scenes=cal_sc)
        okx = np.isfinite(ach_x)
        s_x, c_x = (np.polyfit(np.asarray(cmd_x)[okx], np.asarray(ach_x)[okx], 1)
                    if okx.sum() > 2 else (1.0, 0.0))
        onrig = {"slope": float(s_x), "intercept": float(c_x), "n_scenes": len(cal_sc),
                 "kinematic_slope": cal_slope, "kinematic_intercept": cal_intercept}
        print(f"      on-rig: achieved={s_x:.4f}*commanded{c_x:+.4f}  "
              f"(kinematic was {cal_slope:.4f}/{cal_intercept:+.4f})", flush=True)
        cal_slope, cal_intercept = float(s_x), float(c_x)

    # the argmin loop's own calibration, fitted the same way so the comparison stays fair
    def run_val_argmin(corr: tuple[float, float], gen_x: Any = None,
                       scenes: list[int] | None = None) -> tuple[list, list, list]:
        s_c, c_c = corr
        gen_x = gen_x if gen_x is not None else gen_tr
        cmd, ach, errs = [], [], []
        for s in (scenes if scenes is not None else sorted(val_scenes)):
            hp = pre_tr[s][None]
            v_in_s = float(gen_tr.scene_params(s)["v_in"])
            Zh = cal.ridge_apply(cal.build_action_design(a_grid, np.repeat(hp, len(a_grid), axis=0)), W_p)
            for rr in ratios:
                vt = -rr * v_in_s
                zt = cal.ridge_apply(np.concatenate([hp, [[(vt - c_c) / s_c]]], axis=1), W_tgt_am)[0]
                a_s = float(a_grid[np.argmin(np.linalg.norm(Zh - zt, axis=1))])
                got = sim_v_out(gen_x, v_in_s, a_s)
                cmd.append(vt)
                ach.append(got)
                errs.append(abs(got - vt) / abs(vt) if np.isfinite(got) else float("inf"))
        return cmd, ach, errs

    # The argmin baseline is calibrated on whatever rig it will be SCORED on, exactly like the
    # policy. Giving the new method on-rig numbers and the baseline stale ones would manufacture a
    # win out of the calibration rather than the action rule.
    if args.exec_embodiment:
        cal_sc = sorted(val_scenes)[:args.cal_scenes] if args.cal_scenes else sorted(val_scenes)
        cmd, ach, _ = run_val_argmin((1.0, 0.0), gen_x=gen_x_tr, scenes=cal_sc)
    else:
        cmd, ach, _ = run_val_argmin((1.0, 0.0))
    ok = np.isfinite(ach)
    am_s, am_c = (np.polyfit(np.asarray(cmd)[ok], np.asarray(ach)[ok], 1)
                  if ok.sum() > 2 else (1.0, 0.0))
    print(f"      argmin calibration: achieved={am_s:.4f}*commanded{am_c:+.4f}", flush=True)

    inv = StrikeInverse.fit_constrained(
        sweep_strikes(gen_tr, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20)),
        BALL_MASS, PADDLE_MASS)

    # -- the test split, touched once --------------------------------------------------------------
    print("[7/7] closing the loop on TEST scenes ...", flush=True)
    test_scenes = sorted({s for (s, _) in te_post["keys"]} & set(pre_te))
    if args.max_test_scenes and len(test_scenes) > args.max_test_scenes:
        n_all = len(test_scenes)
        test_scenes = test_scenes[:args.max_test_scenes]
        print(f"      capping test scenes {n_all} -> {len(test_scenes)} "
              f"(--max_test_scenes); the cap applies to every arm equally", flush=True)
    rows: list[dict[str, Any]] = []

    def target_of(hp: np.ndarray, v_star: float, corr: tuple[float, float]) -> np.ndarray:
        s_c, c_c = corr
        return cal.ridge_apply(np.concatenate([hp, [[(v_star - c_c) / s_c]]], axis=1), W_tgt)

    for scene in test_scenes:
        hp = pre_te[scene][None]
        v_in = float(gen_te.scene_params(scene)["v_in"])
        Zh = cal.ridge_apply(cal.build_action_design(a_grid, np.repeat(hp, len(a_grid), axis=0)), W_p)
        for ratio in ratios:
            v_star = -ratio * v_in


            def pred_a(h: np.ndarray, vt: float, corr: tuple[float, float]) -> float:
                return float(np.asarray(pi_best.action_for(h, target_of(h, vt, corr))).ravel()[0])

            def argmin_a(vt: float, corr: tuple[float, float]) -> float:
                s_c, c_c = corr
                zt = cal.ridge_apply(
                    np.concatenate([hp, [[(vt - c_c) / s_c]]], axis=1), W_tgt_am)[0]
                return float(a_grid[np.argmin(np.linalg.norm(Zh - zt, axis=1))])

            other = test_scenes[(test_scenes.index(scene) + 1) % len(test_scenes)]
            hp_o = pre_te[other][None]
            v_star_o = -ratio * float(gen_te.scene_params(other)["v_in"])
            r_wrong = ratios[(ratios.index(ratio) + len(ratios) // 2) % len(ratios)]

            acts = {
                "pred": pred_a(hp, v_star, (1.0, 0.0)),
                "pred_cal": pred_a(hp, v_star, (cal_slope, cal_intercept)),
                "argmin": argmin_a(v_star, (1.0, 0.0)),
                "argmin_cal": argmin_a(v_star, (float(am_s), float(am_c))),
                # the controls run the SAME policy and carry the SAME correction, so a gain from the
                # correction cannot be mistaken for a gain from the policy
                "shuffled": pred_a(hp_o, v_star_o, (1.0, 0.0)),
                "shuffled_cal": pred_a(hp_o, v_star_o, (cal_slope, cal_intercept)),
                "wrong_ratio": pred_a(hp, -r_wrong * v_in, (cal_slope, cal_intercept)),
                "analytic": float(np.asarray(inv.action_for(v_in, v_star, mode="linear")).ravel()[0]),
                "constant": float(np.mean(a_grid)),
            }
            rec: dict[str, Any] = {"scene": scene, "v_in": v_in, "ratio": ratio, "v_star": v_star}
            for name, a in acts.items():
                got = sim_v_out(gen_x_te, v_in, a)
                rec[f"a_{name}"] = a
                rec[f"v_out_{name}"] = got
                rec[f"err_{name}"] = (abs(got - v_star) / abs(v_star)
                                      if np.isfinite(got) else float("inf"))
            rows.append(rec)
        print(f"      scene {scene}: done", flush=True)

    summary: dict[str, Any] = {}
    for name in ARMS:
        e = np.array([r[f"err_{name}"] for r in rows], dtype=float)
        fin = e[np.isfinite(e)]
        vstar = np.array([r["v_star"] for r in rows], dtype=float)
        got = np.array([r[f"v_out_{name}"] for r in rows], dtype=float)
        ok = np.isfinite(got)
        summary[name] = {
            "n": int(len(e)), "n_finite": int(len(fin)),
            "pass_at_2pct": float((e <= 0.02).mean()),
            "pass_at_5pct": float((e <= 0.05).mean()),
            "pass_at_10pct": float((e <= 0.10).mean()),
            "median": float(np.median(fin)) if len(fin) else float("nan"),
            "p90": float(np.quantile(fin, 0.9)) if len(fin) else float("nan"),
            "max": float(fin.max()) if len(fin) else float("nan"),
            "gain": float(np.polyfit(vstar[ok], got[ok], 1)[0]) if ok.sum() > 2 else float("nan"),
            "corr": float(np.corrcoef(vstar[ok], got[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
        }

    boots = {f"{a}_vs_{b}": paired_bootstrap(rows, a, b, args.n_boot, args.seed)
             for a, b in (("pred_cal", "shuffled_cal"), ("pred_cal", "argmin_cal"),
                          ("pred", "argmin"), ("pred_cal", "constant"))}

    res = {"summary": summary, "bootstrap": boots, "rows": rows,
           "openloop": openloop, "selection": sel, "selected_policy": best_name,
           "lam_selection": {"predictor": {str(k): v for k, v in lam_scores.items()},
                             "argmin": {str(k): v for k, v in lam_am_scores.items()},
                             "selected": {"predictor": best_lam, "argmin": lam_am}},
           "config": {"k": red.k, "variance_explained": red.explained, "lam": best_lam,
                      "n_train_post": tr_post["n"], "n_test_scenes": len(test_scenes),
                      "ratios": ratios, "embodiment": args.embodiment,
                      "transfer_root": args.transfer_root, "families": families,
                      "target_sources": sources, "n_members": args.n_members,
                      "exec_embodiment": exec_emb},
           "calibration": {"pred_slope": cal_slope, "pred_intercept": cal_intercept,
                           "argmin_slope": float(am_s), "argmin_intercept": float(am_c),
                           "on_rig": onrig},
           "integrity": {"train_state_mismatch": tr_post.get("worst_state_mismatch"),
                         "test_state_mismatch": te_post.get("worst_state_mismatch")}}
    (out / "action_predictor.json").write_text(json.dumps(res, indent=2, default=float))

    L = ["# A predictor that outputs the action, trained separately and scored by execution", "",
         f"Perception `{args.embodiment}`"
         + (f", tested on `{Path(args.transfer_root).name}`" if args.transfer_root else "")
         + (f", **executed on `{exec_emb}`**" if args.exec_embodiment else "")
         + f". {len(test_scenes)} test scenes (seed {args.test_seed}, scene-disjoint from train), "
         f"commanded ratios {ratios}.", "",
         f"Policy selected on held-out TRAIN scenes: **{best_name}**. "
         f"Latent reduction k={red.k} ({red.explained:.1%} of variance).", "",
         f"Each method gets its OWN steering-map penalty, selected by the same procedure on the "
         f"same held-out train scenes: predictor lam={best_lam:g}, argmin baseline lam={lam_am:g}. "
         f"They differ because the two methods want opposite things from it -- the predictor wants "
         f"the command axis kept sharp, the argmin loop wants a forward model smooth enough to have "
         f"a well-conditioned minimum. Scoring the baseline at the predictor's lam would understate "
         f"it by roughly a factor of three.", "",
         "| arm | pass@2% | pass@5% | pass@10% | median err | p90 | gain | corr |",
         "|---|---|---|---|---|---|---|---|"]
    for name in ARMS:
        s = summary[name]
        L.append(f"| `{name}` | {s['pass_at_2pct']:.1%} | **{s['pass_at_5pct']:.1%}** | "
                 f"{s['pass_at_10pct']:.1%} | {s['median']:.2%} | {s['p90']:.2%} | "
                 f"{s['gain']:+.3f} | {s['corr']:+.3f} |")
    L += ["", "## Paired scene-level bootstrap on the pass@5% difference", "",
          "| comparison | difference | 95% CI | P(diff <= 0) |", "|---|---|---|---|"]
    for k, b in boots.items():
        L.append(f"| {k.replace('_vs_', ' - ')} | {b['diff']:+.1%} | "
                 f"[{b['lo95']:+.1%}, {b['hi95']:+.1%}] | {b['p_le_0']:.4f} |")
    L += ["", "## Open-loop action MAE (held-out train scenes)", "",
          "| policy | action MAE (m/s) |", "|---|---|"]
    for k, v in openloop.items():
        L.append(f"| `{k}` | {v['val_action_mae']:.4f} |")
    L += ["", "`pred`/`pred_cal` is the new inverse policy: it is handed a steered target latent and",
          "returns a continuous action directly, with no grid and no argmin over a forward model.",
          "`argmin`/`argmin_cal` is the existing loop recomputed here on the identical test scenes so",
          "the only difference is the action rule. `shuffled_cal` is the control that matters -- the",
          "same policy with the same correction on ANOTHER scene's target; if it scored comparably the",
          "action would be coming from the context rather than from the edit. `analytic` is the",
          "certified inverse (a ceiling), `constant` is what no information looks like.", ""]
    (out / "ACTION_PREDICTOR.md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L), flush=True)
    print(f"\nwrote {out}/ACTION_PREDICTOR.md and action_predictor.json", flush=True)


if __name__ == "__main__":
    main()
