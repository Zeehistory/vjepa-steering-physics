"""Close the loop: an edited latent belief selects an action that produces the commanded outcome.

Everything before this is readable (probes recover velocity at R^2~0.998) or writable (velocity edits
decode at ~5.7 deg). Both are claims about the latent space talking to itself. This script is the first
one whose output is a physical number::

    H_now    = Enc(pre-contact clip)                    # context: ball at v_in, striker at rest
    H_target = Target(H_now, v*)                        # the outcome, IMAGINED -- no action involved
    H_hat(a) = P(H_now, a)                              # action-conditioned prediction
    a*       = argmin_a || H_hat(a) - H_target ||
    execute a* in MuJoCo, MEASURE the achieved speed    # <- falsifiable, in m/s

**The one thing that makes this non-circular.** ``Target`` may see the context and the commanded speed
and NOTHING else; ``P`` may see the context and a candidate action and nothing else. If the target were
built as ``P(H_now, a_analytic(v*))`` the loop would be a re-derivation of the analytic inverse dressed
up in latents, and it would "succeed" while proving nothing. Concretely, the two models are fit as::

    Target:  h_post ~ A @ h_pre + u * v_out + c        # regress on the OUTCOME
    P:       h_post ~ B @ h_pre + Phi(a) @ W           # regress on the ACTION

Each is an ordinary ridge fit on train scenes. Neither has access to the other's regressor, so the only
route from ``v*`` to ``a*`` runs through latent space -- which is the claim under test.

**Latents are reduced before any of this.** H is 4 layers x 8 x 256 x 1024 ~ 8.4M floats per clip;
regressing that directly on a few hundred scenes would fit noise. A PCA basis is fit on TRAIN post
latents only and applied to everything, so no test clip touches the basis.

**Controls, because a loop that closes is not automatically a loop that works.** Three baselines run
alongside, and the middle one is the one that matters:

* ``analytic`` -- the certified inverse (0.056% median). The ceiling; not a competitor.
* ``shuffled-target`` -- identical machinery, but each scene is given ANOTHER scene's target latent. If
  this scores anywhere near the real loop, the action is being read off the context rather than off the
  edit, and the result is vacuous.
* ``constant`` -- always command the envelope's midpoint. Beats nothing, but it is what "no
  information" looks like on this metric, and reporting it stops a mediocre score from reading as
  meaningful.

Usage::

    PYTHONPATH=. python experiments/threads/paddle-robotics/11_run/close_action_loop.py \
        --latent_root outputs/paddle_strike/latents/paddle \
        --output_dir  outputs/paddle_strike/action_loop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis import velocity_ops as vo
from src.control.strike_inverse import StrikeInverse, sweep_strikes
from src.data.paddle_strike import BALL_MASS, PADDLE_MASS, build_striker
from src.encoders.feature_extractor import LatentDataset

LAYERS = (6, 12, 18, 23)
TOL = 0.05

# Every arm scored on the test split. `_cal` arms carry the train-frozen de-attenuation gain; each has
# an uncorrected twin so the correction's effect is visible rather than assumed.
ARMS = ("latent", "latent_cal", "wrong_ratio", "shuffled", "shuffled_cal", "analytic", "constant")


# -- physics metadata, reconstructed deterministically ---------------------------------------------
# The latent cache stores image-plane `state` but not the world-frame m/s values or the action, because
# extract_latents does not persist `meta`. Rather than change that shared path for every dataset, the
# physics is regenerated here: clip generation is deterministic in (seed, index), so scene_params +
# action_for recover v_in and the action exactly, and simulate(render=False) recovers the measured
# outcome in ~25 ms with no GPU. The reconstruction is then CHECKED against the cached image-plane
# velocity, which is the part that would silently break if the mapping were ever wrong.

def physics_for(gen: Any, scene: int, rank: int) -> dict[str, float]:
    sp = gen.scene_params(scene)
    v_in = float(sp["v_in"])
    ratio = float(sp["ratios"][rank])
    v_p = gen.action_for(v_in, -ratio * v_in)
    r = gen.simulate(v_in, v_p, render=False)
    return {"v_in": v_in, "v_p": float(v_p), "v_out": float(r["v_out_world"]),
            "v_out_img": float(r["v_out_img"]), "ratio": ratio}


def pooled_features(sample: dict, pool: int) -> np.ndarray:
    """Flatten the 4-layer latent state, average-pooling the SPATIAL grid to ``pool`` x ``pool``.

    Reduction at load is not optional. One clip's full state is 4 layers x 8 x 256 x 1024 = 8.4M
    floats = 67 MB in float64, so even 800 clips is 54 GB and the first attempt was SIGKILLed. Nor can
    it be deferred to a bigger node: the full train split would be ~134 GB in float32.

    Pooling SPACE while keeping TIME at full resolution is the right axis to give up, because velocity
    lives in the frame-to-frame change and the post clips are constructed to share a start position --
    so absolute spatial detail is the least informative part of this particular representation. Time is
    left untouched at all 8 tubelet tokens, and all four layers are kept, so the descriptor is still the
    multi-scale state the rest of the project steers, just spatially coarser. At pool=4 a clip is 2 MB.
    """
    T, H, W = sample["grid"]
    parts = []
    for li in LAYERS:
        x = np.asarray(sample["layers"][li], dtype=np.float32)     # (T*H*W, D)
        D = x.shape[-1]
        x = x.reshape(T, H, W, D)
        if pool and (H % pool == 0) and (W % pool == 0):
            bh, bw = H // pool, W // pool
            x = x.reshape(T, pool, bh, pool, bw, D).mean(axis=(2, 4))
        parts.append(x.reshape(-1))
    return np.concatenate(parts)


def load_split(root: Path, gen: Any, window: str, pool: int = 4,
               max_clips: int | None = None, check: bool = True,
               cache_dir: Path | None = None) -> dict[str, Any]:
    """Load one latent cache and pair every clip with its reconstructed physics.

    Pooled descriptors are cached to ``cache_dir``: reading 173 GB of shards takes ~11 minutes, and
    every rerun to retune k or the ridge penalty would otherwise pay it again. The cache key includes
    pool, the clip cap and the seed, so a changed reduction or a differently-seeded generator cannot
    silently reuse the wrong descriptors.
    """
    if cache_dir is not None:
        # The EMBODIMENT (root.parent.name) belongs in the key. Without it, paddle/test_post and
        # franka/test_post both key on "test_post" and the transfer run silently reuses whichever was
        # cached first -- which produced a "paddle -> franka" result identical to the paddle-only one in
        # every digit. Nothing else catches this: the integrity gate compares regenerated PHYSICS, and
        # the two embodiments share physics to 2.5e-12 m/s deliberately, so it validated at 1.86e-09
        # while comparing the wrong latents entirely.
        tag = (f"{root.parent.name}_{root.name}_pool{pool}_n{max_clips or 'all'}"
               f"_seed{gen.seed}.npz")
        cpath = cache_dir / tag
        if cpath.exists():
            z = np.load(cpath, allow_pickle=True)
            print(f"      {root.name}: reusing cached descriptors ({cpath.name})", flush=True)
            return {"X": z["X"], "keys": [tuple(k) for k in z["keys"]],
                    "phys": list(z["phys"]), "n": int(z["n"]),
                    **({"worst_state_mismatch": float(z["worst"])} if "worst" in z else {})}
    ds = LatentDataset(root, layers=list(LAYERS))
    feats, keys, phys = [], [], []
    worst_check = 0.0
    cur_shard = None
    for i in range(len(ds)):
        if max_clips is not None and len(keys) >= max_clips:
            break
        sid = ds._ids[i]
        sr = vo.scene_rank(sid)
        if sr is None:
            continue
        scene, rank = sr
        # Drop the previous shard only when CROSSING into a new one. Clearing after every sample
        # instead re-reads the whole ~4 GB shard once per clip -- 128x the necessary I/O per shard --
        # which looks like a hang rather than an error. Ids are enumerated in shard order, so tracking
        # the current shard is enough to keep exactly one resident.
        shard = ds._index[sid]
        if shard != cur_shard:
            ds._shard_cache.clear()
            cur_shard = shard
        s = ds[i]
        feats.append(pooled_features(s, pool))
        keys.append((scene, rank))
        p = physics_for(gen, scene, 0 if window == "pre" else rank)
        phys.append(p)
        if check and window == "post":
            # the cached clip's own image-plane velocity must agree with the regenerated episode --
            # this is what would catch a wrong index -> (scene, rank) mapping
            v_img_cached = float(np.asarray(vo.clip_velocity(s))[0])
            worst_check = max(worst_check, abs(v_img_cached - p["v_out_img"]))
        if (len(keys) % 200) == 0:
            print(f"        {root.name}: {len(keys)} clips", flush=True)
    out = {"X": np.stack(feats), "keys": keys, "phys": phys, "n": len(keys)}
    if check and window == "post":
        out["worst_state_mismatch"] = worst_check
        # HARD GATE, not a printout. The physics is regenerated rather than read from the cache, so the
        # one thing that can go wrong silently is regenerating a DIFFERENT episode than the one that was
        # encoded -- every latent would then be paired with the wrong action and outcome, and the loop
        # would report a number that means nothing. This caught exactly that: the test split is
        # extracted with seed=2, and reconstructing it with the default seed=0 gave 8.5e-03 here against
        # 1.6e-09 for the correctly-seeded train split.
        if worst_check > 1e-6:
            raise RuntimeError(
                f"{root.name}: regenerated episodes disagree with the cache by {worst_check:.2e} in "
                f"image-plane velocity (tolerance 1e-6). The generator's seed almost certainly does not "
                f"match the seed this cache was extracted with.")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / (f"{root.parent.name}_{root.name}_pool{pool}"
                              f"_n{max_clips or 'all'}_seed{gen.seed}.npz"),
                 X=out["X"], keys=np.array(out["keys"]), phys=np.array(out["phys"], dtype=object),
                 n=out["n"], worst=out.get("worst_state_mismatch", 0.0))
    return out


# -- reduction --------------------------------------------------------------------------------------

class Reducer:
    """Mean-centre + PCA, fit on TRAIN POST latents only, via the GRAM matrix.

    ``np.linalg.svd(Xc, full_matrices=False)`` is the obvious spelling and it does not fit in memory:
    with n=1600 and d=524288 it returns ``Vt`` at (1600, 524288), which is 6.7 GB in the float64 svd
    always upcasts to, on top of a 6.7 GB float64 copy of ``Xc`` itself.

    The Gram route gets the same answer inside a few hundred MB. Eigendecomposing ``G = Xc Xc^T``
    (n x n = 1600 x 1600) gives eigenvectors ``U`` and eigenvalues ``lam``; the principal directions
    follow as ``V_k = Xc^T U_k / sqrt(lam_k)``, so only the k=64 components are ever formed (524288 x 64
    = 134 MB) rather than all 1600. Train scores come straight from ``U_k sqrt(lam_k)``, and anything
    else projects with ``(X - mu) @ V_k``.
    """

    def __init__(self, X: np.ndarray, k: int) -> None:
        self.mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        Xc = X - self.mu                                     # stays float32
        G = (Xc @ Xc.T).astype(np.float64)                   # (n, n)
        lam, U = np.linalg.eigh(G)
        order = np.argsort(lam)[::-1]
        lam, U = lam[order], U[:, order]
        total = float(lam.clip(min=0).sum())
        # drop numerically-null directions before taking k, so a component is never built by dividing
        # by a near-zero singular value
        keep = int((lam > max(lam[0], 0.0) * 1e-10).sum()) if lam.size else 0
        self.k = int(min(k, keep))
        lam_k, U_k = lam[: self.k], U[:, : self.k]
        self.basis = (Xc.T @ (U_k / np.sqrt(lam_k))).T.astype(np.float32)   # (k, d), orthonormal rows
        self.sv = np.sqrt(lam_k)
        self.explained = float(lam_k.sum() / total) if total > 0 else float("nan")
        self._train_scores = (U_k * np.sqrt(lam_k)).astype(np.float32)

    def train_scores(self) -> np.ndarray:
        return self._train_scores

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) @ self.basis.T).astype(np.float64)


def ridge_fit(F: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an unpenalised intercept; returns W with a bias row appended."""
    n = F.shape[0]
    Fb = np.concatenate([F, np.ones((n, 1))], axis=1)
    d = Fb.shape[1]
    R = lam * np.eye(d)
    R[-1, -1] = 0.0
    return np.linalg.solve(Fb.T @ Fb + R, Fb.T @ Y)


def ridge_apply(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.concatenate([F, np.ones((F.shape[0], 1))], axis=1) @ W


def action_features(a: np.ndarray, h_pre: np.ndarray) -> np.ndarray:
    """Features of a candidate action, given the context.

    Quadratic in ``a`` because the strike map, while linear to 0.106% of range in the WORLD, need not be
    linear in latent coordinates. The ``a * (leading context component)`` term lets the action's effect
    depend on the inflow speed, which is the physical coupling (v_out = alpha*a + beta*v_in) expressed
    without ever being told v_in.
    """
    a = np.atleast_1d(a).astype(np.float64)
    lead = h_pre[:, :1] if h_pre.ndim == 2 else h_pre[None, :1]
    lead = np.broadcast_to(lead, (a.shape[0], 1))
    return np.stack([a, a * a], axis=1).astype(np.float64) * 1.0, lead


def build_action_design(a: np.ndarray, H_pre: np.ndarray) -> np.ndarray:
    """[h_pre | a | a^2 | a*h_pre_lead] for a batch of (context, action) pairs."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 1)
    lead = H_pre[:, :1]
    return np.concatenate([H_pre, a, a * a, a * lead], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_root", required=True,
                    help="dir containing train_post/ train_pre/ test_post/ test_pre/")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--embodiment", default="paddle")
    ap.add_argument("--transfer_root", default=None,
                    help="test on a DIFFERENT rendering than the one fit on (appearance transfer)")
    ap.add_argument("--k", type=int, default=64, help="PCA components")
    ap.add_argument("--lam", type=float, default=1.0, help="ridge penalty (overridden by --lam_grid)")
    ap.add_argument("--lam_grid", default="0.001,0.01,0.1,1.0,10.0,100.0",
                    help="ridge penalties to select among, on held-out TRAIN scenes")
    ap.add_argument("--ratios", default="0.75,1.0,1.5,2.0,2.5,2.9",
                    help="commanded |v_out|/v_in values to test")
    ap.add_argument("--n_grid", type=int, default=241, help="action grid points for the argmin")
    ap.add_argument("--pool", type=int, default=4,
                    help="average-pool the 16x16 spatial token grid to this size (see pooled_features)")
    ap.add_argument("--max_train_post", type=int, default=None,
                    help="cap train post clips; the ridge fits need hundreds, not thousands")
    # These MUST match the seeds the caches were extracted with (see slurm_extract_paddle_strike.sh);
    # the integrity gate in load_split fails loudly rather than quietly mispairing latents if they don't.
    ap.add_argument("--feat_cache", default=None,
                    help="dir for cached pooled descriptors; skips the ~11 min shard read on reruns")
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--test_seed", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.latent_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ONE GENERATOR PER SPLIT, seeded to match how that cache was extracted. scene_params draws v_in and
    # the 8 ratios from a seed-derived RNG, so a generator with the wrong seed silently describes a
    # different episode for the same scene id. The splits are scene-disjoint BECAUSE the seeds differ,
    # so a single shared generator cannot be right for both.
    gen_tr = build_striker(args.embodiment, seed=args.train_seed)
    gen_te = build_striker(args.embodiment, seed=args.test_seed)

    print("[1/6] loading latents + reconstructing physics ...", flush=True)
    fc = Path(args.feat_cache) if args.feat_cache else None
    tr_post = load_split(root / "train_post", gen_tr, "post", args.pool, args.max_train_post, cache_dir=fc)
    tr_pre = load_split(root / "train_pre", gen_tr, "pre", args.pool, cache_dir=fc)
    # EMBODIMENT TRANSFER. With --transfer_root the models are fit on `root`'s rendering and tested on
    # a DIFFERENT one. This is the clean appearance-transfer test the two embodiments exist for: the
    # physics is identical between them to 2.5e-12 m/s, so any drop is attributable to appearance alone,
    # with the dynamics held exactly fixed. The generator stays the base embodiment either way, because
    # the physics -- and therefore every reconstructed action and outcome -- is the same object.
    test_root = Path(args.transfer_root) if args.transfer_root else root
    if args.transfer_root:
        # Reconstruct the TEST split's physics with a generator of the TEST rendering's own embodiment.
        # The two embodiments are meant to share physics exactly, and they nearly do -- but not to the
        # last bit: over 18 (scene, rank) pairs at seed 2, 17 agree to 1e-12..1e-10 m/s while one
        # differs by 4.9e-03 m/s (0.16%), with identical contact frames. The likely cause is the
        # constraint solver taking a different path with the arm's 7 extra DOFs in the solve; the module
        # docstring already records that solver tolerance dominates this comparison. Regenerating each
        # rendering with its own generator keeps the integrity gate at a tight 1e-6 instead of loosening
        # it to 1e-3 and losing the ability to catch a genuine seed mismatch (which shows up at 8.5e-03).
        te_emb = test_root.name if test_root.name in ("paddle", "franka") else args.embodiment
        gen_te = build_striker(te_emb, seed=args.test_seed)
        print(f"      TRANSFER: fitting on {root.name}, testing on {test_root.name} "
              f"(test physics regenerated with the {te_emb} generator)", flush=True)
    te_post = load_split(test_root / "test_post", gen_te, "post", args.pool, cache_dir=fc)
    te_pre = load_split(test_root / "test_pre", gen_te, "pre", args.pool, cache_dir=fc)
    if args.transfer_root:
        n_cmp = min(len(tr_post["X"]), len(te_post["X"]))
        same = np.allclose(tr_post["X"][:n_cmp], te_post["X"][:n_cmp])
        d_rel = float(np.linalg.norm(tr_post["X"][:n_cmp] - te_post["X"][:n_cmp])
                      / np.linalg.norm(tr_post["X"][:n_cmp]))
        print(f"      TRANSFER sanity: fit-vs-test descriptor relative difference={d_rel:.3e}",
              flush=True)
        if same or d_rel < 1e-6:
            raise RuntimeError(
                "transfer requested but the test descriptors are identical to the fit ones "
                f"(rel diff {d_rel:.2e}) -- the two renderings must differ. Almost certainly a feature "
                "cache key collision or a wrong --transfer_root.")
    print(f"      descriptor dim={tr_post['X'].shape[1]} (pool={args.pool}), "
          f"{tr_post['X'].nbytes / 1e9:.2f} GB for train post", flush=True)
    print(f"      train post={tr_post['n']} pre={tr_pre['n']}  "
          f"test post={te_post['n']} pre={te_pre['n']}", flush=True)
    print(f"      cache/regeneration agreement (image-plane v): "
          f"train {tr_post.get('worst_state_mismatch', float('nan')):.2e}, "
          f"test {te_post.get('worst_state_mismatch', float('nan')):.2e}", flush=True)

    print("[2/6] fitting PCA on TRAIN POST only ...", flush=True)
    red = Reducer(tr_post["X"], args.k)
    print(f"      k={red.k}, variance explained={red.explained:.4f}", flush=True)

    # index the pre latents by scene so a post clip can find its own context
    pre_tr = {s: red(tr_pre["X"][i][None])[0] for i, (s, _) in enumerate(tr_pre["keys"])}
    pre_te = {s: red(te_pre["X"][i][None])[0] for i, (s, _) in enumerate(te_pre["keys"])}
    Ztr = red.train_scores().astype(np.float64)

    Hpre_tr = np.stack([pre_tr[s] for (s, _) in tr_post["keys"]])
    v_out_tr = np.array([p["v_out"] for p in tr_post["phys"]])
    a_tr = np.array([p["v_p"] for p in tr_post["phys"]])

    # -- choose the ridge penalty on HELD-OUT TRAIN scenes ----------------------------------------
    # lam=1.0 was an arbitrary default and it showed: the first run had achieved-vs-commanded gain
    # 0.878 with correlation 0.977, i.e. the loop tracked the command well but systematically
    # undershot. That is textbook ridge attenuation -- shrinkage pulls the velocity direction in
    # Target toward zero, so H_target for an extreme v* is not extreme enough and the argmin returns a
    # too-timid action. Selecting lam by the actual end-to-end loop error fixes the right thing.
    #
    # Selection uses train scenes ONLY, split fit/val, and the executed outcome on val. The test split
    # is touched exactly once, after lam is frozen.
    scenes_tr = sorted({s for (s, _) in tr_post["keys"]})
    n_val = max(8, len(scenes_tr) // 10)
    val_scenes, fit_scenes = set(scenes_tr[:n_val]), set(scenes_tr[n_val:])
    fit_mask = np.array([s in fit_scenes for (s, _) in tr_post["keys"]])
    ratios_sel = [float(r) for r in args.ratios.split(",")]
    a_grid_sel = np.linspace(0.30, -1.60, args.n_grid)

    # -- de-attenuation gain -----------------------------------------------------------------------
    # Selecting lam alone cannot fix this, and the first run's per-ratio breakdown shows why. The
    # action bias was -0.079 at ratio 0.75, ~0 at 2.0 and +0.069 at 2.9 -- monotone in the command and
    # crossing zero at the MEAN command. That is the signature of ridge attenuating the v_out column of
    # Target: z_target(h, v*) is pulled toward its value at the MEAN command, so an extreme one is not asked
    # for extremely enough and the argmin returns a too-timid action at BOTH ends. Decisive evidence
    # that it is not a perception failure: the shuffled control carried the SAME bias to three decimals
    # (-0.0795 vs -0.0793 etc.), so it is a property of the command axis, not of the scene.
    #
    # How much of the loop this accounts for, measured on the previous run's rows: regressing the
    # loop's action on the analytic one gives slope 0.9013 (paddle) and 0.6837 (franka transfer) at
    # R^2 = 0.995 and 0.998. So the action the loop returns is an almost perfectly LINEAR but shrunk
    # copy of the correct action, and removing that one distortion cuts the action residual from 0.048
    # to 0.026 m/s on the paddle and from 0.129 to 0.012 -- eleven-fold -- on the transfer. The franka
    # transfer was therefore never an information failure; it is a calibration failure.
    #
    # The correction is affine (slope AND intercept: the intercepts above are -0.050 and -0.181, so a
    # pure gain would not do it), and it is fitted against MEASURED outcomes rather than the analytic
    # inverse. That matters: measuring what came out is something a rig can do, whereas fitting to the
    # true law is not, so this stays a procedure that transfers. Train scenes only; the control arms
    # carry the identical correction so the comparison stays honest.

    # simulate() is the entire cost of selection and the action grid is discrete, so the same
    # (v_in, a) pair recurs constantly across the lam grid and the raw/calibrated passes. Memoising
    # makes the second pass nearly free.
    _sim_memo: dict[tuple[float, float], float] = {}

    def sim_v_out(gen: Any, v_in_s: float, a: float) -> float:
        key = (round(v_in_s, 9), round(a, 9))
        if key not in _sim_memo:
            try:
                _sim_memo[key] = float(gen.simulate(v_in_s, a, render=False)["v_out_world"])
            except RuntimeError:
                _sim_memo[key] = float("nan")
        return _sim_memo[key]

    def run_val(lam: float, corr: tuple[float, float]) -> tuple[list[float], list[float], list[float]]:
        """Run the loop on val scenes under correction ``corr``; return commanded, achieved, rel err."""
        s_c, c_c = corr
        Wt = ridge_fit(np.concatenate([Hpre_tr[fit_mask], v_out_tr[fit_mask, None]], axis=1),
                       Ztr[fit_mask], lam)
        Wp = ridge_fit(build_action_design(a_tr[fit_mask], Hpre_tr[fit_mask]), Ztr[fit_mask], lam)
        cmd, ach, errs = [], [], []
        for s in sorted(val_scenes):
            hp = pre_tr[s][None]
            v_in_s = float(gen_tr.scene_params(s)["v_in"])
            Zh = ridge_apply(build_action_design(a_grid_sel,
                                                np.repeat(hp, len(a_grid_sel), axis=0)), Wp)
            for rr in ratios_sel:
                vt = -rr * v_in_s
                v_eff = (vt - c_c) / s_c
                zt = ridge_apply(np.concatenate([hp, [[v_eff]]], axis=1), Wt)[0]
                a_s = float(a_grid_sel[np.argmin(np.linalg.norm(Zh - zt, axis=1))])
                got = sim_v_out(gen_tr, v_in_s, a_s)
                cmd.append(vt)
                ach.append(got)
                errs.append(abs(got - vt) / abs(vt) if np.isfinite(got) else float("inf"))
        return cmd, ach, errs

    def median_err(errs: list[float]) -> float:
        e = np.array(errs)
        return float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else float("inf")

    lam_grid = [float(x) for x in args.lam_grid.split(",")]
    print(f"[2b/6] selecting lam + affine calibration on {len(val_scenes)} held-out TRAIN scenes "
          f"(fit on {len(fit_scenes)}) ...", flush=True)
    IDENT = (1.0, 0.0)
    lam_scores, cal_scores, cal_by_lam = {}, {}, {}
    for lam in lam_grid:
        cmd, ach, errs = run_val(lam, IDENT)
        lam_scores[lam] = median_err(errs)
        # Calibrate against MEASURED outcomes, exactly as a rig would: regress achieved on commanded,
        # then command the pre-image. This needs no ground-truth strike law -- only the ability to
        # measure what came out -- so it is a procedure that survives contact with hardware, unlike a
        # correction fitted against the analytic inverse.
        ok = np.isfinite(ach)
        s_c, c_c = np.polyfit(np.array(cmd)[ok], np.array(ach)[ok], 1) if ok.sum() > 2 else IDENT
        cal_by_lam[lam] = (float(s_c), float(c_c))
        cal_scores[lam] = median_err(run_val(lam, (s_c, c_c))[2])
        print(f"       lam={lam:<8g} val median err: raw={lam_scores[lam]:.2%} "
              f"calibrated={cal_scores[lam]:.2%}  (slope={s_c:.4f} intercept={c_c:+.4f})", flush=True)
    best_lam = min(cal_scores, key=cal_scores.get)
    cal_slope, cal_intercept = cal_by_lam[best_lam]
    print(f"       -> lam={best_lam:g}, calibration achieved={cal_slope:.4f}*commanded"
          f"{cal_intercept:+.4f} (val median {cal_scores[best_lam]:.2%})", flush=True)
    args.lam = best_lam

    print("[3/6] fitting Target (sees v_out, never the action) ...", flush=True)
    F_tgt = np.concatenate([Hpre_tr, v_out_tr[:, None]], axis=1)
    W_tgt = ridge_fit(F_tgt, Ztr, args.lam)
    r_tgt = float(np.linalg.norm(ridge_apply(F_tgt, W_tgt) - Ztr) / np.linalg.norm(Ztr))
    print(f"      train relative residual={r_tgt:.4f}", flush=True)

    print("[4/6] fitting P (sees the action, never v_out) ...", flush=True)
    F_p = build_action_design(a_tr, Hpre_tr)
    W_p = ridge_fit(F_p, Ztr, args.lam)
    r_p = float(np.linalg.norm(ridge_apply(F_p, W_p) - Ztr) / np.linalg.norm(Ztr))
    print(f"      train relative residual={r_p:.4f}", flush=True)

    print("[5/6] fitting the analytic inverse (ceiling baseline) ...", flush=True)
    inv = StrikeInverse.fit_constrained(
        sweep_strikes(gen_tr, [0.85, 0.95, 1.05, 1.15], np.linspace(0.25, -1.45, 20)),
        BALL_MASS, PADDLE_MASS)
    print(f"      alpha={inv.alpha:.5f} beta={inv.beta:.5f}", flush=True)

    print("[6/6] closing the loop on TEST scenes ...", flush=True)
    lo, hi = gen_te.ratio_range
    a_grid = np.linspace(0.30, -1.60, args.n_grid)
    ratios = [float(r) for r in args.ratios.split(",")]
    test_scenes = sorted({s for (s, _) in te_post["keys"]} & set(pre_te))
    rows: list[dict[str, Any]] = []

    for scene in test_scenes:
        h_pre = pre_te[scene][None]                      # (1, k)
        v_in = float(gen_te.scene_params(scene)["v_in"])
        # every candidate action's predicted latent, for this context
        Z_hat = ridge_apply(build_action_design(a_grid, np.repeat(h_pre, len(a_grid), axis=0)), W_p)
        for ratio in ratios:
            v_star = -ratio * v_in
            z_target = ridge_apply(np.concatenate([h_pre, [[v_star]]], axis=1), W_tgt)[0]
            a_star = float(a_grid[np.argmin(np.linalg.norm(Z_hat - z_target, axis=1))])

            # the de-attenuated loop: identical machinery, command replaced by the pre-image under the
            # affine calibration frozen on train scenes above
            v_eff = (v_star - cal_intercept) / cal_slope
            z_cal = ridge_apply(np.concatenate([h_pre, [[v_eff]]], axis=1), W_tgt)[0]
            a_cal = float(a_grid[np.argmin(np.linalg.norm(Z_hat - z_cal, axis=1))])

            # CONTROL 1 -- another SCENE's target at the same ratio. Kept for continuity, but it is a
            # weak null and the first run proved it: scenes differ only in v_in in [0.85, 1.15], so at a
            # fixed ratio the other scene's command is within ~30% of this one, while the commanded
            # range across ratios spans -0.64 to -3.33. It was asking almost the same question, which is
            # why it scored 9.55% against the loop's 8.19% and made a working loop look dead.
            other = test_scenes[(test_scenes.index(scene) + 1) % len(test_scenes)]
            v_star_other = -ratio * float(gen_te.scene_params(other)["v_in"])
            z_shuf = ridge_apply(
                np.concatenate([pre_te[other][None], [[v_star_other]]], axis=1), W_tgt)[0]
            a_shuf = float(a_grid[np.argmin(np.linalg.norm(Z_hat - z_shuf, axis=1))])

            # the control gets the SAME correction. Without this the calibrated loop would be compared
            # against an uncorrected null and any gain would look like evidence for the loop when it is
            # really evidence for the gain.
            v_eff_other = (v_star_other - cal_intercept) / cal_slope
            z_shuf_cal = ridge_apply(
                np.concatenate([pre_te[other][None], [[v_eff_other]]], axis=1), W_tgt)[0]
            a_shuf_cal = float(a_grid[np.argmin(np.linalg.norm(Z_hat - z_shuf_cal, axis=1))])

            # CONTROL 2 -- the real null: this scene's own context, but the target for a DIFFERENT
            # commanded RATIO. Ratio is the dominant axis of the command, so mismatching it is a genuine
            # mismatch. If the loop scores like this, the action is not coming from the edit.
            r_wrong = ratios[(ratios.index(ratio) + len(ratios) // 2) % len(ratios)]
            z_wrong = ridge_apply(
                np.concatenate([h_pre, [[-r_wrong * v_in]]], axis=1), W_tgt)[0]
            a_wrong = float(a_grid[np.argmin(np.linalg.norm(Z_hat - z_wrong, axis=1))])

            a_ana = float(np.asarray(inv.action_for(v_in, v_star, mode="linear")).ravel()[0])
            a_const = float(np.mean(a_grid))

            rec: dict[str, Any] = {"scene": scene, "v_in": v_in, "ratio": ratio, "v_star": v_star}
            for name, a in (("latent", a_star), ("latent_cal", a_cal), ("wrong_ratio", a_wrong),
                            ("shuffled", a_shuf), ("shuffled_cal", a_shuf_cal),
                            ("analytic", a_ana), ("constant", a_const)):
                try:
                    got = float(gen_te.simulate(v_in, a, render=False)["v_out_world"])
                    err = abs(got - v_star) / abs(v_star)
                except RuntimeError:
                    got, err = float("nan"), float("inf")
                rec[f"a_{name}"] = a
                rec[f"v_out_{name}"] = got
                rec[f"err_{name}"] = err
            rows.append(rec)
        print(f"      scene {scene}: done", flush=True)

    summary = {}
    for name in ARMS:
        e = np.array([r[f"err_{name}"] for r in rows], dtype=float)
        fin = e[np.isfinite(e)]
        summary[name] = {
            "n": int(len(e)), "n_finite": int(len(fin)),
            "pass_at_5pct": float((e <= TOL).mean()),
            "median": float(np.median(fin)) if len(fin) else float("nan"),
            "p90": float(np.quantile(fin, 0.9)) if len(fin) else float("nan"),
            "max": float(fin.max()) if len(fin) else float("nan"),
        }

    for name in ARMS:
        vstar = np.array([r["v_star"] for r in rows], dtype=float)
        got = np.array([r[f"v_out_{name}"] for r in rows], dtype=float)
        ok = np.isfinite(got)
        if ok.sum() > 2:
            summary[name]["gain"] = float(np.polyfit(vstar[ok], got[ok], 1)[0])
            summary[name]["corr"] = float(np.corrcoef(vstar[ok], got[ok])[0, 1])

    res = {"summary": summary, "rows": rows,
           "config": {"k": red.k, "variance_explained": red.explained, "lam": args.lam,
                      "n_train_post": tr_post["n"], "n_train_pre": tr_pre["n"],
                      "n_test_scenes": len(test_scenes), "ratios": ratios,
                      "embodiment": args.embodiment},
           "lam_selection": {str(k): v for k, v in lam_scores.items()},
           "calibration_selection": {str(k): {"raw": lam_scores[k], "calibrated": v,
                                              "slope": cal_by_lam[k][0],
                                              "intercept": cal_by_lam[k][1]}
                                     for k, v in cal_scores.items()},
           "fit": {"lam_selected": best_lam, "cal_slope": cal_slope, "cal_intercept": cal_intercept,
                   "target_residual": r_tgt, "predictor_residual": r_p,
                   "alpha": inv.alpha, "beta": inv.beta},
           "integrity": {"train_state_mismatch": tr_post.get("worst_state_mismatch"),
                         "test_state_mismatch": te_post.get("worst_state_mismatch")}}
    (out / "action_loop.json").write_text(json.dumps(res, indent=2, default=float))

    L = ["# Closing the action loop: does an edited belief cause the commanded outcome?", "",
         f"Test scenes: {len(test_scenes)} (seed 2, scene-disjoint from train). "
         f"Commanded ratios: {ratios}. Bar: relative error <= {TOL:.0%}.", "",
         f"Latent reduction: k={red.k} PCA components on train post only "
         f"({red.explained:.1%} of variance).", "",
         "| arm | pass@5% | median err | p90 | max | gain | corr |",
         "|---|---|---|---|---|---|---|"]
    for name in ("latent_cal", "latent", "analytic", "wrong_ratio", "shuffled_cal", "shuffled",
                 "constant"):
        s = summary[name]
        L.append(f"| {name} | {s['pass_at_5pct']:.1%} | {s['median']:.2%} | "
                 f"{s['p90']:.2%} | {s['max']:.2%} | {s.get('gain', float('nan')):+.3f} | "
                 f"{s.get('corr', float('nan')):+.3f} |")
    L += ["", "`latent_cal` is the loop under test: the same machinery, with the command replaced by its",
          f"pre-image under the affine calibration achieved = {cal_slope:.4f}*commanded "
          f"{cal_intercept:+.4f},",
          "fitted on held-out TRAIN scenes against MEASURED outcomes only (no ground-truth law, so the",
          "same two numbers are obtainable on a rig). `latent` is the identical loop uncorrected.",
          "`analytic` is the certified",
          "inverse -- the ceiling, not a competitor. `shuffled_cal`/`shuffled` run the identical",
          "machinery on ANOTHER scene's target and carry the identical correction: they are the controls",
          "that matter, because if they scored comparably the action would be coming from the context",
          "rather than from the edit. `constant` is what no information looks like.", "",
          "Read `latent_cal` against `shuffled_cal`, never against the uncorrected control -- the",
          "correction is applied to both precisely so the gain cannot be mistaken for the loop working.",
          ""]
    (out / "ACTION_LOOP.md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L[:14]), flush=True)
    print(f"\nwrote {out}/ACTION_LOOP.md and action_loop.json", flush=True)


if __name__ == "__main__":
    main()
