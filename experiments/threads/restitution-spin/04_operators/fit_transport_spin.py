

#!/usr/bin/env python
"""The masked TRANSPORT operator on the crosstalk scene: a per-token edit instead of a global vector.

**Why a different model class, rather than more tuning of the existing one.** Every operator fitted on
this scene so far emits ONE 2,097,152-dim vector as a function of the command. Measured on held-out
scenes by ``resolve_ridge_sweep.py``:

    operator                  achieved     reachable-subspace ceiling
    canon (command-only)        0.387            0.441
    cond128 @ L18               0.666            0.826
    cond128 @ L18, PCA basis    0.689            0.826
    cond128 @ L12               0.687            0.869
    cond512 @ L18               0.669            0.868

Two separate walls are visible there. The ceiling itself is one: no choice of coefficients in that
feature class reaches past ~0.87. And under it sits an estimation gap of ~0.18 that four independent
attacks barely dented -- ridge retuning +0.047, a PCA scene code +0.023, per-scene adaptation +0.060,
reduced-rank truncation and 4x more capacity approximately nothing.

The structural suspicion behind both: a velocity change is mostly a *spatial re-indexing* of the ball's
token pattern -- the same disk written at different places over time -- and a single additive vector is
a poor way to express a translation. It has to average the "ball leaves here, arrives there" pattern
over every start position and every speed the fit saw.

**The transport form says WHERE explicitly and learns only WHAT.** The geometry is handed over as soft
trajectory masks and the operator learns velocity -> channel:

    dH[t,i,j,:] = M_b*(c+_t + v_b @ B+_t) + M_a*(c-_t + v_a @ B-_t) + M_U*((v_b - v_a) @ Bd_t)

This is ``velocity_ops.transport_features`` (8 columns) and is linear in its parameters, so per temporal
token it is an 8-dim ridge onto the 1024-dim token. That is p=8 PER TOKEN against p=411 globally, which
attacks the estimation gap and the ceiling at the same time: it can emit a spatially-varying edit that
no global vector can express, while estimating far fewer parameters.

**The masks must be buildable without the target latent, or this is an oracle.** ``M_a`` comes from
clip a's own per-frame centres. ``M_b`` is FORWARD-SIMULATED from clip a's start under the commanded
``v_b`` -- no ``H_b``, no clip-b state. That is only legitimate if the trajectory is actually linear
here, so it was checked before this script was written: per-frame steps have std 0.00024/0.00038
against mean steps of 0.0063/0.0100, and the largest deviation from a straight line is 0.00064 in
normalized units, i.e. 0.01 of a token cell on the 16x16 grid. The ``oracle`` variant (``M_b`` from clip
b's TRUE centres) is fitted alongside purely as a diagnostic: if deployable is far below oracle, the
mask construction is the problem rather than the operator.

**Blindness is preserved exactly as in ``fit_spin_operators.py``**: only matched-spin pairs are used, so
``d_omega`` is identically zero in every training pair and any spin leakage measured at test time cannot
be an artefact of the fit having seen spin move.

**Scored with the same frame as every other operator here** -- ``latent_crosstalk.py``'s per-scene
Gram-Schmidt frame from the two real displacements -- so ``align``/``gain``/``leak`` are directly
comparable to the table above rather than being a new scale.

    PYTHONPATH=. python experiments/threads/restitution-spin/04_operators/fit_transport_spin.py \
        --train_dir .../latents/spin_ball3d/train/vjepa2_large \
        --test_dir  .../latents/spin_ball3d/test/vjepa2_large \
        --layers 12 --sigmas 0.75,1.0,1.5 --out .../analysis/spin_ball3d/transport_L12.json
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
import json
from pathlib import Path

import sys
from pathlib import Path as _P

import numpy as np

from src.analysis import spin_ops as so
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

P_BASE = vo.TRANSPORT_FEATURE_DIM   # 8; --rich adds 8 more (see _phi)


def _token_centers(sample: dict, n_t: int) -> np.ndarray:
    return vo.temporal_token_centers(vo.clip_positions(sample), n_t)


def _offsets(centers: np.ndarray, grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Signed per-token offset ``(token - ball centre)`` in CELL units, shape ``(T*H*W,)`` each.

    Same image->cell convention as ``gaussian_mask`` (x -> column ``x*(W-1)``, y -> row ``y*(H-1)``), so
    these line up token-for-token with the masks.
    """
    T, H, W = grid
    hh = np.arange(H).reshape(1, H, 1)
    ww = np.arange(W).reshape(1, 1, W)
    w_c = (centers[:, 0] * (W - 1)).reshape(T, 1, 1)
    h_c = (centers[:, 1] * (H - 1)).reshape(T, 1, 1)
    return (np.broadcast_to(ww - w_c, (T, H, W)).reshape(-1).astype(np.float64),
            np.broadcast_to(hh - h_c, (T, H, W)).reshape(-1).astype(np.float64))


def _phi(sample_a, sample_b_centers, va, vb, grid, sigma, rich: bool = False) -> np.ndarray:
    """Per-token transport features for one pair, given the TARGET token centres to write toward.

    ``rich`` adds eight columns to the standard eight. The additions are all mask-GATED, so each stays
    local to the ball's tube and the operator remains a spatially-structured edit rather than drifting
    back toward a global one:

      ``M*dx``, ``M*dy`` (per mask)  -- signed offset from the ball centre. Without these every token
          under a mask receives the same channel vector scaled by the mask, so the operator can only
          write a radially symmetric blob. The real ``dH`` for "the ball left here and arrived there" is
          an ASYMMETRIC pattern -- a leading and a trailing edge -- and an offset feature is the
          cheapest basis that can express one.
      ``M_U`` bare, ``M_U*|dv|``, ``M_b*|vb|``, ``M_a*|va|`` -- speed magnitude terms. The existing
          columns are linear in the signed velocity, so any effect that depends on speed regardless of
          direction (motion blur, tube length) cancels between opposite velocities.

    Still p=16 PER TOKEN against 411 globally, so the data efficiency that makes this class attractive
    is preserved -- the point is a better basis, not a bigger one.
    """
    n_t = grid[0]
    ca = _token_centers(sample_a, n_t)
    M_a = vo.gaussian_mask(ca, grid, sigma)
    M_b = vo.gaussian_mask(sample_b_centers, grid, sigma)
    base = vo.transport_features(M_a, M_b, np.asarray(va), np.asarray(vb), grid)
    if not rich:
        return base
    va_, vb_ = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
    mb = M_b.reshape(-1, 1); ma = M_a.reshape(-1, 1)
    mu = np.maximum(M_a, M_b).reshape(-1, 1)
    dxb, dyb = _offsets(sample_b_centers, grid)
    dxa, dya = _offsets(ca, grid)
    sb, sa = float(np.linalg.norm(vb_)), float(np.linalg.norm(va_))
    sd = float(np.linalg.norm(vb_ - va_))
    extra = np.concatenate([
        mb * dxb.reshape(-1, 1), mb * dyb.reshape(-1, 1),
        ma * dxa.reshape(-1, 1), ma * dya.reshape(-1, 1),
        mu, mu * sd, mb * sb, ma * sa], axis=1)
    return np.concatenate([base, extra], axis=1)


def _phi_xl(sample_a, tgt_centers, va, vb, grid, sigmas, Q, h_base) -> np.ndarray:
    """The wide per-token feature set. See ``--rich 2``.

    **Why this is affordable and the global operator's 411 dims were not.** A per-token model trains on
    pairs x tokens: 12,288 x 2,048 = ~25 MILLION rows. The global operator trains on 12,288 pairs whose
    conditioning code is near-constant within a scene, so its effective sample size is closer to the 256
    SCENES -- which is why four separate estimator improvements moved it by at most +0.06 each. Here p in
    the low hundreds is still a vastly better-conditioned problem than p=411 was there.

    **And the measurement says bias, not variance, is what binds this class**: p=8 transport scored 0.539
    on 8 training scenes and 0.570 on 256 -- 32x the data for +0.03. A model that barely improves with
    data is short of capacity, so capacity is what to spend on.

    Columns, per mask scale:
      presence, velocity, and offset up to SECOND order (dx^2, dy^2, dx*dy). First order can write an
      asymmetric leading/trailing edge; second order can write the elongation and shear that a faster
      ball's tube actually has.
    Plus, once only:
      ``z_i = h_i @ Q``, a q-dim projection of the token's OWN latent, and ``z_i (x) dv``. This is the
      local analogue of the scene code, and it is what lets the edit depend on what is currently at that
      token rather than on the command alone. It reads the BASE clip only -- never ``H_b``.
    """
    T, H, W = grid
    ca = _token_centers(sample_a, T)
    va_ = np.asarray(va, dtype=np.float64); vb_ = np.asarray(vb, dtype=np.float64)
    dv = vb_ - va_
    cols = []
    for sg in sigmas:
        M_a = vo.gaussian_mask(ca, grid, sg).reshape(-1, 1)
        M_b = vo.gaussian_mask(tgt_centers, grid, sg).reshape(-1, 1)
        M_u = np.maximum(M_a, M_b)
        for M, cen, v in ((M_b, tgt_centers, vb_), (M_a, ca, va_)):
            dx, dy = _offsets(cen, grid)
            dx = dx.reshape(-1, 1); dy = dy.reshape(-1, 1)
            cols += [M, M * v[0], M * v[1], M * float(np.linalg.norm(v)),
                     M * dx, M * dy, M * dx * dx, M * dy * dy, M * dx * dy]
        cols += [M_u, M_u * dv[0], M_u * dv[1], M_u * float(np.linalg.norm(dv))]
    z = h_base @ Q                                    # (T*H*W, q) local latent context
    cols += [z, z * dv[0], z * dv[1]]
    return np.concatenate(cols, axis=1)


def _deployable_centers(sample_a, vb, grid) -> np.ndarray:
    """Target centres from clip a's START and the COMMANDED velocity -- never from clip b.

    ``clip_velocity`` is per-frame displacement in normalized units, which is exactly the step
    ``forward_sim_positions`` integrates, so no unit conversion enters here.
    """
    pos_a = vo.clip_positions(sample_a)
    n_frames = pos_a.shape[0]
    sim = vo.forward_sim_positions(pos_a[0], np.asarray(vb), n_frames)
    return vo.temporal_token_centers(sim, grid[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="12")
    ap.add_argument("--sigmas", default="0.75,1.0,1.5")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--num_scenes", type=int, default=0)
    ap.add_argument("--test_scenes", type=int, default=48)
    ap.add_argument("--n_vel", type=int, default=4)
    ap.add_argument("--n_spin", type=int, default=4)
    ap.add_argument("--max_cached_shards", type=int, default=1)
    ap.add_argument("--qdim", type=int, default=8,
                   help="--rich 2 only: dims of the local-latent projection per token.")
    ap.add_argument("--blend_operators", default="",
                   help="a conditioned GLOBAL operators dir (e.g. operators_cond128_L12). If given, the "
                        "held-out evaluation also sweeps the blend ``alpha*e_transport + e_global``.\n"
                        "The two classes fail for OPPOSITE reasons, which is the whole argument for "
                        "blending them: transport is bias-limited (0.539 on 8 training scenes, 0.570 on "
                        "256 -- 32x the data for +0.03; and its capacity curve saturates at ~0.64), "
                        "while the global operator is variance-limited (its span reaches 0.869 but it "
                        "extracts only 0.687, and four separate estimator fixes moved that by at most "
                        "+0.06 each). Errors from a bias-limited and a variance-limited predictor are "
                        "substantially decorrelated, so the blend can exceed both -- and if it does not, "
                        "that is itself evidence they are failing on the SAME component of dH.")
    ap.add_argument("--blend_ridge", type=float, default=1e-4,
                   help="ridge for re-solving the global operator, standardized (see resolve_ridge_sweep)")
    ap.add_argument("--rich", type=int, default=0,
                   help="1 = 16 per-token features instead of 8 (offset-from-centre and speed-magnitude "
                        "terms; see _phi). Still tiny next to the 411-dim global operator.")
    ap.add_argument("--shared_t", type=int, default=1,
                    help="1 = also fit a single B shared across temporal tokens (sum the per-t normal "
                         "equations). With 8x fewer parameters it is the lower-variance variant and on "
                         "this scene the physics is t-invariant, so it may well win.")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x]
    L = layers[0]
    sigmas = [float(x) for x in args.sigmas.split(",") if x]

    ds = LatentDataset(args.train_dir, layers=layers, max_cached_shards=args.max_cached_shards)
    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)
    if args.num_scenes:
        sids = sids[: args.num_scenes]
    print(f"[transport] fitting layer {L} on {len(sids)} scenes, sigmas={sigmas}", flush=True)

    # One accumulator per (sigma, mask-variant, temporal token).
    variants = ("deployable", "oracle")
    q = int(args.qdim)
    if args.rich == 2:
        # 22 columns per mask scale (9 target + 9 source + 4 union), plus 3q for the local-latent block.
        P = 22 * len(sigmas) + 3 * q
    else:
        P = P_BASE + (8 if args.rich else 0)
    # One projection for the local-latent context, fixed seed, reused at eval. A different Q between fit
    # and eval would evaluate the operator off its own basis and read as a confident failure.
    Q = np.random.default_rng(2024).standard_normal((1024, q)) / np.sqrt(1024)
    # --rich 2 consumes ALL scales inside one feature vector, so the per-sigma outer loop would build
    # and fit len(sigmas) identical copies. Collapse it to a single pass in that mode.
    outer_sigmas = [sigmas[0]] if args.rich == 2 else sigmas
    print(f"[transport] p={P} per token (rich={args.rich})", flush=True)
    acc = {(sg, vr, t): vo.LinearLS(P, 1024, args.ridge)
           for sg in outer_sigmas for vr in variants for t in range(8)}
    n_pairs = 0

    for n, s in enumerate(sids):
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        sam = {c: ds[i] for c, i in cells.items()}
        grid = tuple(int(x) for x in sam[(0, 0)]["grid"])
        T, H, W = grid
        HW = H * W
        flat = {c: vo.layer_flat(sam[c]["layers"][L]).reshape(T * HW, 1024) for c in sam}

        # MATCHED-SPIN pairs only: d_omega is identically zero in every training pair.
        for sp in range(args.n_spin):
            for ia in range(args.n_vel):
                for ib in range(args.n_vel):
                    if ia == ib:
                        continue
                    ca, cb = (ia, sp), (ib, sp)
                    va, vb = vo.clip_velocity(sam[ca]), vo.clip_velocity(sam[cb])
                    dH = flat[cb] - flat[ca]                       # (T*HW, 1024)
                    tgt = {"deployable": _deployable_centers(sam[ca], vb, grid),
                           "oracle": _token_centers(sam[cb], T)}
                    for sg in outer_sigmas:
                        for vr in variants:
                            # MUST match the eval site's builder exactly. Fitting one feature set and
                            # evaluating another is the one error here that would not raise -- with
                            # matching widths it would just score a confidently wrong operator.
                            phi = (_phi_xl(sam[ca], tgt[vr], va, vb, grid, sigmas, Q, flat[ca])
                                   if args.rich == 2
                                   else _phi(sam[ca], tgt[vr], va, vb, grid, sg, bool(args.rich)))
                            for t in range(T):
                                sl = slice(t * HW, (t + 1) * HW)
                                acc[(sg, vr, t)].add(phi[sl], dH[sl])
                    n_pairs += 1
        del sam, flat
        if (n + 1) % 20 == 0:
            print(f"[transport] scene {n + 1}/{len(sids)}  pairs={n_pairs}", flush=True)

    # Solve. The shared-across-t variant is the sum of the per-t normal equations, exactly.
    B_per_t, B_shared = {}, {}
    for sg in outer_sigmas:
        for vr in variants:
            def _solve(a):
                d = np.diag(np.diag(a.XtX)).copy()
                d[d <= 0] = 1.0
                return np.linalg.solve(a.XtX + args.ridge * d, a.XtY)
            B_per_t[(sg, vr)] = np.stack([_solve(acc[(sg, vr, t)]) for t in range(8)])  # (8,P,1024)
            if args.shared_t:
                XtX = sum(acc[(sg, vr, t)].XtX for t in range(8))
                XtY = sum(acc[(sg, vr, t)].XtY for t in range(8))
                dsh = np.diag(np.diag(XtX)).copy(); dsh[dsh <= 0] = 1.0
                B_shared[(sg, vr)] = np.linalg.solve(XtX + args.ridge * dsh, XtY)        # (P,1024)
    print("[transport] solved; evaluating on held-out scenes", flush=True)

    # --- held-out evaluation, in latent_crosstalk.py's frame ----------------------------------------
    dte = LatentDataset(args.test_dir, layers=layers, max_cached_shards=2)
    tscenes = vo.group_scenes(dte)
    tids = sorted(tscenes)[: args.test_scenes]
    rows = {k: {"align": [], "gain": [], "leak": []}
            for k in [(sg, vr, m) for sg in outer_sigmas for vr in variants for m in ("per_t", "shared")]}

    # --- the global operator to blend against -------------------------------------------------------
    Bg = Pg = mug = None
    if args.blend_operators:
        gdir = Path(args.blend_operators)
        gmeta = json.loads((gdir / "operators_meta.json").read_text())
        if bool(gmeta.get("canon", False)):
            # A canon operator predicts in the ball-centred frame and would need rolling back per scene;
            # transport predicts in the raw frame. Blending the two without that roll would add an edit
            # placed at the grid centre to one placed at the ball, and the result would be meaningless.
            raise SystemExit("--blend_operators must be a canon=0 fit (transport works in the raw frame)")
        gz = np.load(gdir / "operator_vel.npz")
        gXtX = gz[f"XtX_{L}"].astype(np.float64)
        # UNIFORM ridge, deliberately, and this was a bug worth recording. Standardized ridge scales each
        # feature's penalty by its own XtX diagonal, which rebalances the six-orders-of-magnitude spread
        # in feature scale and helps at moderate lambda (it is what took cond128 @ L18 from 0.666 to
        # 0.689 at lambda 0.01). At SMALL lambda it does the opposite: the conditioning features have
        # diagonals ~0.01, so lambda=1e-4 penalises them by ~1e-6 -- effectively nothing -- and this XtX
        # is rank-deficient (min eigenvalue ~0), so its null space blows up. Measured on 12 test scenes:
        # uniform 1e-4 -> 0.689 (matching the operator's known held-out value), standardized 1e-4 ->
        # 0.125. A uniform penalty keeps a floor on every direction, which is what a singular XtX needs.
        Bg = np.linalg.solve(gXtX + args.blend_ridge * np.eye(gXtX.shape[0]),
                             gz[f"XtY_{L}"]).astype(np.float32)
        Pg = gz[f"P_{L}"].astype(np.float64) if int(gmeta.get("cond_dim", 0)) else None
        mug = gz[f"mu_{L}"].astype(np.float64) if f"mu_{L}" in gz.files else None
        del gz
        print(f"[transport] blending against {gdir.name}: B {Bg.shape}", flush=True)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    blend = {a: {"sel": [], "rep": []} for a in alphas}

    for n, s in enumerate(tids):
        cells = {divmod(int(r), args.n_spin): i for r, i in tscenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        sq = so.commutation_square(cells, 0, 0, args.n_vel - 1, args.n_spin - 1)
        sam = {k: dte[i] for k, i in sq.items()}
        grid = tuple(int(x) for x in sam["base"]["grid"])
        T, H, W = grid; HW = H * W

        def fl(key):
            return vo.layer_flat(sam[key]["layers"][L])

        base = fl("base")
        D_vel = fl("vel_only") - base
        D_spin = fl("spin_only") - base
        nv = np.linalg.norm(D_vel)
        u_v = D_vel / (nv + 1e-12)
        perp = D_spin - (D_spin @ u_v) * u_v
        n_perp = np.linalg.norm(perp)
        u_s = perp / (n_perp + 1e-12)

        va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
        tgt = {"deployable": _deployable_centers(sam["base"], vb, grid),
               "oracle": _token_centers(sam["vel_only"], T)}
        for sg in outer_sigmas:
            for vr in variants:
                phi = (_phi_xl(sam["base"], tgt[vr], va, vb, grid, sigmas, Q,
                               base.reshape(T * HW, 1024))
                       if args.rich == 2
                       else _phi(sam["base"], tgt[vr], va, vb, grid, sg, bool(args.rich)))
                for mode in ("per_t", "shared"):
                    if mode == "shared" and not args.shared_t:
                        continue
                    e = np.empty((T * HW, 1024))
                    for t in range(T):
                        sl = slice(t * HW, (t + 1) * HW)
                        Bt = B_per_t[(sg, vr)][t] if mode == "per_t" else B_shared[(sg, vr)]
                        e[sl] = phi[sl] @ Bt
                    e = e.ravel()
                    en = np.linalg.norm(e)
                    r = rows[(sg, vr, mode)]
                    r["align"].append(float(e @ u_v / (en + 1e-12)))
                    r["gain"].append(float(e @ u_v / (nv + 1e-12)))
                    r["leak"].append(float(e @ u_s / (n_perp + 1e-12)))
                    if (Bg is not None and vr == "deployable" and mode == "per_t"
                            and sg == outer_sigmas[0]):
                        e_t = e / (en + 1e-12)

        if Bg is not None:
            fvg = vo.command_features_pos(va, vb, vo.clip_start_pos(sam["base"]))
            if Pg is not None:
                zc = (base - mug if mug is not None else base) @ Pg
                zc = zc / (np.linalg.norm(zc) + 1e-12)
                fvg = np.concatenate([fvg, zc,
                                      np.outer(zc, np.asarray(vb) - np.asarray(va)).ravel()])
            e_c = fvg @ Bg
            e_c = e_c / (np.linalg.norm(e_c) + 1e-12)
            # Both edits are unit-normalized first, so alpha is a pure MIXING angle rather than a
            # rescaling: alpha=0 is the global operator alone, alpha->inf is transport alone, and every
            # value between is a genuine blend. Without normalizing, alpha would confound the mix with
            # the two operators' very different magnitudes.
            half = len(tids) // 2
            side = "sel" if n < half else "rep"
            for a in alphas:
                d = e_t if a >= 1e8 else (a * e_t + e_c)
                blend[a][side].append(float(d @ u_v / (np.linalg.norm(d) + 1e-12)))

        if (n + 1) % 8 == 0:
            print(f"[transport] test scene {n + 1}/{len(tids)}", flush=True)

    summary = {"layer": L, "ridge": args.ridge, "train_pairs": n_pairs, "rich": bool(args.rich), "p": P,
               "n_test_scenes": len(tids), "sigmas": sigmas, "results": {}}
    for k, v in rows.items():
        if not v["align"]:
            continue
        summary["results"]["|".join(map(str, k))] = {
            m: round(float(np.median(v[m])), 4) for m in ("align", "gain", "leak")}
    Path(args.out).write_text(json.dumps(summary, indent=1))

    print(f"\n# Transport operator, layer {L}, p={P}/token ({len(tids)} held-out scenes, "
          f"{n_pairs} train pairs)\n")
    print("| sigma | masks | B | align | gain | leak |")
    print("|---|---|---|---|---|---|")
    for k in sorted(summary["results"], key=lambda x: -summary["results"][x]["align"]):
        d = summary["results"][k]
        sg, vr, m = k.split("|")
        print(f"| {sg} | {vr} | {m} | {d['align']:.3f} | {d['gain']:.3f} | {d['leak']:+.3f} |")
    print("\nCompare: cond128 @ L12 = 0.687 align (ceiling 0.869); cond128+PCA @ L18 = 0.689 (0.826).")
    print("`deployable` is the honest number -- its target mask is forward-simulated from the command")
    print("and never touches clip b. `oracle` uses clip b's true centres and only diagnoses whether a")
    print("shortfall lives in the mask construction or in the operator.")
    if any(blend[a]["rep"] for a in alphas):
        import numpy as _np
        med = {a: {k: float(_np.median(v)) if v else float("nan") for k, v in blend[a].items()}
               for a in alphas}
        best_a = max(alphas, key=lambda a: med[a]["sel"])
        summary["blend"] = {("inf" if a >= 1e8 else f"{a:g}"): med[a] for a in alphas}
        summary["blend_best_alpha"] = "inf" if best_a >= 1e8 else f"{best_a:g}"
        print("\n# Blend: cos(alpha*e_transport + e_global), both unit-normalized first\n")
        print("| alpha | align (sel) | align (rep) |")
        print("|---|---|---|")
        for a in alphas:
            lab = "inf (transport only)" if a >= 1e8 else ("0 (global only)" if a == 0 else f"{a:g}")
            mark = "  <- best on selection" if a == best_a else ""
            print(f"| {lab} | {med[a]['sel']:.3f} | {med[a]['rep']:.3f} |{mark}")
        print("\nalpha=0 and alpha=inf are the two operators alone; anything above BOTH of them is a")
        print("genuine ensemble gain, and alpha is chosen on the selection half only.")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
