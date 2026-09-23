

#!/usr/bin/env python
"""POSE-FIELD command-only 2D-ACCELERATION steer -- the acceleration transposition of the SOLVED
Fourier-in-orientation angular-velocity operator (`steer_angvel_fourier.py`).

THE IDEA (user's, 2026-07-15). The angvel operator is NOT linear in the command. It fits a global field
over the exactly-computable POSE (theta(t)=theta0+omega*tau_t), then steers by RE-EVALUATING that field at
the target pose and differencing:  dH = C.(phi(theta_b) - phi(theta_a)).  omega enters ONLY through theta;
the DC column cancels in the difference so appearance drops out for free. Fourier is incidental -- it is
just the basis that linearizes a *periodic* pose.

Transposed to acceleration: **acceleration is not a latent direction, it REPARAMETRIZES a trajectory.**
The exactly-computable pose under constant accel is
    d(t) = p(t) - p0 = v0*tau_t + 0.5*a*tau_t^2      (displacement from the shared start)
    v(t) = v0 + a*tau_t                              (instantaneous velocity)
so we fit a field over s(t)=(d(t),v(t)) rather than a map from the command, and steer by re-evaluation:
    H_canon[t,cell,ch] ~= sum_k C[t,cell,ch,k] * psi_k(s(t))
    dH_canon[t]        =  C[t] . (psi(s_b(t)) - psi(s_a(t)))
Acceleration never appears in the operator; it enters only through the trajectory it induces.

WHY THIS IS NOT JUST ANOTHER COMMAND MAP (the honest framing). At a FIXED token t the map
(v0,a) -> (d(t),v(t)) is a linear bijection, so per-token this is *formally* a nonlinear model in the
command. The win is the INDUCTIVE BIAS, exactly as for angvel: the field is a function of the pose
*jointly*, not of v0 and a separately, so two scenes with different (v0,a) that reach the same pose at the
same t are tied together by one shared field. That is the structure a command-only operator conditioned on
(v_a,v_b,a) alone cannot express -- and every acceleration operator tried so far (incl. curvature) is
linear IN THE COMMAND, which is precisely the assumption that manufactured the false negative on rotation.

BASIS LADDER (--bases). Orientation lives on a compact circle so 9 sinusoids sufficed; displacement lives
on a bounded PLANE and a token cell's response to it is sharply peaked (a cell lights up only when the
object is near it), so a band-limited Fourier basis in d will blur it. Hence the ladder -- local RBF grids
alongside Fourier and polynomial, all fit in ONE streaming pass and scored by the latent gate:
    lin_a    [1, a]                          the linear-in-command baseline (should reproduce ~0.27 cos)
    lin_dv   [1, d, v]                       linear in the pose
    rbf_d    [1, RBF(d)]                     local field over displacement, no velocity
    rbf_dv   [1, RBF(d) (x) [1,v]]           local field, velocity-conditioned          <- lead candidate
    four_dv  [1, Fourier(d) (x) [1,v]]       band-limited field (the literal angvel transposition)
    poly_dv  [1, poly_3(d,v)]                smooth low-order field

LATENT GATE (run FIRST, --gate_only, CPU/bigmem, NO decoder): held-out cosine between the synthesized dH
and the TRUE H_b - H_a, plus a wrong-command control and a magnitude ratio. Pure linear algebra, minutes,
no GPU. It cleanly separates "wrong model class" from "decoder cannot render it" -- the two failure modes
conflated for weeks on angvel. Decode only what the gate clears.

NOT ruled out by the existing accel negatives: the D1 non-identifiable-fiber result concerns the DECODER's
pre-image, but this operator targets the TRUE on-manifold dH (exactly what the gate measures); and D3
warp-invariance says latents are not translation-equivariant -- the field LEARNS the position dependence
rather than assuming equivariance, so it is not killed either.
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

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config

# Physical scales for feature normalisation (from configs/data/moving_ball_scene_accel2d_mixed.yaml:
# v0 in [0.008,0.016]/frame, |a| in [0.0015,0.0035]/frame^2, 16 frames). These only set the ridge's
# column balance; any consistent choice works because fit and predict share them.
D_SCALE = 0.5      # typical |displacement| over a clip (normalised image units)
V_SCALE = 0.03     # typical |velocity| per frame
A_SCALE = 0.0025   # typical |acceleration| per frame^2


def to_grid(arr, grid):
    T, H, W = grid
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(T, H, W, a.size // (T * H * W))


def canon_roll(x, cen, grid, inverse=False):
    """Roll (T,H,W,D) so the start cell -> grid middle (or back). Exact integer roll (invertible).

    Note: within a scene p0 is SHARED by construction, so this roll is identical for clip a and clip b and
    therefore commutes with the within-scene difference. Canonicalisation does not change any single
    scene's edit -- its entire job is to make the field SHARED ACROSS scenes.
    """
    _, H, W = grid
    w = int(np.clip(round(cen[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(cen[1] * (H - 1)), 0, H - 1))
    dh, dw = (H // 2 - h), (W // 2 - w)
    if inverse:
        dh, dw = -dh, -dw
    return np.roll(x, shift=(dh, dw), axis=(1, 2))


# ---------------------------------------------------------------------------------------------------
# Pose bases.  Each maps d (T,2) [image units] and v (T,2) [units/frame] -> design rows (T,P).
# EVERY basis's column 0 is the constant 1 (the DC/appearance column) -- it cancels in the difference
# psi(s_b) - psi(s_a), which is what makes the edit a pure pose edit with no appearance modelling.
# ---------------------------------------------------------------------------------------------------

def _rbf(d, lim, G, width=1.0):
    """(T,2) -> (T,G*G) Gaussian RBFs on a GxG grid of centres over [-lim,lim]^2."""
    c = np.linspace(-lim, lim, G)
    cx, cy = np.meshgrid(c, c, indexing="ij")
    ctr = np.stack([cx.ravel(), cy.ravel()], -1)              # (G*G, 2)
    sig = width * (2.0 * lim / max(G - 1, 1))
    r2 = ((d[:, None, :] - ctr[None, :, :]) ** 2).sum(-1)     # (T, G*G)
    return np.exp(-r2 / (2.0 * sig ** 2))


def _four1d(x, lim, K):
    """(T,) -> (T,1+2K) [1, cos(pi k x/lim), sin(pi k x/lim), ...] (half-period = the box)."""
    out = [np.ones_like(x)]
    for k in range(1, K + 1):
        out.append(np.cos(k * np.pi * x / lim)); out.append(np.sin(k * np.pi * x / lim))
    return np.stack(out, -1)


def _four2d(d, lim, K):
    """(T,2) -> (T,(1+2K)^2) separable 2D Fourier (outer product of the two axes)."""
    fx, fy = _four1d(d[:, 0], lim, K), _four1d(d[:, 1], lim, K)
    return np.einsum("ti,tj->tij", fx, fy).reshape(d.shape[0], -1)


def _poly(z, order):
    """(T,n) -> all monomials up to `order` (including the constant)."""
    from itertools import combinations_with_replacement
    T, n = z.shape
    cols = [np.ones(T)]
    for o in range(1, order + 1):
        for combo in combinations_with_replacement(range(n), o):
            c = np.ones(T)
            for i in combo:
                c = c * z[:, i]
            cols.append(c)
    return np.stack(cols, -1)


def pose_feats(d, v, a, basis, cfg):
    """(T,2) displacement, (T,2) velocity, (2,) accel -> (T,P) design. Column 0 is always the DC term."""
    T = d.shape[0]
    dn, vn = d / D_SCALE, v / V_SCALE
    one = np.ones((T, 1))
    if basis == "lin_a":
        return np.concatenate([one, np.tile((a / A_SCALE)[None, :], (T, 1))], -1)
    if basis == "lin_dv":
        return np.concatenate([one, dn, vn], -1)
    if basis == "rbf_d":
        return np.concatenate([one, _rbf(dn, cfg["rbf_lim"], cfg["rbf_grid"], cfg["rbf_width"])], -1)
    if basis == "rbf_dv":
        R = _rbf(dn, cfg["rbf_lim"], cfg["rbf_grid"], cfg["rbf_width"])
        V = np.concatenate([one, vn], -1)                                  # (T,3)
        return np.concatenate([one, np.einsum("ti,tj->tij", R, V).reshape(T, -1)], -1)
    if basis == "four_dv":
        F = _four2d(dn, cfg["rbf_lim"], cfg["four_k"])
        V = np.concatenate([one, vn], -1)
        return np.concatenate([one, np.einsum("ti,tj->tij", F, V).reshape(T, -1)], -1)
    if basis == "poly_dv":
        return _poly(np.concatenate([dn, vn], -1), cfg["poly_order"])
    # --- HYBRID bases: pose field + the raw whole-clip command ---------------------------------------
    # V-JEPA's global temporal attention makes the accel code HOLISTIC: even Z_0 (frames 0-1, before the
    # path visibly bends) reads whole-clip accel at R^2=0.99, and the curvature analysis found the true
    # edit's large command-predictable component sits in the EARLY slabs. A pure pose field predicts ~0
    # edit there (d(0)~0 and v(0)=v0+0.5a barely differ), so it structurally misses that component. These
    # bases append the constant command `a`, giving a per-token DC-like term that does NOT cancel -- a
    # strict superset of lin_a, so they can only beat it, and the gate then says whether the win comes
    # from the pose part, the holistic part, or needs both.
    if basis == "rbf_dv_a":
        R = _rbf(dn, cfg["rbf_lim"], cfg["rbf_grid"], cfg["rbf_width"])
        V = np.concatenate([one, vn], -1)
        an = np.tile((a / A_SCALE)[None, :], (T, 1))
        return np.concatenate([one, an, np.einsum("ti,tj->tij", R, V).reshape(T, -1)], -1)
    if basis == "four_dv_a":
        Fo = _four2d(dn, cfg["rbf_lim"], cfg["four_k"])
        V = np.concatenate([one, vn], -1)
        an = np.tile((a / A_SCALE)[None, :], (T, 1))
        return np.concatenate([one, an, np.einsum("ti,tj->tij", Fo, V).reshape(T, -1)], -1)
    if basis == "poly_dva":
        an = np.tile(a / A_SCALE, (T, 1))
        return _poly(np.concatenate([dn, vn, an], -1), cfg["poly_order"])
    raise ValueError(f"unknown basis {basis}")


def basis_dim(basis, cfg):
    d = np.zeros((2, 2)); v = np.zeros((2, 2)); a = np.zeros(2)
    return pose_feats(d, v, a, basis, cfg).shape[1]


# ---------------------------------------------------------------------------------------------------
# Clip pose + streaming ridge
# ---------------------------------------------------------------------------------------------------

def _tubelet_pool(x, T):
    """(F,2) per-frame -> (T,2) per latent token, replicating velocity_ops' tubelet pooling EXACTLY."""
    F = x.shape[0]
    step = max(1, F // T)
    out = np.zeros((T, 2))
    for t in range(T):
        sl = slice(t * step, min(F, (t + 1) * step)) if (t + 1) * step <= F or t < T - 1 \
            else slice(t * step, F)
        out[t] = x[sl].mean(0)
    return out


def clip_pose(sample, T, tau):
    """-> (d (T,2), v (T,2), a (2,), p0 (2,), v0 (2,)) for one clip, all from the packed state."""
    p = vo.clip_frame_positions(sample, T)          # (T,2) tubelet-pooled centre
    v = vo.clip_frame_velocities(sample, T)         # (T,2) tubelet-pooled instantaneous velocity
    a = vo.clip_acceleration(sample)                # (2,)  constant over the clip
    p0 = vo.clip_start_pos(sample)                  # (2,)  frame-0 centre (shared within a scene)
    # The sim records vel(f) = v0 + a*f, which is LINEAR in f, so tubelet-pooling commutes with it and
    # v0 = v_pooled[0] - a*tau_0 is exact.
    v0 = v[0] - a * tau[0]
    return p - p0[None, :], v, a, p0, v0


def pose_from_command(v0, a, F, T):
    """Command-only forward model: (v0,a) -> the pose the trajectory WILL have. No H_b, no frames.

    CRITICAL -- match the simulator's DISCRETE roll-out, not the textbook continuous one. moving_ball's
    `_roll_out` steps ``pos <- pos + cur_vel`` with ``cur_vel(f) = v0 + a*f``, so

        vel(f) = v0 + a*f          d(f) = sum_{j<f} vel(j) = v0*f + a*f*(f-1)/2

    i.e. ``d(f) = (v0 - a/2)*f + (a/2)*f^2`` -- the ``-a/2`` term is why the pixel tracker recovers
    ``a = 2*c2`` from a parabola fit. And because d(f) is QUADRATIC in f, tubelet-pooling does NOT commute
    with it: evaluating d at the pooled time tau_t is not the pooled d. So roll out per FRAME and pool with
    the identical slicing. (vel is linear in f, so its pooling is exact either way.) This is the linchpin
    -- `--validate` asserts this reproduces the packed state to ~1e-7.
    """
    f = np.arange(F, dtype=np.float64)
    vel = v0[None, :] + a[None, :] * f[:, None]                       # (F,2)
    d = np.concatenate([np.zeros((1, 2)), np.cumsum(vel, 0)[:-1]], 0)  # (F,2) displacement from p0
    return _tubelet_pool(d, T), _tubelet_pool(vel, T)


def iter_scene_clips(ds, sids, scenes):
    """Yield (sid, rank, idx) grouped BY SHARD, evicting LatentDataset's shard cache as we pass it.

    LatentDataset caches whole opened shards in RAM (~22GB each); streaming 4000 clips without eviction
    OOMs even on bigmem. Visiting shard-contiguously means each shard is opened exactly once.
    """
    want = [(s, r, i) for s in sids for r, i in sorted(scenes[s].items())]
    want.sort(key=lambda x: ds._index[ds._ids[x[2]]])
    cur = None
    for s, r, i in want:
        sh = ds._index[ds._ids[i]]
        if sh != cur:
            ds._shard_cache.clear()
            cur = sh
        yield s, r, i
    ds._shard_cache.clear()


def fit_fields(ds, sids, scenes, layers, grid, tau, bases, cfg, ridge, chunk=64, log_every=400):
    """ONE streaming pass -> {basis: {L: C (T,P,M)}} plus per-basis column scalers.

    Accumulates normal equations A[b][t] (P,P) and B[b][L][t] (P,M) over clips, so memory is O(P*M) and
    NOT O(N*T*M) -- the whole 4000-clip train split never lands in RAM at once.
    """
    T, H, W = grid
    P = {b: basis_dim(b, cfg) for b in bases}
    A = {b: None for b in bases}
    B = {b: {L: None for L in layers} for b in bases}
    PHI_buf = {b: [] for b in bases}
    Y_buf = {L: [] for L in layers}
    n_seen = 0
    t0 = time.time()

    def flush():
        nonlocal PHI_buf, Y_buf
        if not Y_buf[layers[0]]:
            return
        Y = {L: np.stack(Y_buf[L]) for L in layers}            # (n, T, M)
        for b in bases:
            PH = np.stack(PHI_buf[b])                           # (n, T, P)
            for t in range(T):
                X = PH[:, t, :].astype(np.float64)             # (n,P)
                a_t = X.T @ X
                A[b] = np.zeros((T, P[b], P[b])) if A[b] is None else A[b]
                A[b][t] += a_t
                for L in layers:
                    bt = X.T @ Y[L][:, t, :].astype(np.float64)   # (P,M)
                    if B[b][L] is None:
                        B[b][L] = np.zeros((T, P[b], bt.shape[1]), dtype=np.float32)
                    B[b][L][t] += bt.astype(np.float32)
        PHI_buf = {b: [] for b in bases}
        Y_buf = {L: [] for L in layers}

    for s, r, idx in iter_scene_clips(ds, sids, scenes):
        smp = ds[idx]
        d, v, a, p0, v0 = clip_pose(smp, T, tau)
        for b in bases:
            PHI_buf[b].append(pose_feats(d, v, a, b, cfg))
        for L in layers:
            g = canon_roll(to_grid(smp["layers"][L], grid), p0, grid)
            Y_buf[L].append(g.reshape(T, -1))
        n_seen += 1
        if len(Y_buf[layers[0]]) >= chunk:
            flush()
        if n_seen % log_every == 0:
            print(f"[posefield]   streamed {n_seen} clips ({time.time()-t0:.0f}s)", flush=True)
    flush()
    print(f"[posefield] pass done: {n_seen} clips in {time.time()-t0:.0f}s", flush=True)

    # Column scaling from the accumulated second moments: std_j = sqrt(A_jj/n - (mean_j)^2) is not
    # available without the first moment, so use the RMS sqrt(A_jj/n) -- for the DIFFERENCE-based edit any
    # per-column affine rescale is harmless (it cancels), this only balances the ridge penalty.
    C, SC = {}, {}
    for b in bases:
        sc = np.zeros((T, P[b]))
        for t in range(T):
            rms = np.sqrt(np.maximum(np.diag(A[b][t]) / max(n_seen, 1), 1e-12))
            sc[t] = np.where(rms > 1e-6, rms, 1.0)
        SC[b] = sc
        CL = {}
        for L in layers:
            M = B[b][L].shape[2]
            Cb = np.empty((T, P[b], M), dtype=np.float32)
            for t in range(T):
                S = np.diag(1.0 / sc[t])                          # scale columns -> balanced ridge
                As = S @ A[b][t] @ S + ridge * np.eye(P[b])
                Bs = S @ B[b][L][t].astype(np.float64)
                Cb[t] = (S @ np.linalg.solve(As, Bs)).astype(np.float32)   # un-scale back into raw units
            CL[L] = Cb
            B[b][L] = None
        C[b] = CL
        A[b] = None
        print(f"[posefield] fit basis={b:8s} P={P[b]:4d} -> solved {len(layers)} layers", flush=True)
    return C, SC


def predict_dH(C, basis, cfg, v0, a_a, a_b, F, layers, grid):
    """dH_canon[t] = C[t].(psi(s_b(t)) - psi(s_a(t))). Command-only: needs (v0, a_a, a_b) only."""
    T, H, W = grid
    da, va = pose_from_command(v0, a_a, F, T)
    db, vb = pose_from_command(v0, a_b, F, T)
    fa = pose_feats(da, va, a_a, basis, cfg)
    fb = pose_feats(db, vb, a_b, basis, cfg)
    df = fb - fa                                                  # DC column -> exactly 0
    out = {}
    for L in layers:
        CL = C[L]
        dH = np.einsum("tp,tpm->tm", df, CL).astype(np.float32)
        out[L] = dH.reshape(T, H, W, CL.shape[2] // (H * W))
    return out


def cosine(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def save_viz(path, rows, titles, max_frames=8):
    """Filmstrip: one row per condition. The Goodhart check -- a parabola fit to a soft centroid can be
    satisfied by a path-covering SMEAR with no coherent ball in it. On angular velocity a soft tracker
    'converged' perfectly while the decoder emitted a faint pink smear; only the viz exposed it. If the
    steered row does not show a crisp ball on a visibly bent path, the number is not real.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = rows[0].shape[0]
    idx = np.linspace(0, T - 1, min(max_frames, T)).round().astype(int)
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(1.5 * len(idx), 1.7 * len(rows)), squeeze=False)
    for r, (fr, title) in enumerate(zip(rows, titles)):
        for c, fi in enumerate(idx):
            img = np.transpose(fr[fi], (1, 2, 0)).clip(0, 1)
            axes[r][c].imshow(img); axes[r][c].axis("off")
            if c == 0:
                axes[r][c].set_title(title, fontsize=7, loc="left")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--validate_only", action="store_true", help="only check the forward pose model")
    ap.add_argument("--n_validate", type=int, default=16)
    ap.add_argument("--n_train_scenes", type=int, default=500)
    ap.add_argument("--n_test_scenes", type=int, default=30)
    ap.add_argument("--bases", default="lin_a,lin_dv,rbf_d,rbf_dv,four_dv,poly_dv")
    ap.add_argument("--decode_basis", default=None, help="basis to decode (default: best gate)")
    ap.add_argument("--rbf_grid", type=int, default=8)
    ap.add_argument("--rbf_lim", type=float, default=1.6, help="in D_SCALE units (1.6 -> +-0.8 image)")
    ap.add_argument("--rbf_width", type=float, default=1.0)
    ap.add_argument("--four_k", type=int, default=4)
    ap.add_argument("--poly_order", type=int, default=3)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--gains", default="0.5,1,1.5,2,3,4")
    ap.add_argument("--viz_scenes", type=int, default=6)
    ap.add_argument("--viz_tag", default="v1")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg_yaml = load_config(args.config, args.overrides)
    bases = [b for b in args.bases.split(",") if b]
    cfg = dict(rbf_grid=args.rbf_grid, rbf_lim=args.rbf_lim, rbf_width=args.rbf_width,
               four_k=args.four_k, poly_order=args.poly_order)

    tr = LatentDataset(args.train_dir, layers=cfg_yaml.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg_yaml.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    grid = tuple(int(x) for x in tr[0]["grid"])
    T = grid[0]
    F = int(np.asarray(tr[0]["state"]).shape[0])
    tau = vo.frame_token_times(F, T)
    print(f"[posefield] layers={layers} grid={grid} tau={np.round(tau,1)}", flush=True)
    print(f"[posefield] bases: " + ", ".join(f"{b}(P={basis_dim(b,cfg)})" for b in bases), flush=True)

    # ---- LINCHPIN: the command-only forward model must reproduce the packed GT state exactly ----
    # Everything downstream is a difference of psi(pose); if pose_from_command disagrees with the
    # simulator's discrete roll-out, the operator is fitting one trajectory and steering toward another.
    # The angvel analog was validating the rotor tracker on GT frames (<0.2% err) before trusting anything.
    errs_d, errs_v = [], []
    for i in range(min(args.n_validate, len(te))):
        smp = te[i]
        d_gt, v_gt, a, p0, v0 = clip_pose(smp, T, tau)
        d_cmd, v_cmd = pose_from_command(v0, a, F, T)
        errs_d.append(np.abs(d_gt - d_cmd).max()); errs_v.append(np.abs(v_gt - v_cmd).max())
    te._shard_cache.clear()
    ed, ev = float(np.max(errs_d)), float(np.max(errs_v))
    print(f"[posefield] VALIDATE forward model on {len(errs_d)} clips: "
          f"max|d_gt-d_cmd|={ed:.2e}  max|v_gt-v_cmd|={ev:.2e}", flush=True)
    if ed > 1e-5 or ev > 1e-5:
        raise SystemExit(f"[posefield] FATAL: command-only pose model does not match the packed state "
                         f"(d {ed:.2e}, v {ev:.2e}). Fix the roll-out convention before fitting.")
    print("[posefield] forward model EXACT -> pose is genuinely command-computable.", flush=True)
    if args.validate_only:
        return

    tr_scenes = vo.group_scenes(tr); tr_ids = sorted(tr_scenes)[: args.n_train_scenes]
    print(f"[posefield] streaming {len(tr_ids)} train scenes (canon + pose design) ...", flush=True)
    C, SC = fit_fields(tr, tr_ids, tr_scenes, layers, grid, tau, bases, cfg, args.ridge, args.chunk)

    # ---- LATENT GATE on held-out test: cos(pred dH, true dH) vs wrong-command control ----
    te_scenes = vo.group_scenes(te); te_ids = sorted(te_scenes)[: args.n_test_scenes]
    print(f"[posefield] LATENT GATE on {len(te_ids)} test scenes ...", flush=True)
    rng = np.random.default_rng(0)
    all_a = []
    for s in te_ids:
        for r, i in sorted(te_scenes[s].items()):
            all_a.append(vo.clip_acceleration(te[i]))
    gate = {b: {L: dict(cos=[], cos_ctl=[], magr=[]) for L in layers} for b in bases}
    t_gate = time.time()
    for n_s, s in enumerate(te_ids):
        ranks = sorted(te_scenes[s])
        ia = te_scenes[s][ranks[0]]
        sa = te[ia]
        _, _, a_a, p0, v0 = clip_pose(sa, T, tau)
        Ha_c = {L: canon_roll(to_grid(sa["layers"][L], grid), p0, grid) for L in layers}
        for r in ranks[1:]:
            sb = te[te_scenes[s][r]]
            a_b = vo.clip_acceleration(sb)
            Hb_c = {L: canon_roll(to_grid(sb["layers"][L], grid), p0, grid) for L in layers}
            a_wrong = all_a[int(rng.integers(len(all_a)))]     # a real, but WRONG, accel command
            for b in bases:
                pred = predict_dH(C[b], b, cfg, v0, a_a, a_b, F, layers, grid)
                ctl = predict_dH(C[b], b, cfg, v0, a_a, a_wrong, F, layers, grid)
                for L in layers:
                    true_dH = Hb_c[L] - Ha_c[L]
                    gate[b][L]["cos"].append(cosine(pred[L], true_dH))
                    gate[b][L]["cos_ctl"].append(cosine(ctl[L], true_dH))
                    gate[b][L]["magr"].append(float(np.linalg.norm(pred[L]) /
                                                    (np.linalg.norm(true_dH) + 1e-30)))
        te._shard_cache.clear()
        # This stage is memory-bandwidth bound (each predict streams every basis's whole C tensor), so it
        # is far slower than the fit -- log per scene or the run looks hung.
        done = np.mean(gate[bases[0]][layers[0]]["cos"]) if gate[bases[0]][layers[0]]["cos"] else float("nan")
        print(f"  gate scene {n_s+1}/{len(te_ids)} ({time.time()-t_gate:.0f}s, "
              f"running cos[{bases[0]}]={done:+.3f})", flush=True)

    print("\n==================== LATENT GATE (pose-field accel operator) ====================")
    gate_sum = {}
    for b in bases:
        gate_sum[b] = {}
        row = []
        for L in layers:
            c = np.array(gate[b][L]["cos"]); cc = np.array(gate[b][L]["cos_ctl"])
            mr = np.array(gate[b][L]["magr"])
            gate_sum[b][str(L)] = dict(cos=float(c.mean()), cos_control=float(cc.mean()),
                                       mag_ratio=float(mr.mean()), n=int(len(c)))
            row.append(f"L{L}: {c.mean():+.3f}/{cc.mean():+.3f}")
        best = max(gate_sum[b].values(), key=lambda d: d["cos"])
        print(f"  {b:9s} P={basis_dim(b,cfg):4d}  " + "  ".join(row) +
              f"   || best cos={best['cos']:+.3f} (ctl {best['cos_control']:+.3f})")
    print("  format = cos(pred,true dH) / wrong-command control.  Reference: the LINEAR command operators"
          " reconstruct ~0.27; angvel's winning Fourier field gated 0.588 vs 0.209 -> rho 0.94.")
    winner = max(bases, key=lambda b: max(d["cos"] for d in gate_sum[b].values()))
    print(f"  GATE WINNER: {winner}")

    result = dict(bases=bases, cfg=cfg, ridge=args.ridge, layers=layers, gate=gate_sum,
                  n_train_scenes=len(tr_ids), n_test_scenes=len(te_ids), gate_winner=winner)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[posefield] wrote {args.out}")

    # ---- DECODE steer (only if not gate_only) ----
    if not args.gate_only:
        import torch
        from src.analysis.ball_tracking import measured_acceleration
        from src.decoders import build_decoder
        from src.encoders.feature_extractor import latent_collate
        from src.training.checkpoints import load_checkpoint
        dev = args.device
        decode_bases = [b for b in (args.decode_basis or winner).split(",") if b]
        gains = [float(g) for g in args.gains.split(",")]
        rec0 = te.records[0]
        enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
        cfg_yaml.decoder.state_dim = state_dim
        if cfg_yaml.decoder.out_num_frames <= 0:
            cfg_yaml.decoder.out_num_frames = cfg_yaml.data.num_frames
        decoder = build_decoder(cfg_yaml.decoder, enc_dim, state_dim).to(dev).eval()
        if hasattr(decoder, "prime_layers"):
            decoder.prime_layers([int(x) for x in te.available_layers()])
        load_checkpoint(args.checkpoint, decoder, map_location=dev)
        for pm in decoder.parameters():
            pm.requires_grad_(False)
        print(f"\n[posefield] DECODING {decode_bases} on {len(te_ids)} test scenes, gains={gains}", flush=True)

        @torch.no_grad()
        def _dec(latflat, ref):
            lat = {L: torch.from_numpy(np.ascontiguousarray(
                latflat[L].reshape(1, -1, latflat[L].shape[-1]))).to(dev, ref[L].dtype) for L in layers}
            return decoder(lat, grid).frames

        def dec_accel(latflat, ref, want_frames=False):
            fr = _dec(latflat, ref)
            if fr is None:
                return (np.array([np.nan, np.nan]), None) if want_frames else np.array([np.nan, np.nan])
            f0 = fr[0].cpu()
            m = measured_acceleration(f0)
            a = np.array([m["acc_x"], m["acc_y"]])
            return (a, f0.numpy()) if want_frames else a

        perm = np.random.default_rng(7).permutation(len(te_ids))
        val_sids = set(np.array(te_ids)[perm[: len(te_ids) // 2]].tolist())
        all_results = {}
        for db in decode_bases:
          # fresh, IDENTICALLY-seeded rng per basis so every basis faces the SAME wrong-command draws and
          # the SAME val/test split -- the comparison differs only in the basis.
          rng = np.random.default_rng(11)
          rows = []
          print(f"\n[posefield] ---- decoding basis {db} ----", flush=True)
          for n, s in enumerate(te_ids):
              ranks = sorted(te_scenes[s])
              sa = te[te_scenes[s][ranks[0]]]
              _, _, a_a, p0, v0 = clip_pose(sa, T, tau)
              ref = {int(k): v for k, v in latent_collate([sa])["layers"].items() if int(k) in layers}
              Ha = {L: to_grid(sa["layers"][L], grid) for L in layers}
              for r in ranks[1:]:
                  sb = te[te_scenes[s][r]]
                  a_b = vo.clip_acceleration(sb)
                  dH_c = predict_dH(C[db], db, cfg, v0, a_a, a_b, F, layers, grid)
                  edit = {L: canon_roll(dH_c[L], p0, grid, inverse=True) for L in layers}
                  by_g = {g: dec_accel({L: Ha[L] + g * edit[L] for L in layers}, ref).tolist() for g in gains}
                  true_dH = {L: to_grid(sb["layers"][L], grid) - Ha[L] for L in layers}
                  ceil = dec_accel({L: Ha[L] + true_dH[L] for L in layers}, ref).tolist()
                  noop = dec_accel(Ha, ref).tolist()
                  # RANDOM-COMMAND CONTROL, in PIXELS: synthesize the edit for a WRONG target accel and
                  # decode it, but score it against the TRUE a_b. If the operator is genuinely command-
                  # specific this must collapse toward the floor; if it stays good, the decode is not
                  # actually listening to the command and the headline is an artifact.
                  a_w = all_a[int(rng.integers(len(all_a)))]
                  dH_w = predict_dH(C[db], db, cfg, v0, a_a, a_w, F, layers, grid)
                  edit_w = {L: canon_roll(dH_w[L], p0, grid, inverse=True) for L in layers}
                  # swept over the same gains so the control is read at the SAME selected gain as the headline
                  rand_dec = {str(g): dec_accel({L: Ha[L] + g * edit_w[L] for L in layers}, ref).tolist()
                              for g in gains}
                  rows.append(dict(s=int(s), in_val=(s in val_sids), a_b=a_b.tolist(), a_wrong=a_w.tolist(),
                                   by_gain={str(g): by_g[g] for g in gains}, ceiling=ceil, noop=noop,
                                   rand_cmd=rand_dec))
              te._shard_cache.clear()
              print(f"  decode scene {n+1}/{len(te_ids)}", flush=True)

          def ang(u, w):
              u, w = np.asarray(u), np.asarray(w)
              nu, nw = np.linalg.norm(u), np.linalg.norm(w)
              if not np.isfinite(nu) or not np.isfinite(nw) or nu < 1e-12 or nw < 1e-12:
                  return np.nan
              return float(np.degrees(np.arccos(np.clip(u @ w / (nu * nw), -1, 1))))

          def metrics(pairs):
              pairs = [(t, h) for t, h in pairs if np.all(np.isfinite(h))]
              if len(pairs) < 3:
                  return dict(n=len(pairs), angle=float("nan"))
              A_ = [ang(h, t) for t, h in pairs]
              A_ = [x for x in A_ if np.isfinite(x)]
              mt = np.array([np.linalg.norm(t) for t, _ in pairs])
              mh = np.array([np.linalg.norm(h) for _, h in pairs])
              return dict(n=len(pairs), angle=float(np.mean(A_)), angle_median=float(np.median(A_)),
                          mag_ratio=float(np.mean(mh / (mt + 1e-30))),
                          mag_corr=float(np.corrcoef(mt, mh)[0, 1]) if len(mt) > 2 else float("nan"))

          val_err = {}
          for g in gains:
              pr = [(np.array(r["a_b"]), np.array(r["by_gain"][str(g)])) for r in rows if r["in_val"]]
              m = metrics(pr)
              val_err[str(g)] = m["angle"]
          best = min(gains, key=lambda g: (val_err[str(g)] if np.isfinite(val_err[str(g)]) else 1e9))
          ho = [r for r in rows if not r["in_val"]]
          held = metrics([(np.array(r["a_b"]), np.array(r["by_gain"][str(best)])) for r in ho])
          ceil_m = metrics([(np.array(r["a_b"]), np.array(r["ceiling"])) for r in ho])
          noop_m = metrics([(np.array(r["a_b"]), np.array(r["noop"])) for r in ho])
          rand_m = metrics([(np.array(r["a_b"]), np.array(r["rand_cmd"][str(best)])) for r in ho
                            if "rand_cmd" in r])
          # Does the wrong-command decode at least render the accel it WAS given? (if so the operator is
          # faithfully command-driven and the control's badness is specificity, not breakage)
          rand_own = metrics([(np.array(r["a_wrong"]), np.array(r["rand_cmd"][str(best)])) for r in ho
                              if "rand_cmd" in r])
          curve = {str(g): metrics([(np.array(r["a_b"]), np.array(r["by_gain"][str(g)])) for r in ho])
                   for g in gains}

          # ---- VIZ at the SELECTED gain: unsteered / steered / ceiling / ground-truth target clip ----
          if args.viz_scenes > 0 and args.out:
              vdir = Path(args.out).parent / f"viz_{db}_{args.viz_tag}"
              for s in te_ids[: args.viz_scenes]:
                  ranks = sorted(te_scenes[s])
                  sa = te[te_scenes[s][ranks[0]]]
                  _, _, a_a, p0, v0 = clip_pose(sa, T, tau)
                  ref = {int(k): v for k, v in latent_collate([sa])["layers"].items() if int(k) in layers}
                  Ha = {L: to_grid(sa["layers"][L], grid) for L in layers}
                  r = ranks[-1]
                  sb = te[te_scenes[s][r]]
                  a_b = vo.clip_acceleration(sb)
                  dH = predict_dH(C[db], db, cfg, v0, a_a, a_b, F, layers, grid)
                  ed = {L: canon_roll(dH[L], p0, grid, inverse=True) for L in layers}
                  a_no, f_no = dec_accel(Ha, ref, True)
                  a_st, f_st = dec_accel({L: Ha[L] + best * ed[L] for L in layers}, ref, True)
                  a_ce, f_ce = dec_accel({L: to_grid(sb["layers"][L], grid) for L in layers}, ref, True)
                  rws, tts = [], []
                  for fr, tt, av in ((f_no, "unsteered decode(H_a)", a_no),
                                     (f_st, f"STEERED (pose-field, g={best})", a_st),
                                     (f_ce, "ceiling decode(H_b)", a_ce)):
                      if fr is not None:
                          rws.append(fr); tts.append(f"{tt}\n a=({av[0]:+.4f},{av[1]:+.4f})")
                  gt = sb.get("frames", None)
                  if gt is not None:
                      g_np = np.asarray(gt)
                      rws.append(g_np); tts.append(f"GROUND TRUTH clip b\n a=({a_b[0]:+.4f},{a_b[1]:+.4f})")
                  if rws:
                      save_viz(vdir / f"viz_scene{int(s):04d}.png", rws, tts)
                  te._shard_cache.clear()
              print(f"[posefield] wrote filmstrips -> {vdir}", flush=True)
          print("\n==================== POSE-FIELD ACCEL DECODE RESULT ====================")
          print(f"  basis={db}  best_gain={best}  val_angle=" +
                " ".join(f"{g}:{val_err[str(g)]:.1f}" for g in gains))
          print(f"  HELD-OUT @g{best}: angle={held['angle']:.2f}deg median={held['angle_median']:.2f} "
                f"mag_ratio={held['mag_ratio']:.2f} mag_corr={held['mag_corr']:+.2f} n={held['n']}")
          print(f"  ceiling (paste true dH): angle={ceil_m['angle']:.2f}deg mag_corr={ceil_m['mag_corr']:+.2f}")
          print(f"  no-op (do nothing):      angle={noop_m['angle']:.2f}deg")
          print(f"  RANDOM-CMD ctl vs true target: angle={rand_m['angle']:.2f}deg "
                f"mag_corr={rand_m['mag_corr']:+.2f}   <- must COLLAPSE toward the floor")
          print(f"  RANDOM-CMD ctl vs the accel it was GIVEN: angle={rand_own['angle']:.2f}deg "
                f"(low => operator faithfully renders whatever it is commanded)")
          print("  held-out per-gain: " + " ".join(f"g{g}:{curve[str(g)]['angle']:.1f}" for g in gains))
          print("  reference (DIFFERENT protocol -- see the lin_a run for the apples-to-apples baseline):"
                " best LINEAR cmd operator 14.5deg (mag_corr ~-0.04); decoder-in-the-loop TTO 5.07deg"
                " (mag_corr +0.94); no-op floor ~44-49deg.")
          all_results[db] = dict(decode_basis=db, best_gain=best, val_angle=val_err, heldout=held,
                                 ceiling=ceil_m, noop=noop_m, random_command=rand_m,
                                 random_command_own=rand_own, gain_curve=curve, rows=rows)
        result["decode"] = all_results
        print("\n============ APPLES-TO-APPLES SUMMARY (identical scenes/pairs/gains/controls) ============")
        for b_, r_ in all_results.items():
            print(f"  {b_:9s} held-out angle={r_['heldout']['angle']:6.2f}deg  "
                  f"median={r_['heldout']['angle_median']:6.2f}  "
                  f"mag_corr={r_['heldout']['mag_corr']:+.2f}  @g{r_['best_gain']}  "
                  f"|| rand-cmd ctl={r_['random_command']['angle']:6.2f}deg  "
                  f"ceiling={r_['ceiling']['angle']:5.2f}  noop={r_['noop']['angle']:5.2f}")
        if args.out:
            json.dump(result, open(args.out, "w"), indent=2)
            print(f"[posefield] wrote {args.out}")


if __name__ == "__main__":
    main()
