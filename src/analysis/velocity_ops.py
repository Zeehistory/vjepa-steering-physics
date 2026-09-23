"""Velocity subspace / operator toolkit (Step 2, PI direction 2026-06-27).

The speed-only result proved that a *per-pair, same-scene* difference vector ``Delta = H_b - H_a`` steers
faithfully, but a single *global* averaged vector does not generalize — the velocity factor is localized
to the spatial tokens the ball occupies, so it lands in different tokens per scene. This module builds the
machinery to go from a per-instance edit to a TRANSFERABLE velocity representation:

* **PCA of Delta H** (within-scene + global): is the local velocity subspace low-rank (~2D = velocity's
  two degrees of freedom)? how much higher is the global rank (the token-misalignment cost)?
* **subspace projection** ``P_U(Delta)``: does projecting the difference onto a learned velocity subspace
  ``U`` preserve the steer, while a random same-rank subspace destroys it?
* **ridge operator** ``F_U: Delta v -> Delta H`` fit at the full flattened dimension by streaming normal
  equations (only a 2x2 inverse), so we can steer straight from a velocity command.
* **canonicalization** ``A_s``: roll the token grid so the ball's start cell is canonical, regress in
  canonical coordinates, map back — to recover cross-scene generalization.

Latents come from :class:`encoders.feature_extractor.LatentDataset`: each sample's ``layers[L]`` is an
``(L_tok, D)`` tensor with ``L_tok = T' * Hp * Wp`` tokens (grid ``(T', Hp, Wp) = (8, 16, 16)``) and
``D = 1024``. Velocity ground truth is read from the packed ``obj0_vel_x/obj0_vel_y`` state columns.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_SCENE_RE = re.compile(r"^scene(\d+)_v(\d+)$")


def scene_rank(sample_id: str) -> tuple[int, int] | None:
    m = _SCENE_RE.match(sample_id)
    return (int(m.group(1)), int(m.group(2))) if m else None


def clip_velocity(sample: dict) -> np.ndarray:
    """Ground-truth 2D velocity of a clip = mean of its (obj0_vel_x, obj0_vel_y) state columns."""
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    vx = state[:, keys.index("obj0_vel_x")].mean()
    vy = state[:, keys.index("obj0_vel_y")].mean()
    return np.array([float(vx), float(vy)])


def clip_acceleration(sample: dict) -> np.ndarray:
    """Ground-truth constant 2D acceleration of a clip = mean of its (obj0_acc_x, obj0_acc_y) columns.

    The acceleration analog of :func:`clip_velocity` (Step 2 "Bravo"). For ``scene_accel2d`` the
    acceleration is constant over the clip, so the mean recovers it exactly; every velocity scenario
    leaves these columns at 0, so this returns ``[0, 0]`` there (harmless).
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    ax = state[:, keys.index("obj0_acc_x")].mean()
    ay = state[:, keys.index("obj0_acc_y")].mean()
    return np.array([float(ax), float(ay)])


def clip_angvel(sample: dict) -> np.ndarray:
    """Ground-truth ANGULAR velocity of a clip, embedded as a 2-vector ``[omega, 0]``.

    Angular velocity is a signed SCALAR (sign = CW/CCW, magnitude = spin rate), but the whole
    subspace / ridge / command-feature machinery (``RidgeOperator`` dim 2, ``command_features``) is written
    for 2-vectors, so we embed ``omega`` as ``[omega, 0]``. This lets the entire cmd-U8 accel pipeline
    (``accel_subspace.py`` / ``fit_command_operators_accel.py``) fit the angular-velocity operator unchanged
    (``--quantity angvel``); the ``steer_angvel2d.py`` decode reports the proper scalar metrics. Reads the
    constant ``obj0_omega`` state column (0 for every non-rotation scenario, so this returns ``[0, 0]``).
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    w = state[:, keys.index("obj0_omega")].mean()
    return np.array([float(w), 0.0])


def clip_angvel0(sample: dict) -> float:
    """INITIAL angular velocity ``omega(0)`` of a clip (scalar, rad/frame).

    :func:`clip_angvel` averages the ``obj0_omega`` column, which is the right summary for the
    constant-spin ``scene_angvel2d`` family but NOT for ``scene_angaccel2d``, where ``omega(t)`` ramps and
    the mean is ``omega0 + alpha*(F-1)/2``. Orientation roll-outs need the initial rate specifically, so
    read row 0. For any constant-spin clip this equals :func:`clip_angvel`'s first component.
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    return float(state[0, keys.index("obj0_omega")])


def clip_angaccel(sample: dict) -> np.ndarray:
    """Ground-truth constant ANGULAR ACCELERATION of a clip, embedded as a 2-vector ``[alpha, 0]``.

    The angular-acceleration analog of :func:`clip_angvel`, embedded the same way (angular acceleration is
    a signed SCALAR, but the subspace / ridge / command machinery is written for 2-vectors) so the whole
    cmd-U8 pipeline fits it unchanged. Reads the constant ``obj0_alpha`` state column, which is 0 for every
    non-angular-acceleration scenario (so this returns ``[0, 0]`` there). Returns ``[0, 0]`` for latents
    encoded before ``obj0_alpha`` joined the schema, keeping older caches readable.
    """
    keys = list(sample["state_keys"])
    if "obj0_alpha" not in keys:
        return np.array([0.0, 0.0])
    state = np.asarray(sample["state"])
    a = state[:, keys.index("obj0_alpha")].mean()
    return np.array([float(a), 0.0])


def clip_start_pos(sample: dict) -> np.ndarray:
    """Frame-0 ball center (normalized [0,1]) from (obj0_pos_x, obj0_pos_y)."""
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    return np.array([float(state[0, keys.index("obj0_pos_x")]),
                     float(state[0, keys.index("obj0_pos_y")])])


def clip_frame_positions(sample: dict, T: int) -> np.ndarray:
    """Ball center per LATENT temporal token (T, 2), pooling the video frames of each tubelet.

    V-JEPA tubelet-pools every ``F/T`` consecutive video frames into one latent time token; the ball's
    position for token ``t`` is the mean (obj0_pos_x, obj0_pos_y) over that token's video frames. Used by
    the v0-aware TRAJECTORY canonicalization: unlike start-only ``canon_shift`` (which centres frame 0 but
    lets the v0 ramp carry the ball off-centre by late frames, where the accel curvature signal is
    strongest), this returns the whole path so each latent frame can be rolled to centre the ball.
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    F = state.shape[0]
    xs, ys = keys.index("obj0_pos_x"), keys.index("obj0_pos_y")
    step = max(1, F // T)
    out = np.zeros((T, 2))
    for t in range(T):
        sl = slice(t * step, min(F, (t + 1) * step)) if (t + 1) * step <= F or t < T - 1 else slice(t * step, F)
        out[t] = [float(state[sl, xs].mean()), float(state[sl, ys].mean())]
    return out


def clip_frame_velocities(sample: dict, T: int) -> np.ndarray:
    """Instantaneous ball velocity per LATENT temporal token ``(T, 2)`` (tubelet-pooled obj0_vel columns).

    The velocity analog of :func:`clip_frame_positions`. For ``scene_accel2d`` the per-frame velocity is
    ``v(f) = v0 + a*f`` (constant accel), so token ``t`` (frames ``[t*step, (t+1)*step)``) carries the mean
    instantaneous velocity ``v0 + a*mean_f``. This is what the TEMPORAL-COMPOSITION operator needs: it turns
    an acceleration edit into a sequence of per-token velocity edits, ``Delta v(t) = (a_b - a_a)*mean_f_t``,
    reusing the SOLVED velocity operator machinery per time token. Reads ``obj0_vel_x/obj0_vel_y``; every
    scenario packs those columns (velocity scenarios leave accel at 0, so ``v(f) = v0`` there).
    """
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    F = state.shape[0]
    xs, ys = keys.index("obj0_vel_x"), keys.index("obj0_vel_y")
    step = max(1, F // T)
    out = np.zeros((T, 2))
    for t in range(T):
        sl = slice(t * step, min(F, (t + 1) * step)) if (t + 1) * step <= F or t < T - 1 else slice(t * step, F)
        out[t] = [float(state[sl, xs].mean()), float(state[sl, ys].mean())]
    return out


def frame_token_times(F: int, T: int) -> np.ndarray:
    """Mean video-frame INDEX per latent temporal token ``(T,)`` -- the ``tau_t`` time axis for composition.

    With tubelet size ``F/T`` (e.g. 16 frames -> 8 tokens, step 2), token ``t`` pools frames
    ``[t*step, (t+1)*step)`` so its representative time is their mean index (``2t+0.5`` for step 2). Used to
    build the target per-token velocity ``v_b(t) = v_a(t) + Delta a * tau_t`` at steer time from the command
    alone (no H_b), consistent with :func:`clip_frame_velocities`'s pooling.
    """
    step = max(1, F // T)
    out = np.zeros(T)
    for t in range(T):
        hi = min(F, (t + 1) * step) if (t + 1) * step <= F or t < T - 1 else F
        out[t] = float(np.arange(t * step, hi).mean())
    return out


def traj_shifts(frame_pos: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """Per-frame shifts (T, 2)=(dh,dw) that roll each latent frame's ball cell to the grid CENTRE."""
    _, H, W = grid
    sh = np.zeros((frame_pos.shape[0], 2), dtype=int)
    for t in range(frame_pos.shape[0]):
        h, w = pos_to_cell(frame_pos[t], grid)
        sh[t] = (H // 2 - h, W // 2 - w)
    return sh


def roll_layer_frames(arr_flat: np.ndarray, grid: tuple[int, int, int], shifts_hw: np.ndarray) -> np.ndarray:
    """Roll each temporal token's spatial grid by its OWN ``shifts_hw[t]=(dh,dw)`` (per-frame ``roll_layer``).

    ``shifts_hw`` is ``(T, 2)``. The inverse is ``roll_layer_frames(x, grid, -shifts_hw)`` (np.roll wraps,
    so per-frame roll is exactly invertible). Same token order as :func:`roll_layer`.
    """
    T, H, W = grid
    D = arr_flat.size // (T * H * W)
    x = arr_flat.reshape(T, H, W, D)
    out = np.empty_like(x)
    for t in range(T):
        out[t] = np.roll(x[t], shift=(int(shifts_hw[t][0]), int(shifts_hw[t][1])), axis=(0, 1))
    return out.reshape(-1)


@dataclass
class Scene:
    scene_id: int
    rank_to_idx: dict[int, int] = field(default_factory=dict)


def group_scenes(ds) -> dict[int, dict[int, int]]:
    """Group a LatentDataset into ``{scene_id: {rank: dataset_index}}`` (>=2 ranks only)."""
    scenes: dict[int, dict[int, int]] = defaultdict(dict)
    for i in range(len(ds)):
        sr = scene_rank(ds._ids[i])
        if sr is not None:
            scenes[sr[0]][sr[1]] = i
    return {s: r for s, r in scenes.items() if len(r) >= 2}


# ----------------------------------------------------------------------------------------------------
# flatten / grid helpers (per layer)
# ----------------------------------------------------------------------------------------------------
def layer_flat(sample_layer) -> np.ndarray:
    """(L_tok, D) -> flat (L_tok*D,) float64."""
    return np.asarray(sample_layer, dtype=np.float64).reshape(-1)


def roll_layer(arr_flat: np.ndarray, grid: tuple[int, int, int], shift_hw: tuple[int, int]) -> np.ndarray:
    """Roll a flattened layer's SPATIAL token grid by ``shift_hw = (dh, dw)`` (with wraparound).

    Assumes token order is temporal-major then row-major spatial: index = t*(Hp*Wp) + h*Wp + w. This is
    the standard V-JEPA flattening; Phase-5 validation confirms it empirically (roll -> decode -> the
    ball should appear shifted by the same grid offset). ``A_s`` and ``A_s^{-1}`` are roll/-roll.
    """
    T, H, W = grid
    D = arr_flat.size // (T * H * W)
    x = arr_flat.reshape(T, H, W, D)
    x = np.roll(x, shift=shift_hw, axis=(1, 2))
    return x.reshape(-1)


def pos_to_cell(pos: np.ndarray, grid: tuple[int, int, int]) -> tuple[int, int]:
    """Map a normalized [0,1] (x,y) ball center to a (row=h, col=w) cell in the HpxWp spatial grid.

    Image coords: x is horizontal (-> column w), y is vertical (-> row h). Clamped to the grid.
    """
    _, H, W = grid
    w = int(np.clip(round(pos[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(pos[1] * (H - 1)), 0, H - 1))
    return h, w


def canon_shift(pos: np.ndarray, grid: tuple[int, int, int]) -> tuple[int, int]:
    """Shift (dh, dw) that moves the ball's start cell to the grid CENTER (canonical position)."""
    _, H, W = grid
    h, w = pos_to_cell(pos, grid)
    return (H // 2 - h, W // 2 - w)


def direction_bin(v: np.ndarray, n_bins: int) -> int:
    """Bin a 2D velocity's heading atan2(vy, vx) into ``n_bins`` equal wedges over [0, 2*pi).

    The velocity edit's spatial footprint depends on the ball's heading (different directions traverse
    different tokens), so a single global operator cannot place it correctly. Conditioning the operator
    on the heading bin is the equivariance-free way to make F_U direction-aware: within a wedge the path
    orientation is roughly fixed, so a position-canonicalized ridge per bin can place the edit.
    """
    ang = np.arctan2(float(v[1]), float(v[0])) % (2 * np.pi)
    return int(min(n_bins - 1, int(ang / (2 * np.pi) * n_bins)))


# ----------------------------------------------------------------------------------------------------
# PCA of Delta H (per layer)
# ----------------------------------------------------------------------------------------------------
def pca_gram(deltas: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """PCA via the Gram trick for tall-skinny data (N samples, D >> N features).

    ``deltas`` is (N, D). Returns ``(basis (k, D) orthonormal rows, explained_variance (k,))`` where
    basis rows are the top-k right singular vectors of the mean-centered data.
    """
    X = deltas - deltas.mean(0, keepdims=True)
    G = X @ X.T  # (N, N)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0, None)
    V = V[:, order]
    n = int((w > 1e-12 * (w[0] + 1e-30)).sum())
    if k is not None:
        n = min(n, k)
    sv = np.sqrt(w[:n])
    basis = (X.T @ V[:, :n]) / (sv + 1e-12)  # (D, n)
    return basis.T.copy(), w  # (n, D), full eigvals (=variance*N)


def explained_curve(eigvals: np.ndarray, upto: int = 8) -> list[float]:
    """Cumulative fraction of variance captured by the top-1..upto components."""
    pos = np.clip(eigvals, 0, None)
    tot = pos.sum() + 1e-30
    cum = np.cumsum(pos) / tot
    return [float(cum[min(i, len(cum) - 1)]) for i in range(upto)]


def participation_ratio(eigvals: np.ndarray) -> float:
    """Effective dimensionality = (sum lambda)^2 / sum(lambda^2). ~2 means a 2D subspace dominates."""
    pos = np.clip(eigvals, 0, None)
    return float((pos.sum() ** 2) / ((pos ** 2).sum() + 1e-30))


def principal_angles_bases(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """Principal angles (degrees) between two orthonormal subspaces ``A`` (k,D) and ``B`` (m,D).

    Singular values of ``A @ B.T`` are the cosines of the principal angles. Small angles -> the subspaces
    overlap (share directions); angles near 90 deg -> orthogonal / disentangled. Returns the mean and the
    smallest principal angle (the most-aligned direction pair).
    """
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
    return {"mean_deg": float(ang.mean()), "min_deg": float(ang.min())}


def orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Re-orthonormalize rows of ``basis`` (k,D) via QR (defensive; PCA bases are already ~orthonormal)."""
    Q, _ = np.linalg.qr(basis.T)
    return Q.T.copy()


def project(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project ``delta`` (D,) onto the subspace spanned by orthonormal ``basis`` (k, D): P_U delta."""
    c = basis @ delta            # (k,)
    return basis.T @ c           # (D,)


def random_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random orthonormal (k, D) basis — the same-rank control subspace."""
    A = rng.standard_normal((k, dim))
    Q, _ = np.linalg.qr(A.T)     # (D, k)
    return Q.T.copy()


# ----------------------------------------------------------------------------------------------------
# ridge operator  F_U : Delta v (2,) -> Delta H (D,)    fit by streaming normal equations
# ----------------------------------------------------------------------------------------------------
class RidgeOperator:
    """Linear map B (D x 2) with ``Delta H ~= B Delta v``, fit by accumulating X^T X (2x2) and X^T Y (2xD).

    Solve once: ``B^T = (X^T X + lambda I)^{-1} X^T Y``  (a single 2x2 inverse), so the full flattened
    dimension D = T'*Hp*Wp*1024 is no obstacle — we never materialize Y. ``predict(dv)`` returns B dv.
    """

    def __init__(self, dim: int, ridge: float = 1.0):
        self.dim = dim
        self.ridge = ridge
        self.XtX = np.zeros((2, 2))
        self.XtY = np.zeros((2, dim))
        self.n = 0

    def add(self, dv: np.ndarray, dH: np.ndarray) -> None:
        dv = dv.reshape(2)
        self.XtX += np.outer(dv, dv)
        self.XtY += np.outer(dv, dH)
        self.n += 1

    def solve(self) -> np.ndarray:
        A = self.XtX + self.ridge * np.eye(2)
        self.Bt = np.linalg.solve(A, self.XtY)  # (2, D)
        return self.Bt

    def predict(self, dv: np.ndarray) -> np.ndarray:
        return dv.reshape(2) @ self.Bt          # (D,)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-30))


def rel_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-30))


# ----------------------------------------------------------------------------------------------------
# masked TRAJECTORY-TRANSPORT operator  (PI direction 2026-06-29)
#
# The command-only ridge F_U: Delta v -> Delta H plateaus at ~34 deg (latent cos 0.39) because Delta v
# says WHAT velocity to write but not WHERE in the token grid to write it; the missing ~63% is the
# scene-local PLACEMENT (which tokens the ball's path occupies). We hand the operator that geometry as
# soft trajectory masks built from ground-truth ball centers and let it learn only velocity->channel:
#
#   Delta H_hat[t,i,j,:] = M_b*(c+_t + v_b @ B+_t)     # target mask: write a ball here (+ motion-specific)
#                        + M_a*(c-_t + v_a @ B-_t)     # source mask: remove the old ball (+ motion-specific)
#                        + M_U*((v_b-v_a) @ Bd_t)      # union mask: correction in the changed tube
#
# Each mask carries a velocity-INDEPENDENT BIAS (c+_t writes "a ball is here", c-_t removes it): the
# dominant part of Delta H near the ball is its PRESENCE -- a dark disk, the same at any speed; velocity
# only sets WHERE, which the mask already encodes. A bias-free mask*(v @ B) form cannot emit that constant
# (it must express "ball present" as a linear function of v, which partly cancels across +/- velocities
# and collapses; empirically gave cos~0.13 / shared cos~-0.95). The v-terms are then the secondary
# motion-specific modulation. LINEAR in the params, so per temporal token t it is a ridge of the 8-dim
# per-token feature phi=[M_b, M_b*v_b, M_a, M_a*v_a, M_U*dv] -> Delta H token (D,); B_t is (8, D). See
# ``LinearLS`` + ``transport_features``.
# ----------------------------------------------------------------------------------------------------
def clip_positions(sample: dict) -> np.ndarray:
    """Per-frame ball centers ``(T_frames, 2)`` (normalized [0,1] x,y) from the packed state columns."""
    keys = list(sample["state_keys"])
    state = np.asarray(sample["state"])
    xi, yi = keys.index("obj0_pos_x"), keys.index("obj0_pos_y")
    return np.stack([state[:, xi], state[:, yi]], axis=1).astype(np.float64)


def temporal_token_centers(positions: np.ndarray, n_t: int) -> np.ndarray:
    """Collapse ``(T_frames, 2)`` per-frame centers to ``(n_t, 2)`` per-temporal-token centers.

    V-JEPA's tubelet size is 2, so temporal token ``t`` aggregates frames ``2t`` and ``2t+1``; its center
    is the mean of those two frame centers. Requires ``T_frames == 2 * n_t`` (16 frames -> 8 tokens).
    """
    T = positions.shape[0]
    if T != 2 * n_t:
        raise ValueError(f"expected T_frames={2 * n_t} for n_t={n_t}, got {T}")
    return positions.reshape(n_t, 2, 2).mean(axis=1)


def forward_sim_positions(start: np.ndarray, vel: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear constant-velocity roll-out ``pos[f] = start + vel*f`` (``(n_frames, 2)``), clamped to [0,1].

    Exact for the v2d dataset (constant velocity, no bounces, ball stays in frame), so the TARGET
    trajectory mask is reconstructable at test time from the command alone — no H_b. The clamp guards
    against tiny numerical drift past the frame edge; feasible scenes never actually leave [0,1].
    """
    f = np.arange(n_frames, dtype=np.float64).reshape(-1, 1)
    return np.clip(start.reshape(1, 2) + f * vel.reshape(1, 2), 0.0, 1.0)


def gaussian_mask(centers: np.ndarray, grid: tuple[int, int, int], sigma: float) -> np.ndarray:
    """Peak-normalized Gaussian soft masks ``(T, H, W)`` around per-temporal-token centers.

    ``centers`` is ``(T, 2)`` normalized (x,y). Uses the same image->cell convention as ``pos_to_cell``
    (x -> continuous column ``w_c = x*(W-1)``, y -> continuous row ``h_c = y*(H-1)``), so the masks align
    with ``roll_layer``'s ``(T,H,W,D)`` token layout and the decoder's grid orientation. Max value 1 at
    the center cell; ``sigma`` is in cell units.
    """
    T, H, W = grid
    if centers.shape[0] != T:
        raise ValueError(f"expected {T} centers, got {centers.shape[0]}")
    hh = np.arange(H).reshape(1, H, 1)
    ww = np.arange(W).reshape(1, 1, W)
    w_c = (centers[:, 0] * (W - 1)).reshape(T, 1, 1)
    h_c = (centers[:, 1] * (H - 1)).reshape(T, 1, 1)
    d2 = (hh - h_c) ** 2 + (ww - w_c) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float64)


COMMAND_FEATURE_DIM = 13


def command_features(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Rich command feature vector ``(13,)`` for synthesizing the velocity edit WITHOUT H_b.

    The pixel proof showed velocity lives in a global low-rank subspace, not the ball tokens, so the edit
    should be synthesized from the COMMAND (v_a, v_b) — richer than the bare Delta v the plain ridge uses.
    Columns: ``[1, v_b(2), v_a(2), dv(2), |v_b|, |v_a|, u_b(2), u_a(2)]`` where ``u`` are unit headings.
    The bias + magnitudes + unit directions let a linear map capture direction/speed-dependent structure
    that ``dv`` alone cannot (e.g. the edit's magnitude scaling with speed, sign with heading).
    """
    va = np.asarray(va, dtype=np.float64).reshape(2)
    vb = np.asarray(vb, dtype=np.float64).reshape(2)
    dv = vb - va
    sa = float(np.linalg.norm(va)); sb = float(np.linalg.norm(vb))
    ua = va / (sa + 1e-9); ub = vb / (sb + 1e-9)
    return np.array([1.0, vb[0], vb[1], va[0], va[1], dv[0], dv[1], sb, sa,
                     ub[0], ub[1], ua[0], ua[1]], dtype=np.float64)


COMMAND_FEATURE_DIM_POS = 27


def command_features_pos(va: np.ndarray, vb: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Position-aware command features ``(27,)`` = base ``command_features`` (13) + 14 position terms.

    **Why (3D only).** Under a perspective camera the IMAGE velocity a latent edit must produce depends on
    *where the object is*: on ``rolling_ball3d`` the same world velocity renders 1.20x faster at one end of
    the table than the other, and image speed also varies 1.27x with heading. The latent encodes the 3D
    scene, so the edit for a given image-velocity command is really a function of the underlying WORLD
    velocity ``J(p)^-1 v_img`` — but the base 13-dim ``command_features`` contains velocities ONLY, so a
    linear map on it cannot represent that position-dependent rescaling at all. This is the leading
    suspect for 3D's synthesis gap: ``subspace_U16`` (projecting the TRUE dH) improves on ``subspace_U8``
    (15.4->12.4 deg) while the SYNTHESIZED ``cmd`` does not move at all going U8->U16 (10.90->10.83 deg),
    i.e. the extra coordinates carry velocity structure that no linear function of (v_a, v_b) predicts.

    ``pos`` is the frame-0 ball centre in normalized image coords — known at test time from clip a alone
    (no H_b), so this stays a COMMAND-ONLY operator. Within a scene every rank shares one start, so ``pos``
    is a per-scene constant and these columns let the map apply a *scene-dependent linear transform* to the
    velocity command — exactly the ``J(p)^-1`` modulation the base features cannot express.

    Columns 13..26: ``[q(2), outer(q, vb)(4), outer(q, va)(4), outer(q, dv)(4)]`` with ``q = pos - 0.5``
    (centred so the products are symmetric about the image centre). The outer products are the load-bearing
    part: a bare ``q`` could only add a position-dependent OFFSET, not rescale the command.
    On the 2D datasets this should be ~a no-op (image velocity there is a global constant, independent of
    position), which makes it a clean falsification test of the perspective explanation.
    """
    va = np.asarray(va, dtype=np.float64).reshape(2)
    vb = np.asarray(vb, dtype=np.float64).reshape(2)
    q = np.asarray(pos, dtype=np.float64).reshape(2) - 0.5
    dv = vb - va
    return np.concatenate([
        command_features(va, vb),           # 13
        q,                                  # 2
        np.outer(q, vb).ravel(),            # 4
        np.outer(q, va).ravel(),            # 4
        np.outer(q, dv).ravel(),            # 4
    ])


COMMAND_FEATURE_DIM_QUAD = 26


def command_features_quad(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Quadratic-enriched command features ``(26,)`` = base ``command_features`` (13) + 13 second-order terms.

    Motivation: for ACCELERATION steering the base linear command map is lossy (cmd->U coord cos ~0.72 but
    full reconstruction cos ~0.27). Acceleration effects on the latent are inherently nonlinear (position
    ~ 1/2 a t^2, and direction/speed interact), so a purely linear phi(a_a,a_b) cannot express them. The
    extra columns add the component products the linear basis cannot: the two accelerations' outer squares,
    their cross products, and the squared magnitudes -- keeping the streaming-ridge math unchanged (just a
    wider phi). The velocity operator keeps using the 13-dim ``command_features``; this is accel-only.
    Columns 13..25: ``[vb_x^2, vb_y^2, vb_x vb_y, va_x^2, va_y^2, va_x va_y,
                       vb_x va_x, vb_y va_y, vb_x va_y, vb_y va_x, |dv|, |vb|^2, |va|^2]``.
    """
    va = np.asarray(va, dtype=np.float64).reshape(2)
    vb = np.asarray(vb, dtype=np.float64).reshape(2)
    base = command_features(va, vb)
    dv = vb - va
    quad = np.array([vb[0] * vb[0], vb[1] * vb[1], vb[0] * vb[1],
                     va[0] * va[0], va[1] * va[1], va[0] * va[1],
                     vb[0] * va[0], vb[1] * va[1], vb[0] * va[1], vb[1] * va[0],
                     float(np.linalg.norm(dv)), float(va @ va), float(vb @ vb)],
                    dtype=np.float64)
    return np.concatenate([base, quad])


TRANSFIELD_POSBASIS_DIM = 6


def transfield_posbasis(x: np.ndarray) -> np.ndarray:
    """Smooth 2D position basis for a normalized [0,1] ball centre -> ``[1,px,py,px^2,py^2,px*py]`` (6).

    Used by the 2nd-order TRANSLATION-FIELD accel operator: ranks share pos0+v0 so the target-vs-reference
    trajectory difference is the exact closed form ``Dx(t)=1/2 (a_b-a_a) t^2`` -- i.e. the accel edit is a
    per-frame ball TRANSLATION by ``Dx(t)``, whose latent footprint depends on WHERE the ball sits,
    ``x_a(t)``. The operator feature per token is ``[Dx(t) || Dx(t) (x) transfield_posbasis(x_a(t))]`` so a
    single shared linear map expresses a smoothly position-varying translation Jacobian ``J(x_a) @ Dx``.
    Shared verbatim by ``fit_command_operators_accel_transfield.py`` and ``steer_accel2d.py --features
    transfield`` so fit and steer features are byte-identical.
    """
    px, py = float(x[0]), float(x[1])
    return np.array([1.0, px, py, px * px, py * py, px * py], dtype=np.float64)


TRANSPORT_FEATURE_DIM = 8


def transport_features(M_a: np.ndarray, M_b: np.ndarray, va: np.ndarray, vb: np.ndarray,
                       grid: tuple[int, int, int]) -> np.ndarray:
    """Assemble the per-token transport feature matrix ``phi`` of shape ``(T*H*W, 8)``.

    Row order is ``roll_layer``'s flatten ``index = t*(H*W) + h*W + w`` so that ``phi @ B_t`` reshapes
    straight back to a flat per-layer edit consumable by ``steer_velocity2d._apply_edit``. The 8 columns
    are ``[M_b, M_b*v_bx, M_b*v_by, M_a, M_a*v_ax, M_a*v_ay, M_U*dvx, M_U*dvy]`` (``M_U = max(M_a, M_b)``):
    the bare ``M_b``/``M_a`` columns are the velocity-INDEPENDENT presence write/remove biases, the
    ``M_*v_*`` columns the motion-specific modulation, and ``M_U*dv`` the changed-tube correction.
    """
    Mu = np.maximum(M_a, M_b)
    mb = M_b.reshape(-1, 1); ma = M_a.reshape(-1, 1); mu = Mu.reshape(-1, 1)
    dv = (vb - va).reshape(1, 2)
    return np.concatenate([mb, mb * vb.reshape(1, 2), ma, ma * va.reshape(1, 2), mu * dv],
                          axis=1).astype(np.float64)  # (T*H*W, 8)


class LinearLS:
    """Streaming ridge least-squares ``Y ~= X B`` for arbitrary input dim ``p`` (generalizes RidgeOperator).

    Accumulates ``X^T X`` (``p x p``) and ``X^T Y`` (``p x out``) over batches of rows, then solves
    ``B = (X^T X + lambda I)^{-1} X^T Y`` (a single ``p x p`` inverse). The transport operator uses
    ``p = 8`` (two presence biases + four mask*velocity + two union*dv) and ``out = 1024`` per
    (layer, temporal token).
    """

    def __init__(self, in_dim: int, out_dim: int, ridge: float = 1.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ridge = ridge
        self.XtX = np.zeros((in_dim, in_dim))
        self.XtY = np.zeros((in_dim, out_dim))
        self.n = 0

    def add(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Accumulate a batch: ``X`` is ``(n, in_dim)``, ``Y`` is ``(n, out_dim)``."""
        self.XtX += X.T @ X
        self.XtY += X.T @ Y
        self.n += X.shape[0]

    def solve(self, standardize: bool = False) -> np.ndarray:
        """Solve the ridge system. ``standardize=True`` fixes a scale bug, see below.

        The default applies ``ridge * I`` to the RAW features, which only makes sense when every
        column has comparable scale. ``command_features`` does not: the bias and the unit-heading
        columns have RMS ~1, while every column that carries SPEED (``v_b``, ``v_a``, ``dv``,
        ``|v_b|``, ``|v_a|``) has RMS ~0.02, because image velocities are ~0.02 frame-widths/frame.
        Ridge shrinkage on column ``j`` is ``d_j / (d_j + lambda)`` with ``d_j = (X^T X)_jj``, so at
        ``lambda = 1`` the direction columns keep ~98 % of their signal and the speed columns keep
        ~2 % -- a 47x differential. The fitted operator therefore steers heading and drops magnitude.

        With ``standardize=True`` each column is rescaled to unit RMS before the penalty is applied
        and the coefficients are mapped back afterwards, so the penalty is scale-free and the
        returned ``B`` is still used exactly as ``phi @ B``. Nothing else about the fit changes.
        """
        if not standardize:
            A = self.XtX + self.ridge * np.eye(self.in_dim)
            self.B = np.linalg.solve(A, self.XtY)  # (in_dim, out_dim)
            return self.B
        s = np.sqrt(np.maximum(np.diag(self.XtX) / max(self.n, 1), 1e-30))
        si = np.where(s > 0, 1.0 / s, 1.0)
        A = (si[:, None] * self.XtX * si[None, :]) + self.ridge * np.eye(self.in_dim)
        self.B = si[:, None] * np.linalg.solve(A, si[:, None] * self.XtY)
        return self.B

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.B
