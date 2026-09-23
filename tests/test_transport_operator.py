"""Tests for the masked trajectory-transport operator (Step 2, PI direction 2026-06-29).

These guard the geometry + linear-algebra primitives that the transport edit is built on, BEFORE any
cluster job: mask placement matches the canonical image->cell convention, the constant-velocity
forward-sim reproduces the ground-truth trajectory exactly, temporal-token centers average the right
frame pair, the streaming ridge recovers a known linear map, and — the load-bearing one — a Delta H
synthesized from the masked model with known B's is recovered by the fit at cosine ~1 with the SAME
flatten order ``steer_velocity2d._apply_edit`` consumes (catches any (T,H,W,D) orientation flip).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.analysis import velocity_ops as vo

GRID = (8, 16, 16)
_BASE = os.environ.get("V2D_BASE", ".")
_TRAIN_DIR = f"{_BASE}/outputs/latents/moving_ball_scene_v2d/train/vjepa2_large"


def test_gaussian_mask_peaks_at_pos_to_cell():
    """The Gaussian peak cell must equal the (h,w) cell ``pos_to_cell`` returns for that center, and an
    ON-GRID center (x,y land exactly on a cell) attains the normalized peak value 1.0."""
    pos = np.array([6.0 / 15, 9.0 / 15])  # x*(W-1)=6, y*(H-1)=9 -> exact cell (h=9, w=6)
    centers = np.tile(pos, (GRID[0], 1))
    M = vo.gaussian_mask(centers, GRID, sigma=1.0)
    h, w = vo.pos_to_cell(pos, GRID)
    assert (h, w) == (9, 6)
    for t in range(GRID[0]):
        peak = np.unravel_index(np.argmax(M[t]), M[t].shape)
        assert peak == (h, w), f"t={t}: peak {peak} != pos_to_cell {(h, w)}"
        assert M[t, h, w] == pytest.approx(1.0)  # peak-normalized at an on-grid center
    assert M.max() <= 1.0 + 1e-9


def test_gaussian_mask_orientation_x_is_column():
    """A purely horizontal shift in x must move the peak along the COLUMN axis, not the row axis."""
    c_left = np.tile(np.array([0.1, 0.5]), (GRID[0], 1))
    c_right = np.tile(np.array([0.9, 0.5]), (GRID[0], 1))
    pl = np.unravel_index(np.argmax(vo.gaussian_mask(c_left, GRID, 1.0)[0]), GRID[1:])
    pr = np.unravel_index(np.argmax(vo.gaussian_mask(c_right, GRID, 1.0)[0]), GRID[1:])
    assert pl[0] == pr[0]      # same row
    assert pr[1] > pl[1]       # column increased with x


def test_temporal_token_centers_average_frame_pairs():
    pos = np.arange(16 * 2, dtype=float).reshape(16, 2) / 100.0  # distinct per frame
    c = vo.temporal_token_centers(pos, n_t=8)
    assert c.shape == (8, 2)
    for t in range(8):
        assert c[t] == pytest.approx(0.5 * (pos[2 * t] + pos[2 * t + 1]))
    with pytest.raises(ValueError):
        vo.temporal_token_centers(pos[:15], n_t=8)


def test_forward_sim_linear_and_clamped():
    start = np.array([0.4, 0.4]); vel = np.array([0.02, -0.01])
    p = vo.forward_sim_positions(start, vel, 16)
    assert p.shape == (16, 2)
    assert p[0] == pytest.approx(start)
    assert p[5] == pytest.approx(start + 5 * vel)
    # clamp: a velocity that would exit the frame is held inside [0,1]
    pc = vo.forward_sim_positions(np.array([0.95, 0.5]), np.array([0.1, 0.0]), 16)
    assert pc[:, 0].max() <= 1.0 and pc[:, 0].min() >= 0.0


def test_linearls_recovers_known_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 6))
    Btrue = rng.standard_normal((6, 32))
    Y = X @ Btrue
    op = vo.LinearLS(6, 32, ridge=1e-8)
    for i in range(0, 500, 64):
        op.add(X[i:i + 64], Y[i:i + 64])
    B = op.solve()
    assert op.n == 500
    assert np.allclose(B, Btrue, atol=1e-4)
    assert np.allclose(op.predict(X[:10]), Y[:10], atol=1e-4)


def test_transport_features_shape_and_columns():
    """Columns [M_b, M_b*v_b(2), M_a, M_a*v_a(2), M_U*dv(2)] — note the bare presence-bias columns."""
    M_a = np.zeros(GRID); M_b = np.zeros(GRID)
    M_a[2, 3, 4] = 1.0   # source hot token
    M_b[2, 5, 6] = 1.0   # target hot token
    va = np.array([0.01, 0.02]); vb = np.array([-0.01, 0.03])
    phi = vo.transport_features(M_a, M_b, va, vb, GRID)
    assert phi.shape == (8 * 16 * 16, 8)
    ib = 2 * 256 + 5 * 16 + 6   # roll_layer flatten index of the target hot token
    ia = 2 * 256 + 3 * 16 + 4
    assert phi[ib, 0] == pytest.approx(1.0)           # M_b presence bias
    assert phi[ib, 1:3] == pytest.approx(vb)          # M_b*v_b block
    assert phi[ia, 3] == pytest.approx(1.0)           # M_a presence bias
    assert phi[ia, 4:6] == pytest.approx(va)          # M_a*v_a block
    # union mask is hot at BOTH tokens -> dv block present at each
    assert phi[ib, 6:8] == pytest.approx(vb - va)
    assert phi[ia, 6:8] == pytest.approx(vb - va)
    # a token with NO mask is all zeros (incl. biases) -> no edit there
    assert phi[0] == pytest.approx(np.zeros(8))


def test_roundtrip_synthetic_recovers_delta_h():
    """Synthesize Delta H from the masked model with known per-t B's; the fit must recover it (cos~1),
    proving the feature/flatten ordering round-trips through phi @ B_t -> flat edit."""
    rng = np.random.default_rng(1)
    D = 8
    P = vo.TRANSPORT_FEATURE_DIM
    Btrue = rng.standard_normal((8, P, D))  # per temporal token
    op = {t: vo.LinearLS(P, D, ridge=1e-9) for t in range(8)}

    def synth(M_a, M_b, va, vb):
        phi = vo.transport_features(M_a, M_b, va, vb, GRID)        # (2048, 6)
        flat = np.zeros((2048, D))
        for t in range(8):
            sl = slice(t * 256, (t + 1) * 256)
            flat[sl] = phi[sl] @ Btrue[t]
        return phi, flat.reshape(-1)

    pairs = []
    for _ in range(40):
        start = rng.uniform(0.2, 0.8, 2)
        va = rng.uniform(-0.02, 0.02, 2); vb = rng.uniform(-0.02, 0.02, 2)
        ca = vo.temporal_token_centers(vo.forward_sim_positions(start, va, 16), 8)
        cb = vo.temporal_token_centers(vo.forward_sim_positions(start, vb, 16), 8)
        M_a = vo.gaussian_mask(ca, GRID, 1.0); M_b = vo.gaussian_mask(cb, GRID, 1.0)
        phi, dH = synth(M_a, M_b, va, vb)
        for t in range(8):
            sl = slice(t * 256, (t + 1) * 256)
            op[t].add(phi[sl], dH.reshape(2048, D)[sl])
        pairs.append((M_a, M_b, va, vb, dH))

    Bfit = {t: op[t].solve() for t in range(8)}
    # held-out-ish reconstruction on the same scenes (closed-form should be near exact)
    M_a, M_b, va, vb, dH = pairs[0]
    phi = vo.transport_features(M_a, M_b, va, vb, GRID)
    pred = np.zeros((2048, D))
    for t in range(8):
        sl = slice(t * 256, (t + 1) * 256)
        pred[sl] = phi[sl] @ Bfit[t]
    assert vo.cosine(pred.reshape(-1), dH) > 0.999


@pytest.mark.skipif(not os.path.isdir(_TRAIN_DIR), reason="v2d train cache not present")
def test_forward_sim_matches_gt_positions_on_real_sample():
    """On a real cached clip, forward-sim from (start, GT velocity) must reproduce the stored per-frame
    centers to <1e-5 — the invariant that lets us build the target mask at test time without H_b."""
    from src.encoders.feature_extractor import LatentDataset
    ds = LatentDataset(_TRAIN_DIR, layers=[6])
    s = ds[0]
    gt = vo.clip_positions(s)
    start = gt[0]; vel = vo.clip_velocity(s)
    sim = vo.forward_sim_positions(start, vel, gt.shape[0])
    assert np.abs(sim - gt).max() < 1e-5, f"forward-sim desynced from GT by {np.abs(sim - gt).max()}"
