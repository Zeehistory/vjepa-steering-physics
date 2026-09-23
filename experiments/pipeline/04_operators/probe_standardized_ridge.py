"""Leave-one-scene-out: does standardizing the ridge features recover MAGNITUDE control?

Decoder-free and memory-lean. The edit lives entirely in span(U), and the H->v readout is linear, so
everything factors through per-layer dot products -- we hold ONE layer's 96 clips at a time (~0.8 GB)
and accumulate, instead of materialising the 6 GB concatenated design matrix.

  Gram:  G = sum_L Xc_L Xc_L^T          (exact: concat dot product = sum of per-layer dot products)
  read:  v(z) = sum_L (z_L - mu_L) . Xc_L^T . alpha,   alpha = (G + lam I)^-1 Y
"""
import sys, numpy as np
sys.path.insert(0, '.')
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

B = 'outputs/filmstrip/latents'
A = 'outputs/analysis'
LAY = [6, 12, 18, 23]
RIDGE = 1.0
KUS = [4, 8, 16]        # global_basis_L*.npy is saved with 16 rows


def solve(XtX, XtY, n, standardize):
    """Ridge solve. standardize=True rescales each feature column to unit RMS BEFORE applying the
    penalty, then maps the coefficients back, so `phi @ W` is unchanged in form."""
    p = XtX.shape[0]
    if not standardize:
        return np.linalg.solve(XtX + RIDGE * np.eye(p), XtY)
    s = np.sqrt(np.maximum(np.diag(XtX) / max(n, 1), 1e-30))
    si = 1.0 / s
    return si[:, None] * np.linalg.solve((si[:, None] * XtX * si[None, :]) + RIDGE * np.eye(p),
                                         si[:, None] * XtY)


def run(ds_name, art_name, gain):
    ds = LatentDataset(f'{B}/{ds_name}/test/vjepa2_large', layers=LAY)
    sc = vo.group_scenes(ds)
    keys = [(s, r) for s in sorted(sc) for r in sorted(sc[s])]
    kidx = {k: i for i, k in enumerate(keys)}
    N = len(keys)
    q = {}
    for (s, r) in keys:
        q[(s, r)] = np.asarray(vo.clip_velocity(ds[sc[s][r]]), float).reshape(2)
    Y = np.array([q[k] for k in keys])

    pairs = []
    for s in sorted(sc):
        rk = sorted(sc[s]); a = rk[0]
        for b in rk[1:]:
            pairs.append((s, a, b, vo.command_features(q[(s, a)], q[(s, b)])))

    KUMAX = max(KUS)
    U = {L: np.load(f'{A}/{art_name}/subspace/global_basis_L{L}.npy').astype(np.float64)[:KUMAX]
         for L in LAY}

    G = np.zeros((N, N))                       # centered Gram, accumulated over layers
    anchorXc = np.zeros((len(pairs), N))       # (H_a - mu) . Xc^T
    editXc = {}                                # per (std, layer): filled below
    UXc = {}                                   # (8, N) per layer: U_k . Xc^T
    pairUdH = {L: np.zeros((len(pairs), KUMAX)) for L in LAY}

    for L in LAY:
        Xl = np.stack([vo.layer_flat(ds[sc[s][r]]['layers'][L]).astype(np.float64)
                       for (s, r) in keys])
        mu = Xl.mean(0)
        Xc = Xl - mu
        G += Xc @ Xc.T
        UXc[L] = U[L] @ Xc.T                                   # (KUMAX, N)
        for pi, (s, a, b, phi) in enumerate(pairs):
            ia, ib = kidx[(s, a)], kidx[(s, b)]
            pairUdH[L][pi] = U[L] @ (Xl[ib] - Xl[ia])
            anchorXc[pi] += Xc[ia]  @ Xc.T                      # (N,)
        del Xl, Xc

    alpha = np.linalg.solve(G + 1e3 * np.eye(N), Y)             # (N, 2)
    r2 = 1 - ((G @ alpha - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()

    scenes = sorted(sc)

    def score(getcoef, ku):
        C, Ah = [], []
        for i, p in enumerate(pairs):
            proj = anchorXc[i].copy()
            for L in LAY:
                proj = proj + gain * (getcoef(i, p, L) @ UXc[L][:ku])
            C.append(np.linalg.norm(q[(p[0], p[2])])); Ah.append(np.linalg.norm(proj @ alpha))
        C = np.array(C); Ah = np.array(Ah)
        return np.corrcoef(C, Ah)[0, 1], np.polyfit(C, Ah, 1)[0], Ah.mean(), Ah.std()

    res = {}
    for ku in KUS:
        # ORACLE ceiling: the TRUE delta projected onto U[:ku] (what subspace_Uk measures).
        # gain is 1.0 for the oracle -- it is not a synthesised edit, so it is not gain-calibrated.
        cc = np.array([[np.linalg.norm(q[(p[0], p[2])]) for p in pairs]]).ravel()
        Ah = []
        for i, p in enumerate(pairs):
            proj = anchorXc[i].copy()
            for L in LAY:
                proj = proj + (pairUdH[L][i][:ku] @ UXc[L][:ku])
            Ah.append(np.linalg.norm(proj @ alpha))
        Ah = np.array(Ah)
        res[f'oracle subspace_U{ku}'] = (np.corrcoef(cc, Ah)[0, 1], np.polyfit(cc, Ah, 1)[0],
                                        Ah.mean(), Ah.std())
        for std in (False, True):
            # leave-one-scene-out refit for every fold, then score every pair once
            coef = {}
            for held in scenes:
                tr = [i for i, pp in enumerate(pairs) if pp[0] != held]
                Phi = np.array([pairs[i][3] for i in tr])
                XtX = Phi.T @ Phi
                Wu = {L: solve(XtX, Phi.T @ pairUdH[L][tr][:, :ku], len(tr), std) for L in LAY}
                for i, pp in enumerate(pairs):
                    if pp[0] == held:
                        coef[i] = {L: pp[3] @ Wu[L] for L in LAY}
            tag = 'standardized' if std else 'baseline'
            res[f'cmd_U{ku} {tag}'] = score(lambda i, p, L: coef[i][L], ku)
    return r2, res, len(pairs)


for nm, art, g in [('moving_ball_scene_v2d_mixed', 'moving_ball_v2d_mixed', 2.5),
                   ('rolling_ball3d', 'rolling_ball3d', 2.0)]:
    r2, o, npair = run(nm, art, g)
    print(f'\n== {nm}  (readout R2={r2:.4f}, {npair} pairs, leave-one-scene-out)', flush=True)
    print(f'{"fit":26s} {"corr(|cmd|,|ach|)":>18s} {"slope":>8s} {"mean|ach|":>10s} {"sd|ach|":>9s}')
    for k, (c, s, m, sd) in o.items():
        print(f'{k:26s} {c:+18.3f} {s:+8.3f} {m:10.4f} {sd:9.4f}')
