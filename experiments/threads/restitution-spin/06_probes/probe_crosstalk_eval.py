

#!/usr/bin/env python
"""Spin crosstalk under velocity steering, via a CERTIFIED phase probe instead of a decoder.

John's second bar -- "angular velocity may deviate, but not by more than ~10-20%" -- needs a spin
number in physical units. Reading it off pixels is blocked: seven decoder runs render the marker as a
static smudge (omega_corr ~0), and correctly-projected phase supervision degrades velocity without
moving phase. But ``theta_probe.py`` shows per-frame phase IS in the latent -- median 18 deg at L18
against an 89 deg phase-free floor -- so the information the decoder cannot express is nonetheless
present and linearly readable.

So the instrument here is a ridge probe ``H -> (cos theta, sin theta)`` per temporal token, from which
omega is the slope of unwrapped theta. This is NOT the dimensionless latent-cosine "leak" quoted
earlier; it is a readout in rad/frame, directly comparable to the dataset's ``obj0_omega`` and therefore
to a percentage bar.

**It is only worth as much as its certification, so that is measured first and reported alongside every
result.** ``--certify`` fits the probe on train scenes and scores omega on held-out REAL clips against
ground truth. If that correlation is weak the instrument is not usable and the script says so rather
than proceeding.

**The honest caveat, reported not buried.** The probe is fit on real latents and then applied to EDITED
ones. A velocity edit can push latents off the manifold the probe was fit on, and a probe read there is
an extrapolation. Two controls bound this:
  * ``gt_vel`` -- a REAL clip at the commanded velocity, whose spin is identical to base by the factorial
    design. Its measured deviation is the instrument's own floor, on-manifold.
  * ``rand`` -- a norm-matched random edit of the same size, which perturbs the latent as much as the
    steering edit does without steering anything. If the steered deviation is no larger than this, what
    is being measured is sensitivity to perturbation, not crosstalk.

    PYTHONPATH=. python experiments/threads/restitution-spin/06_probes/probe_crosstalk_eval.py --train_dir ... --test_dir ... \
        --model .../nl_L23_h2048.pt --out ...
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
import torch

from src.analysis import spin_ops as so
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset

from fit_transport_spin import _phi_xl, _deployable_centers  # noqa: E402
from fit_nonlinear_spin import TokenMLP  # noqa: E402


def _tokens(layer_arr, grid) -> np.ndarray:
    """(T,H,W,D) -> (T,D), pooled over space: one row per temporal token."""
    T, H, W = grid
    return np.asarray(layer_arr, dtype=np.float64).reshape(T, H * W, -1).mean(axis=1)


def _omega_from_theta(cs: np.ndarray) -> float:
    """Slope of unwrapped atan2 over temporal tokens, in rad per TOKEN."""
    th = np.unwrap(np.arctan2(cs[:, 1], cs[:, 0]))
    t = np.arange(th.shape[0], dtype=np.float64)
    if th.shape[0] < 3:
        return float("nan")
    A = np.stack([t, np.ones_like(t)], axis=1)
    return float(np.linalg.lstsq(A, th, rcond=None)[0][0])


class ThetaProbe:
    """Ridge H -> (cos theta, sin theta) per temporal token, with train-set standardisation + PCA."""

    def __init__(self, layer: int, pca_dim: int = 256, alpha: float = 1.0):
        self.L, self.k, self.alpha = layer, pca_dim, alpha

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ThetaProbe":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.centre = Xs.mean(0)
        _U, _S, Vt = np.linalg.svd(Xs - self.centre, full_matrices=False)
        self.P = Vt[: self.k].T
        Z = (Xs - self.centre) @ self.P
        G = Z.T @ Z
        self.W = np.linalg.solve(G + self.alpha * np.eye(G.shape[0]), Z.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (((X - self.mu) / self.sd) - self.centre) @ self.P
        out = Z @ self.W
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def _collect(ds, ti, layer, limit):
    X, Y, om, ids = [], [], [], []
    for i in range(min(limit, len(ds))):
        s = ds[i]
        st = np.asarray(s["state"], dtype=np.float64)
        if st.ndim != 2:
            continue
        grid = tuple(int(x) for x in s["grid"])
        T = grid[0]
        F = st.shape[0]
        if F % T != 0:
            continue
        th = st[:, ti].reshape(T, F // T)
        cs = np.stack([np.cos(th).mean(1), np.sin(th).mean(1)], axis=1)
        cs = cs / (np.linalg.norm(cs, axis=1, keepdims=True) + 1e-12)
        X.append(_tokens(s["layers"][layer], grid))
        Y.append(cs)
        om.append(so.clip_spin(s))
        ids.append(i)
    return np.concatenate(X), np.concatenate(Y), np.array(om), ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--model", required=True, help="saved velocity operator (.pt)")
    ap.add_argument("--out", required=True)
    # MUST equal the operator's layer -- see the assertion below. 18 reads theta slightly better
    # (18.0 deg vs 21.2 deg) but the edit never reaches it, so it would measure nothing.
    ap.add_argument("--probe_layer", type=int, default=23)
    ap.add_argument("--gains", default="1,1.5")
    ap.add_argument("--n_fit_clips", type=int, default=512)
    ap.add_argument("--n_scenes", type=int, default=24)
    ap.add_argument("--squares_per_scene", type=int, default=3)
    ap.add_argument("--n_vel", type=int, default=4)
    ap.add_argument("--n_spin", type=int, default=4)
    ap.add_argument("--min_omega_corr", type=float, default=0.6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    gains = [float(x) for x in args.gains.split(",") if x]
    PL = args.probe_layer

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    L = int(ck["layer"])
    sigmas, Q = ck["sigmas"], ck["Q"]
    mu_t = torch.from_numpy(ck["mu_f"]).to(dev)
    sd_t = torch.from_numpy(ck["sd_f"]).to(dev)
    op = TokenMLP(ck["p_in"], ck["hidden"], 1024, ck["depth"]).to(dev)
    op.load_state_dict(ck["state_dict"])
    op.eval()
    print(f"[xt] velocity operator at layer {L}; phase probe at layer {PL}", flush=True)
    if PL != L:
        # We edit stored latents directly rather than re-running the encoder, so an edit at layer L
        # leaves every other layer bit-identical. Probing a different layer would therefore report a
        # spin deviation of exactly zero -- and "velocity steering does not disturb spin" is the
        # result being tested for, so this would look like success instead of a no-op.
        raise SystemExit(
            f"probe_layer ({PL}) must equal the operator's layer ({L}). Editing layer {L} leaves "
            f"layer {PL} untouched, so the measured deviation would be trivially zero.")

    layers = sorted({L, PL})
    dtr = LatentDataset(args.train_dir, layers=layers, max_cached_shards=1)
    dte = LatentDataset(args.test_dir, layers=layers, max_cached_shards=2)
    keys = list(dtr[0]["state_keys"])
    ti = keys.index("obj0_theta")

    Xtr, Ytr, _, _ = _collect(dtr, ti, PL, args.n_fit_clips)
    probe = ThetaProbe(PL).fit(Xtr, Ytr)
    print(f"[xt] probe fit on {Xtr.shape[0]} tokens", flush=True)

    # ---- CERTIFY on held-out REAL clips ------------------------------------------------------------
    Xte, Yte, om_gt, ids = _collect(dte, ti, PL, 256)
    T0 = int(np.asarray(dte[ids[0]]["grid"])[0])
    pred = probe.predict(Xte)
    ang = np.degrees(np.arccos(np.clip((pred * Yte).sum(1), -1, 1)))
    om_hat = np.array([_omega_from_theta(pred[i * T0:(i + 1) * T0]) for i in range(len(ids))])
    F_per_tok = int(np.asarray(dte[ids[0]]["state"]).shape[0]) // T0
    om_hat_per_frame = om_hat / F_per_tok
    ok = np.isfinite(om_hat_per_frame) & np.isfinite(om_gt)
    corr = float(np.corrcoef(om_hat_per_frame[ok], om_gt[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    cert = {"theta_median_ang_err_deg": round(float(np.median(ang)), 2),
            "omega_corr_vs_gt": round(corr, 4),
            "omega_median_abs_err": round(float(np.median(np.abs(
                om_hat_per_frame[ok] - om_gt[ok]))), 5),
            "n_clips": int(ok.sum())}
    print(f"[xt] CERTIFICATION: theta {cert['theta_median_ang_err_deg']} deg, "
          f"omega corr {cert['omega_corr_vs_gt']}, |err| {cert['omega_median_abs_err']}", flush=True)
    if not (corr >= args.min_omega_corr):
        raise SystemExit(
            f"REFUSING to report crosstalk: probe omega correlates {corr:.3f} with truth on real "
            f"held-out clips (need >= {args.min_omega_corr}). An uncertified readout would produce a "
            f"spin number that describes the probe, not the model.")

    # ---- steering ----------------------------------------------------------------------------------
    scenes = vo.group_scenes(dte)
    sids = sorted(scenes)[: args.n_scenes]
    rng = np.random.default_rng(0)
    rows = []
    for n, s in enumerate(sids):
        cells = {divmod(int(r), args.n_spin): i for r, i in scenes[s].items()}
        if len(cells) != args.n_vel * args.n_spin:
            continue
        picks = [(a, sp, b, sp2) for a, sp, b, sp2 in
                 [(rng.integers(args.n_vel), rng.integers(args.n_spin),
                   rng.integers(args.n_vel), rng.integers(args.n_spin))
                  for _ in range(args.squares_per_scene * 3)] if a != b and sp != sp2]
        for (vi_a, si_a, vi_b, si_b) in picks[: args.squares_per_scene]:
            sq = so.commutation_square(cells, int(vi_a), int(si_a), int(vi_b), int(si_b))
            sam = {k: dte[i] for k, i in sq.items()}
            grid = tuple(int(x) for x in sam["base"]["grid"])
            T, H, W = grid
            va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
            base_flat = vo.layer_flat(sam["base"]["layers"][L]).reshape(T * H * W, 1024)
            tgt = _deployable_centers(sam["base"], vb, grid)
            phi = _phi_xl(sam["base"], tgt, va, vb, grid, sigmas, Q, base_flat).astype(np.float32)
            with torch.no_grad():
                e = op((torch.from_numpy(phi).to(dev) - mu_t) / sd_t).cpu().numpy().astype(np.float64)

            # The edit lives at layer L; the probe reads layer PL. When they differ the edit does not
            # touch the probe's layer at all and any "deviation" would be trivially zero -- so this is
            # only meaningful when the steering layer feeds the probe layer, i.e. L == PL. Guard it.
            variants = {"base": None, "gt_vel": "gt"}
            for g in gains:
                variants[f"V_g{g:g}"] = g
            rr = np.random.default_rng(int(s) * 1000 + vi_a)
            rnd = rr.standard_normal(e.shape)
            rnd *= np.linalg.norm(e) / (np.linalg.norm(rnd) + 1e-12)
            variants["rand"] = "rand"

            rec = {"scene": int(s)}
            for name, spec in variants.items():
                if spec == "gt":
                    toks = _tokens(sam["vel_only"]["layers"][PL], grid)
                else:
                    base_pl = np.asarray(sam["base"]["layers"][PL], dtype=np.float64).reshape(
                        T * H * W, -1).copy()
                    if spec == "rand":
                        base_pl += rnd
                    elif spec is not None:
                        base_pl += float(spec) * e
                    toks = base_pl.reshape(T, H * W, -1).mean(axis=1)
                cs = probe.predict(toks)
                rec[name] = {"omega_tok": _omega_from_theta(cs)}
            rec["omega_gt_base"] = so.clip_spin(sam["base"])
            rows.append(rec)
        if (n + 1) % 4 == 0:
            print(f"[xt] scene {n + 1}/{len(sids)}  squares={len(rows)}", flush=True)

    conds = ["gt_vel"] + [f"V_g{g:g}" for g in gains] + ["rand"]
    summary = {"probe_layer": PL, "operator_layer": L, "n_squares": len(rows),
               "certification": cert, "conditions": {}}
    for c in conds:
        rel, ab = [], []
        for r in rows:
            ob, oc = r["base"]["omega_tok"], r[c]["omega_tok"]
            if not (np.isfinite(ob) and np.isfinite(oc)):
                continue
            ab.append(abs(oc - ob))
            if abs(ob) > 1e-6:
                rel.append(abs(oc - ob) / abs(ob))
        summary["conditions"][c] = {
            "spin_dev_rel_median": round(float(np.median(rel)), 4) if rel else None,
            "spin_dev_abs_median_per_token": round(float(np.median(ab)), 6) if ab else None,
            "n": len(ab)}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print(f"\n# Spin deviation under velocity steering, via certified phase probe "
          f"({len(rows)} squares)\n")
    print(f"probe certification: theta {cert['theta_median_ang_err_deg']} deg, "
          f"omega corr {cert['omega_corr_vs_gt']} vs ground truth on real held-out clips\n")
    print("| condition | spin deviation (relative) | (rad/token) |")
    print("|---|---|---|")
    for c in conds:
        v = summary["conditions"][c]
        f = lambda x, n=4: "n/a" if x is None else f"{x:+.{n}f}"
        print(f"| `{c}` | {f(v['spin_dev_rel_median'])} | {f(v['spin_dev_abs_median_per_token'], 5)} |")
    print("\nBAR: spin deviation <= 0.10-0.20. `gt_vel` is a REAL clip at the commanded velocity with")
    print("identical spin by construction -- its deviation is the instrument floor, and a steered")
    print("deviation at or below it is not measurable. `rand` is a norm-matched perturbation of the")
    print("same size: if steering matches it, the sensitivity is to perturbation, not to steering.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
