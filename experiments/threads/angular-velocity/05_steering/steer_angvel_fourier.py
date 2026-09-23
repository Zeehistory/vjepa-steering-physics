

#!/usr/bin/env python
"""FOURIER-IN-ORIENTATION command-only angular-velocity steer (with a cheap latent GATE before any decode).

Why this and not the earlier attempts:
  * cart cmd-axis (rho 0.005) and polar cmd-U8 (rho 0.14, warp-capped 0.54) are limited because the latent's
    Cartesian footprint of an orientation change is NONLINEAR in the angle (cos/sin theta) -- a linear map in
    delta-omega cannot express it, and the polar warp->unwarp round-trip loses half the signal.
  * rotation-transport (resample H_a by d-theta) is a clean ZERO: V-JEPA latents are NOT rotation-equivariant,
    so moving tokens does not move the decoded omega.
The right basis that LINEARIZES rotation is FOURIER-IN-ORIENTATION. After center-canonicalization (roll each
clip's rotation centre to the grid middle -- exact, invertible, removes the per-scene centre confound), each
latent token value is a periodic function of the object orientation theta(t)=theta0+omega*tau_t. The bar's
pi-symmetry lives in the EVEN harmonics, the marker's 2pi cue in the FUNDAMENTAL. So we fit a GLOBAL per-token
model  H_canon[t, cell, d] ~= sum_k C[t,cell,d,k] * phi_k(theta(t)),  phi = [1, cos, sin, cos2, sin2, ...],
learned from real training clips (a tiny per-token ridge). To steer omega_a->omega_b we RE-EVALUATE the model
at the target orientation and take the difference (the DC/appearance term cancels):
    dH_canon[t] = C[t] . (phi(theta_b(t)) - phi(theta_a(t)))
then un-canonicalize (roll back) and add to H_a. Command-only: needs only (centre, theta0, omega_a, omega_b),
all in the packed state -- no H_b.

LATENT GATE (run FIRST, --gate_only, CPU/bigmem, NO decoder): held-out cosine between the SYNTHESIZED dH and
the TRUE H_b - H_a, plus a shuffled-command control and a magnitude ratio. If cos >> shuffled the operator
reconstructs the real orientation edit -> worth decoding; if ~0 it does not and we skip the GPU job.
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

import argparse, json
from pathlib import Path
import numpy as np

from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config


def to_grid(arr, grid):
    T, H, W = grid
    a = np.asarray(arr, dtype=np.float32)
    return a.reshape(T, H, W, a.size // (T * H * W))


def canon_roll(x, cen, grid, inverse=False):
    """Roll (T,H,W,D) so the rotation centre cell -> grid middle (or back). Exact integer roll (invertible)."""
    _, H, W = grid
    w = int(np.clip(round(cen[0] * (W - 1)), 0, W - 1))
    h = int(np.clip(round(cen[1] * (H - 1)), 0, H - 1))
    dh, dw = (H // 2 - h), (W // 2 - w)
    if inverse:
        dh, dw = -dh, -dw
    return np.roll(x, shift=(dh, dw), axis=(1, 2))


def phi_feats(theta, order):
    """Fourier design row for a scalar orientation theta -> [1, cos, sin, cos2, sin2, ...] (1+2*order,)."""
    out = [np.ones_like(theta)]
    for k in range(1, order + 1):
        out.append(np.cos(k * theta)); out.append(np.sin(k * theta))
    return np.stack(out, -1)


def clip_theta0(sample):
    keys = list(sample["state_keys"]); st = np.asarray(sample["state"])
    return float(st[0, keys.index("obj0_theta")])


def scene_center(sample):
    keys = list(sample["state_keys"]); st = np.asarray(sample["state"])
    return np.array([st[0, keys.index("obj0_pos_x")], st[0, keys.index("obj0_pos_y")]])


def build_samples(ds, sids, scenes, layers, grid, tau, order):
    """Collect center-canonicalized latents + Fourier design rows for every clip. Returns per-layer arrays:
    Yc[L] (N, T, HWD) canon latents (float32), PHI (N, T, P) design, and metadata lists."""
    P = 1 + 2 * order
    T, H, W = grid
    HWD = H * W
    meta = []  # (sid, idx, omega, theta0, center)
    for s in sids:
        for rk, idx in scenes[s].items():
            smp = ds[idx]
            meta.append((s, idx, float(vo.clip_angvel(smp)[0]), clip_theta0(smp), scene_center(smp)))
    N = len(meta)
    PHI = np.empty((N, T, P), dtype=np.float64)
    Ys = {L: None for L in layers}
    bufs = {L: [] for L in layers}
    for i, (s, idx, om, th0, cen) in enumerate(meta):
        smp = ds[idx]
        theta_t = th0 + om * tau                       # (T,)
        PHI[i] = phi_feats(theta_t, order)
        for L in layers:
            g = canon_roll(to_grid(smp["layers"][L], grid), cen, grid)  # (T,H,W,D)
            T_, H_, W_, D = g.shape
            bufs[L].append(g.reshape(T_, H_ * W_ * D))
    for L in layers:
        Ys[L] = np.stack(bufs[L]).astype(np.float32)   # (N, T, HWD*Dcollapsed) -> actually (N,T,H*W*D)
    return meta, PHI, Ys


def fit_fourier(PHI, Ys, layers, ridge):
    """Per (layer, token t): C[t] = (Phi_t^T Phi_t + lam I)^-1 Phi_t^T Y_t.  Phi_t (N,P), Y_t (N, HWD).
    Returns {L: C (T, P, HWD)}."""
    N, T, P = PHI.shape
    C = {}
    for L in layers:
        Y = Ys[L]                                       # (N, T, M)
        M = Y.shape[2]
        CL = np.empty((T, P, M), dtype=np.float32)
        for t in range(T):
            X = PHI[:, t, :]                            # (N,P)
            A = X.T @ X + ridge * np.eye(P)
            B = X.T @ Y[:, t, :]                        # (P, M)
            CL[t] = np.linalg.solve(A, B).astype(np.float32)
        C[L] = CL
    return C


def predict_dH_canon(C, om_a, om_b, th0, tau, order, layers, grid):
    """dH_canon[t] = C[t] . (phi(theta_b(t)) - phi(theta_a(t))). Returns {L: (T,H,W,D)}."""
    T, H, W = grid
    fa = phi_feats(th0 + om_a * tau, order)             # (T,P)
    fb = phi_feats(th0 + om_b * tau, order)
    df = (fb - fa)                                      # (T,P)
    out = {}
    for L in layers:
        CL = C[L]                                       # (T,P,M)
        M = CL.shape[2]
        dH = np.einsum("tp,tpm->tm", df, CL).astype(np.float32)  # (T, M)
        out[L] = dH.reshape(T, H, W, M // (H * W))
    return out


def cosine(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--n_train_scenes", type=int, default=125)
    ap.add_argument("--n_test_scenes", type=int, default=30)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--gains", default="0.5,1,1.5,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--viz_scenes", type=int, default=0,
                    help="after picking the best gain, re-decode the first N test scenes and dump the "
                         "FULL clips (GT a / decode(H_a) / decode(H_a+edit) / decode(H_b) / GT b) as npz")
    ap.add_argument("--viz_dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    grid = tuple(int(x) for x in tr[0]["grid"])
    T = grid[0]
    F = int(np.asarray(tr[0]["state"]).shape[0])
    tau = vo.frame_token_times(F, T)
    print(f"[fourier] layers={layers} grid={grid} order={args.order} ridge={args.ridge} tau={np.round(tau,1)}", flush=True)

    tr_scenes = vo.group_scenes(tr); tr_ids = sorted(tr_scenes)[: args.n_train_scenes]
    print(f"[fourier] building {len(tr_ids)} train scenes (center-canon + Fourier design) ...", flush=True)
    _, PHI_tr, Ys_tr = build_samples(tr, tr_ids, tr_scenes, layers, grid, tau, args.order)
    print(f"[fourier] fitting per-token Fourier model (N={PHI_tr.shape[0]}) ...", flush=True)
    C = fit_fourier(PHI_tr, Ys_tr, layers, args.ridge)
    del Ys_tr

    # ---- LATENT GATE on held-out test: cos(pred dH_canon, true dH_canon) vs shuffled-command control ----
    te_scenes = vo.group_scenes(te); te_ids = sorted(te_scenes)[: args.n_test_scenes]
    print(f"[fourier] LATENT GATE on {len(te_ids)} test scenes ...", flush=True)
    gate = {L: dict(cos=[], cos_shuf=[], magr=[]) for L in layers}
    # cache canon latents per test clip for reuse in decode
    for s in te_ids:
        ranks = sorted(te_scenes[s])
        wmap = {r: float(vo.clip_angvel(te[te_scenes[s][r]])[0]) for r in ranks}
        base = min(ranks, key=lambda r: abs(wmap[r]))
        sa = te[te_scenes[s][base]]
        cen = scene_center(sa); th0 = clip_theta0(sa)
        Ha_canon = {L: canon_roll(to_grid(sa["layers"][L], grid), cen, grid) for L in layers}
        wa = wmap[base]
        # a wrong (shuffled) target for the control: pick another scene's omega spread midpoint
        for r in ranks:
            if r == base:
                continue
            wb = wmap[r]
            sb = te[te_scenes[s][r]]
            Hb_canon = {L: canon_roll(to_grid(sb["layers"][L], grid), cen, grid) for L in layers}
            pred = predict_dH_canon(C, wa, wb, th0, tau, args.order, layers, grid)
            wb_wrong = -wb if abs(wb) > 1e-6 else 0.15   # sign-flipped command (must NOT match true dH)
            pred_shuf = predict_dH_canon(C, wa, wb_wrong, th0, tau, args.order, layers, grid)
            for L in layers:
                true_dH = Hb_canon[L] - Ha_canon[L]
                gate[L]["cos"].append(cosine(pred[L], true_dH))
                gate[L]["cos_shuf"].append(cosine(pred_shuf[L], true_dH))
                gate[L]["magr"].append(float(np.linalg.norm(pred[L]) / (np.linalg.norm(true_dH) + 1e-30)))
    print("\n==================== LATENT GATE (Fourier orientation operator) ====================")
    gate_summary = {}
    for L in layers:
        c = np.array(gate[L]["cos"]); cs = np.array(gate[L]["cos_shuf"]); mr = np.array(gate[L]["magr"])
        gate_summary[L] = dict(cos=float(c.mean()), cos_shuffled=float(cs.mean()), mag_ratio=float(mr.mean()),
                               n=len(c))
        print(f"  L{L:2d}: cos(pred,true dH)={c.mean():+.3f}  shuffled-cmd cos={cs.mean():+.3f}  "
              f"mag_ratio={mr.mean():.3f}  n={len(c)}")
    best_gate = max(gate_summary.values(), key=lambda d: d["cos"])
    print(f"  BEST layer cos={best_gate['cos']:+.3f} (shuffled {best_gate['cos_shuffled']:+.3f}). "
          f"Decode worthwhile if cos >> shuffled and > ~0.4.")

    result = dict(order=args.order, ridge=args.ridge, layers=layers, gate=gate_summary,
                  n_train_scenes=len(tr_ids), n_test_scenes=len(te_ids))

    # ---- DECODE steer (only if not gate_only) ----
    if not args.gate_only:
        import torch
        from src.analysis.ball_tracking import measured_angvel
        from src.decoders import build_decoder
        from src.encoders.feature_extractor import latent_collate
        from src.training.checkpoints import load_checkpoint
        DARK, RED = 0.25, 0.08
        dev = args.device
        gains = [float(g) for g in args.gains.split(",")]
        rec0 = te.records[0]
        enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
        cfg.decoder.state_dim = state_dim
        if cfg.decoder.out_num_frames <= 0:
            cfg.decoder.out_num_frames = cfg.data.num_frames
        decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(dev).eval()
        if hasattr(decoder, "prime_layers"):
            decoder.prime_layers([int(x) for x in te.available_layers()])
        load_checkpoint(args.checkpoint, decoder, map_location=dev)
        for pm in decoder.parameters():
            pm.requires_grad_(False)

        @torch.no_grad()
        def decode_frames(latflat, ref):
            lat = {L: torch.from_numpy(np.ascontiguousarray(latflat[L].reshape(1, -1, latflat[L].shape[-1]))).to(dev, ref[L].dtype) for L in layers}
            fr = decoder(lat, grid).frames
            return None if fr is None else fr[0].cpu()

        def honest(latflat, ref):
            fr = decode_frames(latflat, ref)
            return float("nan") if fr is None else float(measured_angvel(fr, darkness_thresh=DARK, red_thresh=RED)["omega"])

        rng = np.random.default_rng(0)
        perm = rng.permutation(len(te_ids)); val_sids = set(np.array(te_ids)[perm[:len(te_ids)//2]].tolist())
        rows = []
        for n, s in enumerate(te_ids):
            ranks = sorted(te_scenes[s])
            wmap = {r: float(vo.clip_angvel(te[te_scenes[s][r]])[0]) for r in ranks}
            base = min(ranks, key=lambda r: abs(wmap[r]))
            sa = te[te_scenes[s][base]]
            cen = scene_center(sa); th0 = clip_theta0(sa); wa = wmap[base]
            ref = {int(k): v for k, v in latent_collate([sa])["layers"].items() if int(k) in layers}
            Ha_cart = {L: to_grid(sa["layers"][L], grid) for L in layers}
            for r in ranks:
                if r == base:
                    continue
                wb = wmap[r]
                dH_canon = predict_dH_canon(C, wa, wb, th0, tau, args.order, layers, grid)
                # un-canonicalize the edit and add to H_a in native (cart) frame
                edit = {L: canon_roll(dH_canon[L], cen, grid, inverse=True) for L in layers}
                om_by_gain = {g: honest({L: Ha_cart[L] + g * edit[L] for L in layers}, ref) for g in gains}
                rows.append(dict(s=s, in_val=(s in val_sids), wb=wb, om=om_by_gain))
            print(f"  decode scene {n+1}/{len(te_ids)}", flush=True)

        def metrics(pairs):
            pairs = [(t, h) for t, h in pairs if np.isfinite(h)]
            if len(pairs) < 3:
                return dict(n=len(pairs), rho=float('nan'), sign_acc=float('nan'), mag_ratio=float('nan'), mae=float('nan'))
            t = np.array([a for a, _ in pairs]); h = np.array([b for _, b in pairs])
            return dict(n=len(pairs), rho=float(np.corrcoef(t, h)[0, 1]), sign_acc=float(np.mean(np.sign(h) == np.sign(t))),
                        mag_ratio=float(np.polyfit(t, h, 1)[0]), mae=float(np.mean(np.abs(h - t))))
        val_mse = {}
        for g in gains:
            pr = [(rw["wb"], rw["om"][g]) for rw in rows if rw["in_val"] and np.isfinite(rw["om"][g])]
            val_mse[g] = float(np.mean([(b - a) ** 2 for a, b in pr])) if pr else float("inf")
        best = min(gains, key=lambda g: val_mse[g])
        held = metrics([(rw["wb"], rw["om"][best]) for rw in rows if not rw["in_val"]])
        val = metrics([(rw["wb"], rw["om"][best]) for rw in rows if rw["in_val"]])
        curve = {g: metrics([(rw["wb"], rw["om"][g]) for rw in rows if not rw["in_val"]]) for g in gains}
        hr = [rw for rw in rows if not rw["in_val"]]; sh = list(rng.permutation([rw["wb"] for rw in hr]))
        rand = metrics([(sh[i], hr[i]["om"][best]) for i in range(len(hr))])
        print("\n==================== FOURIER-ORIENTATION DECODE RESULT ====================")
        print(f"  best_gain={best} val_mse=" + " ".join(f"{g}:{val_mse[g]:.4f}" for g in gains))
        print(f"  HELD-OUT @g{best}: rho={held['rho']:+.3f} sign={held['sign_acc']:.2f} mag={held['mag_ratio']:+.3f} mae={held['mae']:.4f} n={held['n']}")
        print(f"  val      @g{best}: rho={val['rho']:+.3f} sign={val['sign_acc']:.2f} mag={val['mag_ratio']:+.3f}")
        print(f"  random-command:   rho={rand['rho']:+.3f} sign={rand['sign_acc']:.2f}")
        print("  held-out per-gain rho: " + " ".join(f"g{g}:{curve[g]['rho']:+.2f}(m{curve[g]['mag_ratio']:+.2f})" for g in gains))
        print("  reference: interp ceiling 0.87 (needs H_b); polar cmd-U8 0.14; rotation-transport 0.0")
        result.update(dict(best_gain=best, val_mse=val_mse, heldout=held, val=val, random_command=rand,
                           gain_curve={str(g): curve[g] for g in gains}))

        # ---- qualitative dump at the chosen gain: whole clips, for the slide GIFs ----
        if args.viz_scenes and args.viz_dir:
            vd = Path(args.viz_dir); vd.mkdir(parents=True, exist_ok=True)
            held_sids = [s for s in te_ids if s not in val_sids]   # only steer clips we did not tune on
            for s in held_sids[: args.viz_scenes]:
                ranks = sorted(te_scenes[s])
                wmap = {r: float(vo.clip_angvel(te[te_scenes[s][r]])[0]) for r in ranks}
                base = min(ranks, key=lambda r: abs(wmap[r]))
                # the widest omega gap in the scene -> the most legible before/after
                tgt = max((r for r in ranks if r != base), key=lambda r: abs(wmap[r] - wmap[base]))
                sa, sb = te[te_scenes[s][base]], te[te_scenes[s][tgt]]
                cen, th0, wa, wb = scene_center(sa), clip_theta0(sa), wmap[base], wmap[tgt]
                ref = {int(k): v for k, v in latent_collate([sa])["layers"].items() if int(k) in layers}
                Ha_cart = {L: to_grid(sa["layers"][L], grid) for L in layers}
                Hb_cart = {L: to_grid(sb["layers"][L], grid) for L in layers}
                edit = {L: canon_roll(dh, cen, grid, inverse=True) for L, dh in
                        predict_dH_canon(C, wa, wb, th0, tau, args.order, layers, grid).items()}
                f_uns = decode_frames(Ha_cart, ref)
                f_ste = decode_frames({L: Ha_cart[L] + best * edit[L] for L in layers}, ref)
                f_tgt = decode_frames(Hb_cart, ref)
                if any(f is None for f in (f_uns, f_ste, f_tgt)):
                    print(f"  [viz] scene {s:05d}: decoder returned no frames, skipped", flush=True)
                    continue
                om = {k: float(measured_angvel(v, darkness_thresh=DARK, red_thresh=RED)["omega"])
                      for k, v in (("uns", f_uns), ("ste", f_ste), ("tgt", f_tgt))}
                np.savez_compressed(
                    vd / f"viz_scene{s:05d}.npz",
                    gt_a=np.asarray(sa["frames"], np.float16),
                    gt_b=np.asarray(sb["frames"], np.float16),
                    unsteered=f_uns.numpy().astype(np.float16),
                    steered=f_ste.numpy().astype(np.float16),
                    target=f_tgt.numpy().astype(np.float16),
                    omega_a=np.float32(wa), omega_b=np.float32(wb), gain=np.float32(best),
                    om_unsteered=np.float32(om["uns"]), om_steered=np.float32(om["ste"]),
                    om_target=np.float32(om["tgt"]))
                print(f"  [viz] scene {s:05d}: omega_a={wa:+.3f} -> commanded {wb:+.3f} | "
                      f"decoded before {om['uns']:+.3f} after {om['ste']:+.3f} "
                      f"(decode(H_b) {om['tgt']:+.3f})", flush=True)
            print(f"[fourier] viz npz -> {vd}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[fourier] wrote {args.out}")


if __name__ == "__main__":
    main()
