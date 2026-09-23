

#!/usr/bin/env python
"""Command-only rotational steering with the FOURIER-IN-ORIENTATION operator -- angular VELOCITY or
angular ACCELERATION, through one code path.

The angular-velocity steer is solved (held-out rho=0.94, beating the interp ceiling that needs the real
H_b). This script generalizes it to angular ACCELERATION, and the single most important property is that
NOTHING about the operator changes between the two: the quantity only selects which orientation trajectory
theta(t) the command implies.

    angvel   : theta(t) = theta0 + omega*tau                    (command = omega)
    angaccel : theta(t) = theta0 + omega0*tau + 0.5*alpha*tau^2 (command = alpha)

The headline experiment is ZERO-SHOT TRANSFER ACROSS KINEMATIC ORDER: fit the operator on constant-omega
clips only (``--train_dir`` = the angvel latents), then steer angular acceleration with NO refit. The
operator is indexed by ORIENTATION, never by the command, so if it is really a model of the SO(2) group
action it must transfer to a kinematic order it never saw. ``--train_dir`` = the angaccel latents instead
gives the matched/in-domain control, which upper-bounds what the zero-shot number could reach.

Every reported number is held-out and pixel-verified: the edit is decoded and the quantity is read back off
the rendered frames by the honest tracker (``measured_angvel`` / ``measured_angaccel``), which was validated
against ground-truth frames first. Controls: a shuffled-command latent gate and a random-command decode.

Run ``--gate_only`` first (CPU/bigmem, no decoder): if cos(pred dH, true H_b-H_a) is not well above the
shuffled-command control, skip the GPU decode.
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

from src.analysis import fourier_orientation as fo
from src.analysis import velocity_ops as vo
from src.encoders.feature_extractor import LatentDataset
from src.utils.config import load_config

QUANTITIES = ("angvel", "angaccel")


def clip_command(sample):
    """The full rotational command of a clip: ``(centre, theta0, omega0, alpha)``.

    All of it is read from the packed ground-truth state -- never from H_b -- so a steer built from this is
    command-only. ``alpha`` is 0 for the constant-spin family and for latents encoded before ``obj0_alpha``
    joined the schema, which keeps the older angvel caches readable by this script unchanged.
    """
    keys = list(sample["state_keys"])
    st = np.asarray(sample["state"])
    centre = np.array([st[0, keys.index("obj0_pos_x")], st[0, keys.index("obj0_pos_y")]])
    theta0 = float(st[0, keys.index("obj0_theta")])
    omega0 = float(st[0, keys.index("obj0_omega")])          # row 0 = INITIAL rate (ramps if alpha != 0)
    alpha = float(st[:, keys.index("obj0_alpha")].mean()) if "obj0_alpha" in keys else 0.0
    return centre, theta0, omega0, alpha


def target_of(sample, quantity):
    """The scalar being steered/measured for this quantity."""
    _, _, omega0, alpha = clip_command(sample)
    return alpha if quantity == "angaccel" else omega0


def design_rows(theta0, omega0, alpha, tau, args):
    """``(T, P)`` design rows for a command, in the configured basis."""
    th = fo.theta_of(theta0, omega0, alpha, tau)
    om = fo.omega_of(omega0, alpha, tau)
    return fo.design_matrix(th, om, order=args.order, harmonics=args.harmonics, basis=args.basis)


def build_train(ds, sids, scenes, layers, grid, tau, args):
    """Center-canonicalized latents + design rows for every training clip.

    Also returns ``groups`` -- the row span of each scene -- which the per-token scale fit needs in
    order to form WITHIN-SCENE pairs (across scenes the centre and theta0 differ, so a cross-scene
    difference is not an edit of the kind the operator is ever asked to make).
    """
    T, H, W = grid
    meta, groups = [], []
    for s in sids:
        start = len(meta)
        for _rk, idx in scenes[s].items():
            meta.append(idx)
        groups.append((start, len(meta) - start))
    N = len(meta)
    P = fo.n_features(args.order, args.harmonics, args.basis)
    PHI = np.empty((N, T, P), dtype=np.float64)
    bufs = {L: [] for L in layers}
    for i, idx in enumerate(meta):
        smp = ds[idx]
        cen, th0, om0, al = clip_command(smp)
        PHI[i] = design_rows(th0, om0, al, tau, args)
        for L in layers:
            g = fo.to_grid(smp["layers"][L], grid)
            if not args.no_canon:
                g = fo.canon_roll(g, cen, grid)
            bufs[L].append(g.reshape(T, -1))
    Ys = {L: np.stack(bufs[L]).astype(np.float32) for L in layers}
    return PHI, Ys, groups


def fit_per_token_scale(PHI, Ys, groups, C, layers, grid, max_scenes=40):
    """Least-squares per-temporal-token rescaling ``s_t`` of the synthesized edit, fit on TRAIN.

    Why this and not one scalar. The operator's edit is a rotation of the orientation code, and the
    orientation displacement it has to realize grows with the kinematic order: for angular VELOCITY
    ``dtheta(t) = domega * tau_t`` spans about 29x across the 8 temporal tokens, but for angular
    ACCELERATION ``dtheta(t) = 0.5 * dalpha * tau_t^2`` spans about **841x** -- roughly 0.002 rad at
    token 0 and 2.0 rad at token 7. One global gain therefore serves wildly different per-token
    magnitudes, and the reported ``mag_ratio ~ 0.53`` is an average over that spread, not a uniform
    shortfall. This fits the SHAPE of the correction and leaves the overall scale to the existing gain
    sweep (``s`` is normalized to mean 1, so the two knobs cannot fight).

    ``s_t = sum_pairs <pred_t, true_t> / sum_pairs ||pred_t||^2`` -- the LS scale, accumulated over
    within-scene pairs and summed over layers so it stays ONE knob family, like the global gain.
    Nothing here reads the test split, and when the train latents are the constant-omega family the
    scale is as zero-shot as the operator it corrects.
    """
    T = grid[0]
    num = np.zeros(T, dtype=np.float64)
    den = np.zeros(T, dtype=np.float64)
    n_pairs = 0
    for start, cnt in groups[:max_scenes]:
        if cnt < 2:
            continue
        for j in range(1, cnt):
            a, b = start, start + j
            pred = predict_dH_rows(C, PHI[a], PHI[b], layers, grid)
            for L in layers:
                pt = pred[L].reshape(T, -1)
                tt = Ys[L][b].astype(np.float64) - Ys[L][a].astype(np.float64)
                num += (pt * tt).sum(axis=1)
                den += (pt * pt).sum(axis=1)
            n_pairs += 1
    s = num / np.maximum(den, 1e-30)
    m = float(np.mean(s))
    return (s / m if abs(m) > 1e-12 else np.ones(T)), n_pairs


def predict_dH_rows(C, row_a, row_b, layers, grid):
    """``fo.predict_dH`` on raw design rows -- named so the scale fit reads like the decode path."""
    return fo.predict_dH(C, row_a, row_b, layers, grid)


def metrics(pairs):
    pairs = [(t, h) for t, h in pairs if np.isfinite(h)]
    if len(pairs) < 3:
        return dict(n=len(pairs), rho=float("nan"), sign_acc=float("nan"),
                    mag_ratio=float("nan"), mae=float("nan"))
    t = np.array([a for a, _ in pairs]); h = np.array([b for _, b in pairs])
    return dict(n=len(pairs), rho=float(np.corrcoef(t, h)[0, 1]),
                sign_acc=float(np.mean(np.sign(h) == np.sign(t))),
                mag_ratio=float(np.polyfit(t, h, 1)[0]), mae=float(np.mean(np.abs(h - t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_dir", required=True, help="latents the operator is FIT on (angvel = zero-shot)")
    ap.add_argument("--test_dir", required=True, help="latents that are STEERED")
    ap.add_argument("--quantity", choices=QUANTITIES, default="angaccel")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gate_only", action="store_true")
    ap.add_argument("--n_train_scenes", type=int, default=125)
    ap.add_argument("--n_test_scenes", type=int, default=30)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--harmonics", choices=["all", "even", "odd"], default="all")
    ap.add_argument("--basis", choices=list(fo.BASES), default="orientation")
    ap.add_argument("--no_canon", action="store_true", help="ablate center-canonicalization")
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--gains", default="0.5,1,1.5,2,3,4")
    ap.add_argument("--per_token_gain", action="store_true",
                    help="rescale the synthesized edit per TEMPORAL TOKEN by a least-squares factor fit "
                         "on the same TRAIN latents the operator is fit on (normalized to mean 1, so the "
                         "global gain sweep is unchanged). Motivated by the order asymmetry: the angular "
                         "ACCELERATION edit's orientation displacement spans ~841x across the 8 tokens "
                         "against ~29x for angular velocity, so a single scalar gain is the wrong shape")
    ap.add_argument("--ptg_scenes", type=int, default=40,
                    help="train scenes used for the per-token scale fit (it is a T-vector; 40 scenes is "
                         "280 within-scene pairs, already far more than it has parameters)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz_scenes", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    Q = args.quantity

    tr = LatentDataset(args.train_dir, layers=cfg.encoder.layers)
    te = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in tr[0]["layers"].keys())
    grid = tuple(int(x) for x in tr[0]["grid"])
    F = int(np.asarray(tr[0]["state"]).shape[0])
    tau = vo.frame_token_times(F, grid[0])
    tr_alpha = abs(clip_command(tr[0])[3]) > 1e-9
    print(f"[fourier:{Q}] layers={layers} grid={grid} order={args.order} harmonics={args.harmonics} "
          f"basis={args.basis} canon={not args.no_canon} ridge={args.ridge}", flush=True)
    print(f"[fourier:{Q}] TRAIN latents look like {'ANGACCEL (matched fit)' if tr_alpha else 'constant-omega ANGVEL (ZERO-SHOT transfer)'}"
          f" -> {args.train_dir}", flush=True)

    tr_scenes = vo.group_scenes(tr); tr_ids = sorted(tr_scenes)[: args.n_train_scenes]
    print(f"[fourier:{Q}] building {len(tr_ids)} train scenes ...", flush=True)
    PHI_tr, Ys_tr, groups_tr = build_train(tr, tr_ids, tr_scenes, layers, grid, tau, args)
    print(f"[fourier:{Q}] fitting per-token operator (N={PHI_tr.shape[0]}, P={PHI_tr.shape[2]}) ...", flush=True)
    C = fo.fit_operator(PHI_tr, Ys_tr, layers, args.ridge)
    tok_scale = np.ones(grid[0])
    if args.per_token_gain:
        tok_scale, n_ptg = fit_per_token_scale(PHI_tr, Ys_tr, groups_tr, C, layers, grid,
                                               args.ptg_scenes)
        print(f"[fourier:{Q}] per-token scale (mean-1, {n_ptg} train pairs): "
              + " ".join(f"t{t}:{v:.3f}" for t, v in enumerate(tok_scale)), flush=True)
    del Ys_tr

    # ---------------- LATENT GATE: cos(synthesized dH, TRUE H_b - H_a) vs shuffled-command control -------
    te_scenes = vo.group_scenes(te); te_ids = sorted(te_scenes)[: args.n_test_scenes]
    print(f"[fourier:{Q}] LATENT GATE on {len(te_ids)} test scenes ...", flush=True)
    gate = {L: dict(cos=[], cos_shuf=[], magr=[]) for L in layers}
    for s in te_ids:
        ranks = sorted(te_scenes[s])
        qmap = {r: target_of(te[te_scenes[s][r]], Q) for r in ranks}
        base = min(ranks, key=lambda r: abs(qmap[r]))
        sa = te[te_scenes[s][base]]
        cen, th0, om0, al_a = clip_command(sa)
        Ha = {L: (fo.to_grid(sa["layers"][L], grid) if args.no_canon
                  else fo.canon_roll(fo.to_grid(sa["layers"][L], grid), cen, grid)) for L in layers}
        row_a = design_rows(th0, om0, al_a, tau, args)
        for r in ranks:
            if r == base:
                continue
            sb = te[te_scenes[s][r]]
            _, _, om0_b, al_b = clip_command(sb)
            Hb = {L: (fo.to_grid(sb["layers"][L], grid) if args.no_canon
                      else fo.canon_roll(fo.to_grid(sb["layers"][L], grid), cen, grid)) for L in layers}
            row_b = design_rows(th0, om0_b, al_b, tau, args)
            pred = fo.predict_dH(C, row_a, row_b, layers, grid)
            # control: sign-flipped command -> a DIFFERENT trajectory, must not reconstruct the true edit
            if Q == "angaccel":
                row_w = design_rows(th0, om0, -al_b if abs(al_b) > 1e-9 else 0.01, tau, args)
            else:
                row_w = design_rows(th0, -om0_b if abs(om0_b) > 1e-9 else 0.15, 0.0, tau, args)
            pred_w = fo.predict_dH(C, row_a, row_w, layers, grid)
            for L in layers:
                true_dH = Hb[L] - Ha[L]
                gate[L]["cos"].append(fo.cosine(pred[L], true_dH))
                gate[L]["cos_shuf"].append(fo.cosine(pred_w[L], true_dH))
                gate[L]["magr"].append(float(np.linalg.norm(pred[L]) / (np.linalg.norm(true_dH) + 1e-30)))
    print(f"\n============ LATENT GATE (Fourier orientation operator, {Q}) ============")
    gate_summary = {}
    for L in layers:
        c = np.array(gate[L]["cos"]); cs = np.array(gate[L]["cos_shuf"]); mr = np.array(gate[L]["magr"])
        gate_summary[L] = dict(cos=float(c.mean()), cos_shuffled=float(cs.mean()),
                               mag_ratio=float(mr.mean()), n=len(c))
        print(f"  L{L:2d}: cos(pred,true dH)={c.mean():+.3f}  shuffled-cmd={cs.mean():+.3f}  "
              f"mag_ratio={mr.mean():.3f}  n={len(c)}")
    best_gate = max(gate_summary.values(), key=lambda d: d["cos"])
    print(f"  BEST layer cos={best_gate['cos']:+.3f} (shuffled {best_gate['cos_shuffled']:+.3f})")

    result = dict(quantity=Q, order=args.order, harmonics=args.harmonics, basis=args.basis,
                  canon=not args.no_canon, ridge=args.ridge, layers=layers, gate=gate_summary,
                  n_train_scenes=len(tr_ids), n_test_scenes=len(te_ids),
                  train_dir=args.train_dir, test_dir=args.test_dir,
                  zero_shot=bool(Q == "angaccel" and not tr_alpha),
                  per_token_gain=bool(args.per_token_gain),
                  token_scale=[round(float(v), 4) for v in tok_scale])

    # ---------------- DECODE: pixel-verified steer ------------------------------------------------------
    if not args.gate_only:
        import torch
        from src.analysis.ball_tracking import measured_angaccel, measured_angvel
        from src.decoders import build_decoder
        from src.encoders.feature_extractor import latent_collate
        from src.training.checkpoints import load_checkpoint
        DARK, RED = 0.25, 0.08     # tight thresholds: the big-object gate needs these (default 0.5/0.25 fails)
        dev = args.device
        gains = [float(g) for g in args.gains.split(",")]
        rec0 = te.records[0]
        enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
        # Size the decoder's state head from the CHECKPOINT, not from these latents. The head only
        # predicts state (mode C) and is unused for frame decoding, but load_state_dict is strict, so a
        # width mismatch would refuse to load. The angaccel latents carry the extra ``obj0_alpha`` column
        # (state_dim 15) while the decoder was trained at 14 -- rendering is identical either way.
        ck_sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
        ck_state_dim = next((int(v.shape[0]) for k, v in ck_sd.items()
                             if k.endswith("state_head.3.bias")), state_dim)
        if ck_state_dim != state_dim:
            print(f"[fourier:{Q}] latents state_dim={state_dim} but checkpoint state head={ck_state_dim}"
                  f" -> building the decoder at {ck_state_dim} (state head unused for frame decode)")
        del ck_sd
        cfg.decoder.state_dim = ck_state_dim
        if cfg.decoder.out_num_frames <= 0:
            cfg.decoder.out_num_frames = cfg.data.num_frames
        decoder = build_decoder(cfg.decoder, enc_dim, ck_state_dim).to(dev).eval()
        if hasattr(decoder, "prime_layers"):
            decoder.prime_layers([int(x) for x in te.available_layers()])
        load_checkpoint(args.checkpoint, decoder, map_location=dev)
        for pm in decoder.parameters():
            pm.requires_grad_(False)

        def read(fr):
            if Q == "angaccel":
                m = measured_angaccel(fr, darkness_thresh=DARK, red_thresh=RED)
                return float(m["alpha"])
            m = measured_angvel(fr, darkness_thresh=DARK, red_thresh=RED)
            return float(m["omega"])

        @torch.no_grad()
        def honest(latflat, ref, want_frames=False):
            lat = {L: torch.from_numpy(np.ascontiguousarray(
                latflat[L].reshape(1, -1, latflat[L].shape[-1]))).to(dev, ref[L].dtype) for L in layers}
            fr = decoder(lat, grid).frames
            if fr is None:
                return (float("nan"), None) if want_frames else float("nan")
            v = read(fr[0].cpu())
            return (v, fr[0].cpu()) if want_frames else v

        rng = np.random.default_rng(0)
        perm = rng.permutation(len(te_ids))
        val_sids = set(np.array(te_ids)[perm[: len(te_ids) // 2]].tolist())
        rows, ceil_rows, viz = [], [], []
        for n, s in enumerate(te_ids):
            ranks = sorted(te_scenes[s])
            qmap = {r: target_of(te[te_scenes[s][r]], Q) for r in ranks}
            base = min(ranks, key=lambda r: abs(qmap[r]))
            sa = te[te_scenes[s][base]]
            cen, th0, om0, al_a = clip_command(sa)
            ref = {int(k): v for k, v in latent_collate([sa])["layers"].items() if int(k) in layers}
            Ha_cart = {L: fo.to_grid(sa["layers"][L], grid) for L in layers}
            row_a = design_rows(th0, om0, al_a, tau, args)
            for r in ranks:
                if r == base:
                    continue
                sb = te[te_scenes[s][r]]
                _, _, om0_b, al_b = clip_command(sb)
                tgt = qmap[r]
                row_b = design_rows(th0, om0_b, al_b, tau, args)
                dH = fo.predict_dH(C, row_a, row_b, layers, grid)
                edit = {L: (dH[L] if args.no_canon else fo.canon_roll(dH[L], cen, grid, inverse=True))
                        for L in layers}
                if args.per_token_gain:
                    edit = {L: edit[L] * tok_scale.reshape(-1, 1, 1, 1) for L in layers}
                om_by_gain = {g: honest({L: Ha_cart[L] + g * edit[L] for L in layers}, ref) for g in gains}
                rows.append(dict(s=s, in_val=(s in val_sids), tgt=float(tgt), om=om_by_gain))
                # CEILING: decode the TRUE H_b. Bounds what any edit could achieve, and separates
                # "operator failed" from "decoder cannot render this quantity".
                Hb_cart = {L: fo.to_grid(sb["layers"][L], grid) for L in layers}
                ceil_rows.append(dict(tgt=float(tgt), om=honest(Hb_cart, ref)))
            if n < args.viz_scenes:
                viz.append(int(s))
            print(f"  decode scene {n+1}/{len(te_ids)}", flush=True)

        val_mse = {}
        for g in gains:
            pr = [(rw["tgt"], rw["om"][g]) for rw in rows if rw["in_val"] and np.isfinite(rw["om"][g])]
            val_mse[g] = float(np.mean([(b - a) ** 2 for a, b in pr])) if pr else float("inf")
        best = min(gains, key=lambda g: val_mse[g])
        held = metrics([(rw["tgt"], rw["om"][best]) for rw in rows if not rw["in_val"]])
        val = metrics([(rw["tgt"], rw["om"][best]) for rw in rows if rw["in_val"]])
        curve = {g: metrics([(rw["tgt"], rw["om"][g]) for rw in rows if not rw["in_val"]]) for g in gains}
        ceiling = metrics([(rw["tgt"], rw["om"]) for rw in ceil_rows])
        hr = [rw for rw in rows if not rw["in_val"]]
        sh = list(rng.permutation([rw["tgt"] for rw in hr]))
        rand = metrics([(sh[i], hr[i]["om"][best]) for i in range(len(hr))])
        print(f"\n============ FOURIER-ORIENTATION DECODE RESULT ({Q}) ============")
        print(f"  fit on: {'CONSTANT-OMEGA latents -> ZERO-SHOT across kinematic order' if result['zero_shot'] else 'matched latents (in-domain control)'}")
        print(f"  best_gain={best}  val_mse=" + " ".join(f"{g}:{val_mse[g]:.5f}" for g in gains))
        print(f"  HELD-OUT @g{best}: rho={held['rho']:+.3f} sign={held['sign_acc']:.2f} "
              f"mag={held['mag_ratio']:+.3f} mae={held['mae']:.5f} n={held['n']}")
        print(f"  val      @g{best}: rho={val['rho']:+.3f} sign={val['sign_acc']:.2f}")
        print(f"  CEILING (decode true H_b): rho={ceiling['rho']:+.3f} sign={ceiling['sign_acc']:.2f} "
              f"mag={ceiling['mag_ratio']:+.3f} n={ceiling['n']}")
        print(f"  random-command control:    rho={rand['rho']:+.3f} sign={rand['sign_acc']:.2f}")
        print("  held-out per-gain rho: " + " ".join(
            f"g{g}:{curve[g]['rho']:+.2f}(m{curve[g]['mag_ratio']:+.2f})" for g in gains))
        result.update(dict(best_gain=best, val_mse=val_mse, heldout=held, val=val, ceiling=ceiling,
                           random_command=rand, gain_curve={str(g): curve[g] for g in gains}))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[fourier:{Q}] wrote {args.out}")


if __name__ == "__main__":
    main()
