

#!/usr/bin/env python
"""John's question, both halves, in PIXELS: steer velocity -- how much does each quantity move?

The acceptance bars are stated on decoded motion, so both must be read off decoded frames:
  * velocity steered by >= 80-90% of the commanded amount;
  * angular velocity deviating by no more than ~10-20%.

``pixel_eval_nonlinear.py`` answers the velocity half and deliberately reports NO spin number, because
the velocity decoder does not draw the marker (marker mass 0.0002). This script answers both, and it
REFUSES to run rather than quote a spin number off a decoder that cannot render the marker -- an
unreadable marker still yields *a* fitted omega, namely a slope through noise, which would look like
"steering does not disturb spin" when the truth is "spin is not rendered". That failure mode is the
whole reason ``decoder_gate.py`` exists; this script re-checks it inline so it cannot be skipped.

**The controls are what make the spin number mean anything.**
  * ``gt_vel`` -- the real clip at the commanded velocity. By the factorial design its spin is IDENTICAL
    to base, so ``omega(gt_vel) - omega(base)`` is pure readout noise: the floor below which no measured
    "deviation" is real. Quoting a steered deviation without it would be quoting the instrument.
  * ``rand`` -- a norm-matched random edit at the same layer. Bounds how much omega moves when the
    latents are merely perturbed rather than steered.

Spin deviation is reported RELATIVE to |omega_base| (John's bar is a percentage) and also in absolute
rad/frame, since a relative error explodes for clips whose true spin is near zero.

    PYTHONPATH=. python experiments/threads/restitution-spin/06_probes/pixel_crosstalk_eval.py --config ... --model ... \
        --checkpoint <a decoder that PASSES decoder_gate.py> --test_dir ... --out ...
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
from src.analysis import spin_tracking as st
from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_velocity, measured_velocity_marker_invariant
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

from fit_transport_spin import _phi_xl, _deployable_centers  # noqa: E402
from fit_nonlinear_spin import TokenMLP  # noqa: E402

MIN_MARKER_MASS = 0.20   # decoded/rendered warm mass; below this the marker is not being drawn


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def _decode(decoder, latents, grid):
    out = decoder(latents, grid)
    return None if out.frames is None else out.frames[0].cpu().clamp(0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="saved nonlinear operator (.pt)")
    ap.add_argument("--checkpoint", required=True, help="decoder that passes decoder_gate.py")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gains", default="1,1.5")
    ap.add_argument("--n_scenes", type=int, default=24)
    ap.add_argument("--squares_per_scene", type=int, default=3)
    ap.add_argument("--n_vel", type=int, default=4)
    ap.add_argument("--n_spin", type=int, default=4)
    ap.add_argument("--allow_unrendered_marker", action="store_true",
                    help="quote spin numbers even if the marker is not drawn (produces fiction)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    gains = [float(x) for x in args.gains.split(",") if x]

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    L = int(ck["layer"])
    sigmas, Q = ck["sigmas"], ck["Q"]
    mu_t = torch.from_numpy(ck["mu_f"]).to(dev)
    sd_t = torch.from_numpy(ck["sd_f"]).to(dev)
    model = TokenMLP(ck["p_in"], ck["hidden"], 1024, ck["depth"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    cfg = load_config(args.config)
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers, max_cached_shards=2)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=dev)
    print(f"[xt] operator layer {L}; decoder {args.checkpoint}", flush=True)

    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.n_scenes]
    rng = np.random.default_rng(0)
    rows, mass_ratios = [], []

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
            sam = {k: ds[i] for k, i in sq.items()}
            grid = tuple(int(x) for x in sam["base"]["grid"])
            T, H, W = grid
            Ha = _to_dev(sam["base"], layers, dev)
            Hb = _to_dev(sam["vel_only"], layers, dev)

            va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
            base_flat = vo.layer_flat(sam["base"]["layers"][L]).reshape(T * H * W, 1024)
            tgt = _deployable_centers(sam["base"], vb, grid)
            phi = _phi_xl(sam["base"], tgt, va, vb, grid, sigmas, Q, base_flat).astype(np.float32)
            X = (torch.from_numpy(phi).to(dev) - mu_t) / sd_t
            e = model(X)

            def add(edit_t):
                return {Lk: (t + edit_t.reshape(1, t.shape[1], t.shape[2]) if Lk == L else t)
                        for Lk, t in Ha.items()}

            fr = {"base": _decode(decoder, Ha, grid), "gt_vel": _decode(decoder, Hb, grid)}
            for g in gains:
                fr[f"V_g{g:g}"] = _decode(decoder, add(g * e), grid)
            r = torch.randn_like(e)
            r = r * (e.norm() / (r.norm() + 1e-12))
            fr["rand"] = _decode(decoder, add(r), grid)
            if any(v is None for v in fr.values()):
                continue

            # Is the marker actually drawn? Checked on the BASE decode, per square.
            gt_frames = sam["base"].get("frames")
            wm_dec = float((fr["base"][:, 0] - fr["base"][:, 2] - st.MARKER_RB_THRESH).clamp(min=0).sum())
            if gt_frames is not None and torch.is_tensor(gt_frames) and gt_frames.numel() > 1:
                gg = gt_frames.float()
                gg = gg / 255.0 if gg.max() > 1.5 else gg
                wm_gt = float((gg[:, 0] - gg[:, 2] - st.MARKER_RB_THRESH).clamp(min=0).sum())
                if wm_gt > 1e-9:
                    mass_ratios.append(wm_dec / wm_gt)

            rec = {"scene": int(s)}
            for k, f in fr.items():
                mv, mvi, ms = (measured_velocity(f), measured_velocity_marker_invariant(f),
                               st.measured_spin(f))
                rec[k] = {"vel_x": mv["vel_x"], "vel_y": mv["vel_y"],
                          "vel_x_mi": mvi["vel_x"], "vel_y_mi": mvi["vel_y"],
                          "omega": ms["omega"], "omega_nvalid": ms["n_valid"],
                          "omega_resid": ms["residual"]}
            rows.append(rec)
        if (n + 1) % 4 == 0:
            print(f"[xt] scene {n + 1}/{len(sids)}  squares={len(rows)}", flush=True)

    smear = float(np.nanmedian(mass_ratios)) if mass_ratios else float("nan")
    print(f"\n[xt] marker mass decoded/rendered = {smear:.4f}", flush=True)
    if not (smear >= MIN_MARKER_MASS) and not args.allow_unrendered_marker:
        raise SystemExit(
            f"REFUSING to report spin crosstalk: marker mass {smear:.4f} < {MIN_MARKER_MASS}. This "
            f"decoder does not draw the marker, so any omega read off it is a slope through noise and "
            f"would masquerade as 'velocity steering leaves spin alone'. Use a decoder that passes "
            f"experiments/threads/restitution-spin/06_probes/decoder_gate.py, or pass --allow_unrendered_marker to produce fiction.")

    def deliv(cond, mi=False):
        sx, sy = ("vel_x_mi", "vel_y_mi") if mi else ("vel_x", "vel_y")
        out = []
        for r in rows:
            b = np.array([r["base"][sx], r["base"][sy]])
            g = np.array([r["gt_vel"][sx], r["gt_vel"][sy]]) - b
            x = np.array([r[cond][sx], r[cond][sy]]) - b
            if np.isfinite(g @ g) and g @ g > 0 and np.all(np.isfinite(x)):
                out.append(float(x @ g / (g @ g)))
        return np.array(out)

    def spin_dev(cond):
        rel, ab = [], []
        for r in rows:
            ob, oc = r["base"]["omega"], r[cond]["omega"]
            if not (np.isfinite(ob) and np.isfinite(oc)):
                continue
            ab.append(abs(oc - ob))
            if abs(ob) > 1e-6:
                rel.append(abs(oc - ob) / abs(ob))
        return np.array(rel), np.array(ab)

    conds = ["gt_vel"] + [f"V_g{g:g}" for g in gains] + ["rand"]
    summary = {"n_squares": len(rows), "layer": L, "decoder": args.checkpoint,
               "marker_mass_decoded_over_rendered": smear, "conditions": {}}
    for c in conds:
        d, dmi = deliv(c), deliv(c, mi=True)
        rel, ab = spin_dev(c)
        summary["conditions"][c] = {
            "delivered_median": round(float(np.median(d)), 4) if d.size else None,
            "delivered_median_marker_invariant": round(float(np.median(dmi)), 4) if dmi.size else None,
            "spin_dev_rel_median": round(float(np.median(rel)), 4) if rel.size else None,
            "spin_dev_abs_median": round(float(np.median(ab)), 6) if ab.size else None,
            "n": int(d.size),
        }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print(f"\n# Velocity steering: BOTH quantities in pixels ({len(rows)} held-out squares, layer {L})\n")
    print("| condition | velocity delivered | spin dev (rel) | spin dev (rad/frame) |")
    print("|---|---|---|---|")
    for c in conds:
        v = summary["conditions"][c]
        f = lambda x, n=3: "n/a" if x is None else f"{x:+.{n}f}"
        print(f"| `{c}` | {f(v['delivered_median'])} | {f(v['spin_dev_rel_median'])} "
              f"| {f(v['spin_dev_abs_median'], 5)} |")
    print("\nBARS: velocity delivered >= 0.80-0.90; spin deviation <= 0.10-0.20 relative.")
    print("`gt_vel` is the REAL clip at the commanded velocity -- same spin by construction, so its")
    print("spin deviation is the readout floor. A steered deviation at or below it is not measurable.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
