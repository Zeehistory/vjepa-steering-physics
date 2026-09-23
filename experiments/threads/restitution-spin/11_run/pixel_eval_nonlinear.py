

#!/usr/bin/env python
"""Does the nonlinear operator's latent win survive to PIXELS? The measurement the PI's bar is stated in.

**Why this is not optional.** The nonlinear per-token operator reaches latent alignment 0.804 with leak
0.071 on 48 held-out scenes, which meets both acceptance bars after calibration. But latent cosine has
already proved an OPTIMISTIC proxy on this exact scene: the cond128 operator scored 0.666 in latent and
delivered only 27% of the commanded velocity in decoded video, saturating under a gain sweep. The bar --
"velocity steered by 80-90% of the desired amount" -- is a statement about delivered motion, so it has
to be read off pixels.

**What is measured.** For each held-out commutation square, all conditions decode through the SAME
checkpoint and are compared to the DECODED base, so any constant bias of the decoder or the tracker
cancels instead of being charged to the steer:

    delivered = (v_edit - v_base) . (v_gt - v_base) / |v_gt - v_base|^2
    heading   = angle between (v_edit - v_base) and (v_gt - v_base)

``v_gt`` comes from decoding the REAL clip that differs from the base only in its velocity cell, so
``delivered`` is a fraction of an achievable change rather than of a nominal command -- the decoder's own
fidelity limit cancels out of the ratio.

**The controls.** A norm-matched random edit, which on this decoder reads 0.001 delivered at 75 degrees;
anything near that is noise. And a gain sweep, because the linear operator's failure mode was
specifically that scaling the edit up stopped delivering velocity and started degrading heading -- if
the nonlinear operator scales cleanly where the linear one saturated, that is the substantive difference
and it should be visible here rather than asserted.

**The decoder renders velocity but not spin** (speed corr 0.942, heading 1.4 deg; marker mass 0.0002).
So this reports the VELOCITY half of the question honestly and reports no pixel number for spin
crosstalk -- the latent leak measurement stands on its own for that, and a spin number read off a
decoder that cannot draw the marker would be fiction.

    PYTHONPATH=. python experiments/threads/restitution-spin/11_run/pixel_eval_nonlinear.py \
        --config configs/train/spin_ball3d_decoder.yaml \
        --model .../analysis/spin_ball3d/nl_L12_h1024.pt \
        --checkpoint .../runs/spin_ball3d_decoder/checkpoints/last.pt \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --out .../analysis/spin_ball3d/pixel_nonlinear.json
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
sys.path.insert(0, str(_P(__file__).resolve().parent))

import numpy as np
import torch

from src.analysis import spin_ops as so
from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

from fit_nonlinear_spin import TokenMLP                       # noqa: E402
from fit_transport_spin import _phi_xl, _deployable_centers   # noqa: E402


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def _decode_measure(decoder, latents, grid):
    out = decoder(latents, grid)
    if out.frames is None:
        return {"vel_x": np.nan, "vel_y": np.nan}
    return measured_velocity(out.frames[0].cpu())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_scenes", type=int, default=24)
    ap.add_argument("--squares_per_scene", type=int, default=3)
    ap.add_argument("--n_vel", type=int, default=4)
    ap.add_argument("--n_spin", type=int, default=4)
    ap.add_argument("--gains", default="1,1.25,1.5,2,3")
    ap.add_argument("--max_cached_shards", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    dev = args.device
    ck = torch.load(args.model, map_location=dev, weights_only=False)
    L = int(ck["layer"])
    sigmas, Q = ck["sigmas"], ck["Q"]
    mu_t = torch.from_numpy(ck["mu_f"]).to(dev)
    sd_t = torch.from_numpy(ck["sd_f"]).to(dev)
    model = TokenMLP(ck["p_in"], ck["hidden"], 1024, ck["depth"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[pix] operator: layer {L}, p_in {ck['p_in']}, hidden {ck['hidden']}", flush=True)

    cfg = load_config(args.config, args.overrides)
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers,
                       max_cached_shards=args.max_cached_shards)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(dev).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=dev)

    scenes = vo.group_scenes(ds)
    sids = sorted(scenes)[: args.num_scenes]
    gains = [float(x) for x in args.gains.split(",") if x]
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
            sam = {k: ds[i] for k, i in sq.items()}
            grid = tuple(int(x) for x in sam["base"]["grid"])
            T, H, W = grid
            Ha = _to_dev(sam["base"], layers, dev)

            va, vb = vo.clip_velocity(sam["base"]), vo.clip_velocity(sam["vel_only"])
            base_flat = vo.layer_flat(sam["base"]["layers"][L]).reshape(T * H * W, 1024)
            tgt = _deployable_centers(sam["base"], vb, grid)
            phi = _phi_xl(sam["base"], tgt, va, vb, grid, sigmas, Q, base_flat).astype(np.float32)
            with torch.no_grad():
                X = (torch.from_numpy(phi).to(dev) - mu_t) / sd_t
                e = model(X)                                  # (T*H*W, 1024) torch on device

            def add(edit_t):
                """Add a per-token edit at layer L; other layers pass through untouched."""
                out = {}
                for Lk, t in Ha.items():
                    out[Lk] = t + edit_t.reshape(1, t.shape[1], t.shape[2]) if Lk == L else t
                return out

            Hb = _to_dev(sam["vel_only"], layers, dev)
            # THE CEILING CONTROL. The decoder reads layers [6,12,18,23]; the operator only edits L.
            # So swap in the GROUND-TRUTH layer-L latent and leave the other three at base -- the most
            # any single-layer-L edit can possibly deliver. If `gt_vel` reads ~1.0 but this reads ~0.03,
            # the operator is not failing; the measurement is asking one of four decoder inputs to carry
            # the whole velocity change while the other three still say "the ball moves at v_a".
            conds = {"base": _decode_measure(decoder, Ha, grid),
                     "gt_vel": _decode_measure(decoder, Hb, grid),
                     "gt_vel_Lonly": _decode_measure(
                         decoder, {Lk: (Hb[Lk] if Lk == L else t) for Lk, t in Ha.items()}, grid)}
            # WHERE does decodable velocity actually live? Same swap, one decoder layer at a time, then
            # cumulatively. Layer 12 was chosen on latent alignment, which is blind to the decoder; if
            # some other layer carries the velocity the decoder actually reads, that is where to fit.
            for Lk in layers:
                conds[f"gt_only_L{Lk}"] = _decode_measure(
                    decoder, {k: (Hb[k] if k == Lk else t) for k, t in Ha.items()}, grid)
            for j in range(2, len(layers) + 1):
                keep = set(layers[:j])
                conds[f"gt_cum_{layers[j - 1]}"] = _decode_measure(
                    decoder, {k: (Hb[k] if k in keep else t) for k, t in Ha.items()}, grid)
            for g in gains:
                conds[f"V_g{g:g}"] = _decode_measure(decoder, add(g * e), grid)
            r = torch.randn_like(e)
            r = r * (e.norm() / (r.norm() + 1e-12))
            conds["rand"] = _decode_measure(decoder, add(r), grid)

            rows.append({"scene": int(s), "cond": {k: {"vel_x": float(v["vel_x"]),
                                                       "vel_y": float(v["vel_y"])}
                                                   for k, v in conds.items()}})
        if (n + 1) % 4 == 0:
            print(f"[pix] scene {n + 1}/{len(sids)}  squares={len(rows)}", flush=True)

    # --- summarise ---------------------------------------------------------------------------------
    def stat(cond):
        fr, hd = [], []
        for r in rows:
            c = r["cond"]
            b = np.array([c["base"]["vel_x"], c["base"]["vel_y"]])
            g = np.array([c["gt_vel"]["vel_x"], c["gt_vel"]["vel_y"]])
            x = np.array([c[cond]["vel_x"], c[cond]["vel_y"]])
            dg, dx = g - b, x - b
            ng = np.linalg.norm(dg)
            if not np.isfinite(ng) or ng < 1e-9 or not np.isfinite(dx).all():
                continue
            fr.append(float(dx @ dg / (ng * ng)))
            cs = float(dx @ dg / (np.linalg.norm(dx) * ng + 1e-12))
            hd.append(float(np.degrees(np.arccos(np.clip(cs, -1, 1)))))
        return (float(np.median(fr)), float(np.median(hd)), len(fr)) if fr else (np.nan, np.nan, 0)

    summary = {"n_squares": len(rows), "layer": L, "model": str(args.model), "conditions": {}}
    sweep = [c for c in rows[0]["cond"] if c.startswith(("gt_only_", "gt_cum_"))] if rows else []
    for cond in ["gt_vel", "gt_vel_Lonly"] + sweep + [f"V_g{g:g}" for g in gains] + ["rand"]:
        d, h, k = stat(cond)
        summary["conditions"][cond] = {"delivered": round(d, 4), "heading_deg": round(h, 2), "n": k}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print(f"\n# Nonlinear operator in PIXELS ({len(rows)} held-out squares, layer {L})\n")
    print("| condition | delivered | heading err |")
    print("|---|---|---|")
    for cond, d in summary["conditions"].items():
        print(f"| {cond} | {d['delivered']:+.3f} | {d['heading_deg']:.1f} deg |")
    print("\nBar: 80-90% delivered. Linear reference on this same decoder: 0.270 delivered at best gain,")
    print("saturating (0.265 at g6, 0.270 at g8) with heading degrading past g3.")
    print("`rand` is norm-matched and read 0.001 at 74.9 deg for the linear operator -- the noise floor.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
