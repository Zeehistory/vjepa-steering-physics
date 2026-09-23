

#!/usr/bin/env python
"""Can the SPIN be read off the DECODER's output at all? Run this before trusting any steering number.

The pixel readouts were certified against ground truth on RENDERED frames
(``experiments/threads/restitution-spin/01_data/validate_spin_ball3d.py``). That is not the same claim as being able to read them off
the DECODER's frames, and the difference is exactly where this experiment could fail silently.

Nothing in the decoder's loss constrains the marker's phase. ``frame_position`` pins the ball's soft
centroid, ``trajectory``/``velocity`` constrain the auxiliary state head, and the reconstruction terms
reward the temporal average -- so a decoder can drive its loss down while smearing the marker into a
phase-averaged ring around the ball. That is the rotational analogue of the translation smear that
``frame_position`` was introduced to kill. A smeared marker still yields *a* number from
``measured_spin``: an essentially random slope through noise. Steering numbers computed on top of it
would be meaningless and would look like "spin steering does not work" rather than "spin is not
rendered", which are very different conclusions.

So this gate decodes REAL held-out clips -- no steering, no edits -- and asks only whether the two
quantities survive the round trip through the decoder:

  * ``omega`` recovered from decoded frames vs the clip's true omega: correlation, median absolute
    error, and the fraction of clips where the azimuth is readable at all;
  * ``|v|`` and heading likewise, as the already-trusted control -- velocity decoding is known to work
    on this decoder family, so if velocity also fails the problem is the checkpoint, not the marker;
  * the marker's warm-pixel mass on decoded vs rendered frames, which is the direct measure of smear.

Exits non-zero if the spin channel does not clear the gate, so a chained pipeline stops here instead of
producing a confident null result downstream.

    PYTHONPATH=. python experiments/threads/restitution-spin/06_probes/decoder_gate.py \
        --config configs/train/spin_ball3d_decoder.yaml \
        --test_dir .../latents/spin_ball3d/test/vjepa2_large \
        --checkpoint .../runs/spin_ball3d_decoder/checkpoints/last.pt \
        --output_dir .../analysis/spin_ball3d/decoder_gate
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

# Deliberately permissive: this gate is not asking whether the decoder is good, only whether the spin
# channel carries SIGNAL. A correlation of 0.7 would be a poor decoder and a perfectly usable
# instrument for measuring a steering effect; a correlation of 0.1 means the experiment cannot be run.
GATES = {
    "omega_corr_min": 0.60,        # decoded-vs-true omega correlation across held-out clips
    "omega_readable_frac": 0.80,   # fraction of clips with >= 3 readable marker frames
    "speed_corr_min": 0.60,        # the velocity control, which is known to work on this decoder family
}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_clips", type=int, default=96)
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers, max_cached_shards=2)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())

    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    dec = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(args.device).eval()
    if hasattr(dec, "prime_layers"):
        dec.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, dec, map_location=args.device)
    print(f"[gate] decoder loaded from {args.checkpoint}", flush=True)

    rows = []
    n = min(args.num_clips, len(ds))
    for i in range(n):
        sam = ds[i]
        grid = tuple(int(x) for x in sam["grid"])
        batch = latent_collate([sam])
        H = {int(k): v.to(args.device) for k, v in batch["layers"].items() if int(k) in layers}
        res = dec(H, grid)
        if res.frames is None:
            continue
        fr = res.frames[0].cpu().clamp(0.0, 1.0)

        ms = st.measured_spin(fr)
        mv = measured_velocity(fr)
        # Reported ALONGSIDE, not instead of. The marker-invariant centroid is certified on ground-truth
        # renders (identical accuracy, omega dependence in the x-error 0.151 -> 0.009), but a binary
        # ball mask is a different instrument on BLURRY decoded frames than on crisp ones, and that has
        # not been certified. Carrying both lets the gate show whether they agree here; swapping first
        # and checking later is how an uncertified readout becomes the number of record.
        mv_mi = measured_velocity_marker_invariant(fr)
        warm_dec = float((fr[:, 0] - fr[:, 2] - st.MARKER_RB_THRESH).clamp(min=0).sum())
        gt_frames = sam.get("frames")
        warm_gt = float("nan")
        if gt_frames is not None and torch.is_tensor(gt_frames) and gt_frames.numel() > 1:
            g = gt_frames.float()
            if g.max() > 1.5:
                g = g / 255.0
            warm_gt = float((g[:, 0] - g[:, 2] - st.MARKER_RB_THRESH).clamp(min=0).sum())

        v_gt = vo.clip_velocity(sam)
        rows.append({
            "id": ds._ids[i],
            "omega_gt": so.clip_spin(sam), "omega_dec": ms["omega"],
            "omega_nvalid": ms["n_valid"], "omega_resid": ms["residual"],
            "speed_gt": float(np.linalg.norm(v_gt)),
            "speed_dec": float(np.hypot(mv["vel_x"], mv["vel_y"])),
            "head_gt": float(np.arctan2(v_gt[1], v_gt[0])),
            "head_dec": float(np.arctan2(mv["vel_y"], mv["vel_x"])),
            "speed_dec_mi": float(np.hypot(mv_mi["vel_x"], mv_mi["vel_y"])),
            "head_dec_mi": float(np.arctan2(mv_mi["vel_y"], mv_mi["vel_x"])),
            "warm_decoded": warm_dec, "warm_rendered": warm_gt,
        })
        if (i + 1) % 24 == 0:
            print(f"[gate] {i + 1}/{n}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    ok = np.isfinite(col("omega_dec"))
    readable = float((col("omega_nvalid") >= 3).mean())
    omega_corr = float(np.corrcoef(col("omega_gt")[ok], col("omega_dec")[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    omega_mae = float(np.median(np.abs(col("omega_dec")[ok] - col("omega_gt")[ok]))) if ok.sum() else float("nan")
    sp_ok = np.isfinite(col("speed_dec"))
    speed_corr = float(np.corrcoef(col("speed_gt")[sp_ok], col("speed_dec")[sp_ok])[0, 1]) if sp_ok.sum() > 2 else float("nan")
    dh = np.angle(np.exp(1j * (col("head_dec") - col("head_gt"))))
    head_mae_deg = float(np.degrees(np.median(np.abs(dh[sp_ok]))))
    sp_ok_mi = np.isfinite(col("speed_dec_mi"))
    speed_corr_mi = (float(np.corrcoef(col("speed_gt")[sp_ok_mi], col("speed_dec_mi")[sp_ok_mi])[0, 1])
                     if sp_ok_mi.sum() > 2 else float("nan"))
    dh_mi = np.angle(np.exp(1j * (col("head_dec_mi") - col("head_gt"))))
    head_mae_mi = float(np.degrees(np.median(np.abs(dh_mi[sp_ok_mi])))) if sp_ok_mi.sum() else float("nan")
    smear = float(np.nanmedian(col("warm_decoded") / np.maximum(col("warm_rendered"), 1e-9)))

    checks = {
        "omega_corr": (omega_corr, GATES["omega_corr_min"], omega_corr >= GATES["omega_corr_min"]),
        "omega_readable_frac": (readable, GATES["omega_readable_frac"], readable >= GATES["omega_readable_frac"]),
        "speed_corr": (speed_corr, GATES["speed_corr_min"], speed_corr >= GATES["speed_corr_min"]),
    }
    passed = all(c[2] for c in checks.values())
    summary = {"n_clips": len(rows), "checkpoint": args.checkpoint, "passed": bool(passed),
               "omega_corr": omega_corr, "omega_median_abs_err": omega_mae,
               "omega_readable_frac": readable, "speed_corr": speed_corr,
               "heading_median_abs_err_deg": head_mae_deg,
               "speed_corr_marker_invariant": speed_corr_mi,
               "heading_median_abs_err_deg_marker_invariant": head_mae_mi,
               "marker_mass_decoded_over_rendered": smear, "gates": GATES}
    (out / "decoder_gate.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))

    print(f"\n# decoder gate ({len(rows)} held-out clips) -> {'PASS' if passed else 'FAIL'}\n")
    print("| check | value | limit | pass |")
    print("|---|---|---|---|")
    for k, (v, lim, good) in checks.items():
        print(f"| `{k}` | {v:.4f} | {lim} | {'yes' if good else '**NO**'} |")
    print(f"\nomega median |err| = {omega_mae:.4f} rad/frame; heading median |err| = {head_mae_deg:.1f} deg")
    print(f"marker-invariant readout: speed corr {speed_corr_mi:.4f}, heading |err| {head_mae_mi:.1f} deg"
          "  (uncertified on decoded frames; shown for comparison, gate uses the darkness readout)")
    print(f"marker mass decoded/rendered = {smear:.3f}  (<<1 means the marker is being smeared away)")
    print(f"\nwrote {out/'decoder_gate.json'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
