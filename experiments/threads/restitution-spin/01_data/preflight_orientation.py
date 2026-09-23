

#!/usr/bin/env python
"""Is the projected orientation loss SUPERVISING or trainng against noise? Check before spending GPU.

``frame_orientation_projected_loss`` compares a differentiable detector reading of the PREDICTION's
marker direction against a target built from the scene's ellipse geometry. Both halves were certified
on RENDERED frames (target within 0.2 px of the tracker, detector 4.7 deg from GT theta). Neither claim
transfers automatically to DECODED frames, and that gap is where the previous attempt died: the
``_phase`` run trained this term for 19k steps and produced green mush with omega_corr 0.28.

The specific hazard is a chicken-and-egg: the detector reads a direction off the marker in the
PREDICTION. Our best velocity decoder renders no marker at all (mass 0.000), so its detector output is
a direction fitted to nothing, and optimizing toward the target from there is optimizing noise. A
decoder that already draws *a* marker -- even a static, faint one -- is a different starting point.

So this reports, for each candidate init, on held-out clips:
  * ``target_vs_gt_theta`` -- sanity on the TARGET alone (independent of any decoder);
  * ``detector_on_rendered`` -- the certified 4.7 deg number, recomputed here as a control;
  * ``detector_on_decoded`` -- the number that actually matters: how far the detector's reading of THIS
    decoder's frames is from the target. Near 90 deg means no usable signal to descend;
  * ``body_on_ball_frac`` -- how often the detector's rotation CENTRE lands on the ball rather than in
    the room, which is the documented way this detector fails.

A candidate is a sane init if the detector tracks the ball (``body_on_ball_frac`` high) even when the
direction is wrong -- direction being wrong is what training is for; the CENTRE being wrong means the
gradient is meaningless.

    PYTHONPATH=.:scripts/spin python experiments/threads/restitution-spin/01_data/preflight_orientation.py --config ... \
        --checkpoints name=path[,name=path...] --test_dir ...
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

from src.analysis import spin_tracking as _st
from src.decoders.loss_functions import _soft_marker_direction
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def projected_target(state_np: np.ndarray, keys: list[str]) -> np.ndarray:
    """(T,2) unit target direction per frame, same construction as the loss."""
    ti, xi, yi = keys.index("obj0_theta"), keys.index("obj0_pos_x"), keys.index("obj0_pos_y")
    theta, px, py = state_np[:, ti], state_np[:, xi], state_np[:, yi]
    wc = _st.unproject_to_ball_plane(np.stack([px, py], axis=1))
    s_, cz_ = np.sin(_st.MARKER_POLAR), np.cos(_st.MARKER_POLAR)
    off = _st.MARKER_OFFSET
    p0 = _st.project(wc)
    pC = _st.project(wc + np.array([0.0, 0.0, off * cz_]))
    pA = _st.project(wc + np.array([off * s_, 0.0, off * cz_]))
    pB = _st.project(wc + np.array([0.0, off * s_, off * cz_]))
    C = pC - p0
    A = pA - p0 - C
    B = pB - p0 - C
    d = A * np.cos(theta)[:, None] + B * np.sin(theta)[:, None] + C
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def ang_err_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    c = np.clip((u * v).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def body_centre_on_ball(frames: torch.Tensor, thresh: float = 0.5) -> float:
    """Fraction of frames whose darkness-gated centre lands inside the ball's dark blob."""
    x = frames.detach().cpu().float()
    gray = x.mean(dim=1)
    dark = (1.0 - gray)
    mask = (dark > thresh).float()
    T, H, W = mask.shape
    ys = torch.linspace(0, 1, H).view(1, H, 1)
    xs = torch.linspace(0, 1, W).view(1, 1, W)
    m = mask.sum(dim=(1, 2))
    ok = []
    for t in range(T):
        if m[t] < 1:
            ok.append(0.0); continue
        cx = float((mask[t] * xs[0]).sum() / m[t])
        cy = float((mask[t] * ys[0]).sum() / m[t])
        iy, ix = int(round(cy * (H - 1))), int(round(cx * (W - 1)))
        ok.append(float(mask[t, iy, ix] > 0.5))       # centre sits ON dark material
    return float(np.mean(ok))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", required=True, help="name=path,name=path")
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--n_clips", type=int, default=24)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    cfg = load_config(args.config)
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers, max_cached_shards=2)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    keys = list(ds[0]["state_keys"])
    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames

    n = min(args.n_clips, len(ds))
    # --- target sanity + detector on RENDERED frames (no decoder involved) --------------------------
    tgt_all, rend_err, rend_body = [], [], []
    for i in range(n):
        s = ds[i]
        stn = np.asarray(s["state"], dtype=np.float64)
        if stn.ndim == 1:
            continue
        tgt = projected_target(stn, keys)
        tgt_all.append(tgt)
        g = s["frames"].float()
        g = g / 255.0 if g.max() > 1.5 else g
        u = _soft_marker_direction(g.unsqueeze(0), body_thresh=0.5, marker_thresh=0.18)[0].numpy()
        rend_err.append(ang_err_deg(u, tgt))
        rend_body.append(body_centre_on_ball(g))
    rend_err = np.concatenate(rend_err) if rend_err else np.array([np.nan])

    out = {"n_clips": n,
           "detector_on_rendered_median_deg": round(float(np.nanmedian(rend_err)), 2),
           "body_on_ball_frac_rendered": round(float(np.mean(rend_body)), 3),
           "target_sweep_deg": round(float(np.ptp(np.degrees(np.arctan2(
               np.concatenate(tgt_all)[:, 1], np.concatenate(tgt_all)[:, 0])))), 1),
           "decoders": {}}

    for spec in args.checkpoints.split(","):
        name, path = spec.split("=", 1)
        dec = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(dev).eval()
        if hasattr(dec, "prime_layers"):
            dec.prime_layers([int(x) for x in ds.available_layers()])
        load_checkpoint(path, dec, map_location=dev)
        errs, bodies, masses = [], [], []
        for i in range(n):
            s = ds[i]
            stn = np.asarray(s["state"], dtype=np.float64)
            if stn.ndim == 1:
                continue
            grid = tuple(int(x) for x in s["grid"])
            batch = latent_collate([s])
            H = {int(k): v.to(dev) for k, v in batch["layers"].items() if int(k) in layers}
            res = dec(H, grid)
            if res.frames is None:
                continue
            fr = res.frames[0].cpu().clamp(0.0, 1.0)
            u = _soft_marker_direction(fr.unsqueeze(0), body_thresh=0.5, marker_thresh=0.18)[0].numpy()
            errs.append(ang_err_deg(u, projected_target(stn, keys)))
            bodies.append(body_centre_on_ball(fr))
            masses.append(float((fr[:, 0] - fr[:, 2] - _st.MARKER_RB_THRESH).clamp(min=0).sum()))
        e = np.concatenate(errs) if errs else np.array([np.nan])
        out["decoders"][name] = {
            "detector_on_decoded_median_deg": round(float(np.nanmedian(e)), 2),
            "body_on_ball_frac": round(float(np.mean(bodies)), 3),
            "warm_mass_median": round(float(np.median(masses)), 2),
        }
        print(f"[pre] {name}: detector {out['decoders'][name]['detector_on_decoded_median_deg']} deg, "
              f"body-on-ball {out['decoders'][name]['body_on_ball_frac']}, "
              f"warm mass {out['decoders'][name]['warm_mass_median']}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))

    print("\n# Preflight: projected orientation loss\n")
    print(f"target direction sweeps {out['target_sweep_deg']} deg over a clip "
          "(near 360 = genuinely phase-informative)")
    print(f"detector on RENDERED frames: {out['detector_on_rendered_median_deg']} deg "
          f"(certified 4.7); body-on-ball {out['body_on_ball_frac_rendered']}\n")
    print("| init | detector on decoded | body-on-ball | warm mass |")
    print("|---|---|---|---|")
    for k, v in out["decoders"].items():
        print(f"| {k} | {v['detector_on_decoded_median_deg']:.1f} deg | {v['body_on_ball_frac']:.2f} "
              f"| {v['warm_mass_median']:.1f} |")
    print("\nREAD: body-on-ball near 1 means the rotation CENTRE is on the ball, so the gradient is")
    print("meaningful even where the direction is still wrong -- that is what training fixes. A low")
    print("body-on-ball, or a decoder with no marker mass, means there is nothing to orient and the")
    print("term would descend on noise, which is how the previous attempt reached green mush.")


if __name__ == "__main__":
    main()
