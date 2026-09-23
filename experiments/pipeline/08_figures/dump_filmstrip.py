

#!/usr/bin/env python
"""Dump the frames behind a Real-vs-Decoded-steered filmstrip, for any quantity.

For each held-out scene it takes the scene's anchor clip ``a`` (rank 0) and target clip ``b``
(last rank) -- the same pair every steering script scores -- and writes four aligned frame stacks:

  real_a       ground-truth frames of the anchor clip                     (the starting condition)
  real_b       ground-truth frames of the target clip                     ("Real": the commanded outcome)
  dec_a        decode(H_a)                                               (decoder's unsteered recon)
  dec_steer    decode(H_a + g * (phi(q_a,q_b) W_U) U)                     ("Steered": command-only edit)

The edit is byte-for-byte the ``cmd_U8_s{g}`` method of ``steer_velocity2d.py`` /
``steer_accel2d.py`` / ``steer_angvel2d.py``: command-only synthesis in the global subspace U, at
the gain those scripts selected on a disjoint validation half. ``dec_a`` is included because it is
the honest control -- it separates "the decoder cannot draw this scene" from "the edit did not
land".

Nothing here is fit; it only replays saved operators. Usage::

    PYTHONPATH=. python experiments/pipeline/08_figures/dump_filmstrip.py --config ... --test_dir ... \
        --artifacts_dir ... --checkpoint ... --output_dir ... --quantity velocity --gain 2.0
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

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import (measured_acceleration, measured_angvel,
                                        measured_velocity)
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

# quantity -> (clip accessor, pixel readout, human label)
QUANT = {
    "velocity": (vo.clip_velocity, measured_velocity, "velocity"),
    "accel": (vo.clip_acceleration, measured_acceleration, "acceleration"),
    "gravity": (vo.clip_acceleration, measured_acceleration, "gravity"),
    "angvel": (vo.clip_angvel, measured_angvel, "angular velocity"),
}


def _to_dev(sample, layers, device):
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


def _apply_edit(Ha, edit_flat, grid, device):
    T, H, W = grid
    out = {}
    for L, t in Ha.items():
        Ltok = t.shape[1]
        e = torch.from_numpy(np.asarray(edit_flat[L], dtype=np.float32)).to(device)
        out[L] = t + e.reshape(1, Ltok, -1)
    return out


@torch.no_grad()
def _decode(decoder, latents, grid):
    out = decoder(latents, grid)
    if out.frames is None:
        return None
    return out.frames[0].detach().cpu().float().clamp(0, 1).numpy()   # (T,C,H,W)


def _readout(fn, frames_np):
    """Run the pixel readout on a (T,C,H,W) numpy stack, tolerating readout failures."""
    if frames_np is None:
        return {}
    try:
        m = fn(torch.from_numpy(frames_np))
        return {k: (float(v) if np.isscalar(v) or isinstance(v, (int, float)) else v)
                for k, v in m.items() if isinstance(v, (int, float, np.floating))}
    except Exception as e:  # noqa: BLE001 - a readout hiccup must not lose the frames
        return {"readout_error": str(e)[:200]}


def _maybe_load_ema(checkpoint: str, decoder, device: str, use_ema: bool) -> bool:
    """Swap in the EMA shadow weights, which the training loop keeps but nothing was reading.

    With ema_decay 0.9999 over a long run the shadow is a much smoother point than the last SGD
    iterate, and for a reconstruction decoder that shows up directly as less speckle. Returns
    whether the swap happened.
    """
    if not use_ema:
        return False
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema = ck.get("ema")
    if not ema:
        print("[ema] checkpoint has no EMA shadow; using raw weights")
        return False
    missing = decoder.load_state_dict({k: v.to(device) for k, v in ema.items()}, strict=False)
    print(f"[ema] loaded EMA shadow ({len(ema)} tensors)"
          + (f"; missing={len(missing.missing_keys)}" if missing.missing_keys else ""))
    return True


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--quantity", required=True, choices=sorted(QUANT))
    p.add_argument("--gain", type=float, required=True,
                   help="the gain selected on the validation half by the matching steer_*.py")
    p.add_argument("--cmd_ku", type=int, default=8)
    p.add_argument("--cmd_std", action="store_true",
                   help="use the STANDARDIZED-ridge command operator (cmd_Wu_std*_L*.npy), which "
                        "actually tracks commanded SPEED; the default fit steers heading only.")
    p.add_argument("--num_scenes", type=int, default=6)
    p.add_argument("--pair_mode", default="rank_extremes",
                   choices=["rank_extremes", "max_mag_ratio"],
                   help="Which two of a scene's 8 ranks to show. 'rank_extremes' (default) is "
                        "rank 0 -> last rank, the pair every steering script SCORES. "
                        "'max_mag_ratio' instead picks the pair maximising |q_b|/|q_a|, for a "
                        "figure where the commanded change is visible in SPEED and not only in "
                        "heading -- illustrative only, NOT the scored pair.")
    p.add_argument("--use_ema", action="store_true",
                   help="decode with the EMA shadow weights instead of the last iterate")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts_dir)
    device = args.device
    clip_q, readout, qlabel = QUANT[args.quantity]

    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)
    scene_ids = sorted(scenes)[: args.num_scenes]
    print(f"[filmstrip] {args.quantity}: {len(scenes)} scenes cached, dumping {len(scene_ids)}; "
          f"layers={layers}")

    tag = "" if args.cmd_ku == 8 else f"_ku{args.cmd_ku}"
    if args.cmd_std:
        tag = "_std" + tag
    Wu = {L: np.load(art / f"cmd_Wu{tag}_L{L}.npy").astype(np.float64) for L in layers}
    Ubasis = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    # The decoder's auxiliary state head must be built at the width it was TRAINED at, which is not
    # necessarily the cache's state_dim: `_state_keys()` grew from 12 -> 14 -> 15 columns over the
    # project, so a decoder trained last spring against a cache re-extracted today mismatches on
    # `state_head.3` alone. The head is auxiliary -- frame decoding never reads it -- so we take the
    # width from the checkpoint and leave everything else untouched.
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_sd = ck["model"]
    head_w = [v for k, v in ck_sd.items() if k.endswith("state_head.3.weight")]
    if head_w and int(head_w[0].shape[0]) != state_dim:
        print(f"[filmstrip] state_head width {int(head_w[0].shape[0])} in checkpoint vs "
              f"{state_dim} in cache -> building the head at the checkpoint's width")
        state_dim = int(head_w[0].shape[0])
    del ck, ck_sd
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)
    used_ema = _maybe_load_ema(args.checkpoint, decoder, device, args.use_ema)

    manifest = {"quantity": args.quantity, "quantity_label": qlabel, "gain": args.gain,
                "cmd_ku": args.cmd_ku, "cmd_std": bool(args.cmd_std), "layers": layers, "test_dir": args.test_dir,
                "checkpoint": args.checkpoint, "artifacts_dir": str(art), "used_ema": used_ema,
                "edit": "H_a + gain * (phi(q_a,q_b) @ W_U) @ U[:ku]  (command-only, no H_b)",
                "scenes": {}}

    for s in scene_ids:
        ranks = sorted(scenes[s])
        if args.pair_mode == "max_mag_ratio":
            # Speeds are permuted against heading by the generators (deliberately, to decorrelate
            # the two), so rank 0 -> last rank lands on an arbitrary magnitude pair -- often two
            # near-equal speeds differing only in direction. Pick the pair that actually separates
            # in magnitude instead.
            mags = {r: float(np.linalg.norm(np.atleast_1d(clip_q(ds[scenes[s][r]]))))
                    for r in ranks}
            ra = min(mags, key=mags.get)
            rb = max(mags, key=mags.get)
        else:
            ra, rb = ranks[0], ranks[-1]
        ia, ib = scenes[s][ra], scenes[s][rb]
        sa, sb = ds[ia], ds[ib]
        grid = tuple(int(x) for x in sa["grid"])
        qa, qb = clip_q(sa), clip_q(sb)

        phi = vo.command_features(qa, qb)
        edit = {L: args.gain * ((phi @ Wu[L]) @ Ubasis[L][: Wu[L].shape[1]]) for L in layers}

        Ha = _to_dev(sa, layers, device)
        dec_a = _decode(decoder, Ha, grid)
        dec_steer = _decode(decoder, _apply_edit(Ha, edit, grid, device), grid)

        def _gt(sample):
            fr = sample["frames"]
            fr = fr.detach().cpu().float().numpy() if torch.is_tensor(fr) else np.asarray(fr)
            return fr.clip(0, 1)

        real_a, real_b = _gt(sa), _gt(sb)
        np.savez_compressed(out / f"scene{s:05d}.npz",
                            real_a=real_a.astype(np.float16),
                            real_b=real_b.astype(np.float16),
                            dec_a=dec_a.astype(np.float16),
                            dec_steer=dec_steer.astype(np.float16))
        manifest["scenes"][f"scene{s:05d}"] = {
            "q_a": np.asarray(qa).tolist(), "q_b": np.asarray(qb).tolist(),
            "rank_a": int(ra), "rank_b": int(rb), "pair_mode": args.pair_mode,
            "readout_real_b": _readout(readout, real_b),
            "readout_dec_a": _readout(readout, dec_a),
            "readout_dec_steer": _readout(readout, dec_steer),
        }
        r = manifest["scenes"][f"scene{s:05d}"]
        shown = {k: round(v, 5) for k, v in r["readout_dec_steer"].items()
                 if isinstance(v, float)}
        print(f"  scene{s:05d}: q_b={np.round(qb, 5).tolist()} steered_readout={shown}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[filmstrip] wrote {len(scene_ids)} scenes + manifest.json -> {out}")


if __name__ == "__main__":
    main()
