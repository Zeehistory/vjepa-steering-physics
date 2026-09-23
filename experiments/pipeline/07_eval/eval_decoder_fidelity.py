

#!/usr/bin/env python
"""Score a decoder's reconstruction fidelity against ground-truth frames, per scene.

Answers one question: does `decode(H)` look like the clip H was encoded from? That is the ceiling
on any steering filmstrip -- a steered frame can never look better than an unsteered reconstruction.

Reports PSNR, SSIM and LPIPS over held-out clips, plus the same three restricted to the foreground
(the pixels the GT says are object). The foreground numbers are the ones that matter here: these
scenes are mostly flat background, so a decoder that renders a clean background and a mushy ball
still posts a flattering whole-frame PSNR.

    PYTHONPATH=. python experiments/pipeline/07_eval/eval_decoder_fidelity.py \
        --config configs/train/hifi_v2d_mixed_decoder.yaml \
        --test_dir ... --checkpoint ... [--label hifi] [--num_clips 48] --out FILE.json
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

from src.decoders import build_decoder
from src.decoders.loss_functions import ssim as ssim_fn
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config


def _psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    se = (a - b) ** 2
    if mask is not None:
        m = mask.expand_as(se)
        if m.sum() < 1:
            return float("nan")
        mse = se[m].mean()
    else:
        mse = se.mean()
    mse = float(mse.clamp_min(1e-12))
    return float(10.0 * np.log10(1.0 / mse))


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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="decoder")
    p.add_argument("--num_clips", type=int, default=48)
    p.add_argument("--fg_thresh", type=float, default=0.5,
                   help="GT luma below this counts as foreground (the pipeline's darkness rule)")
    p.add_argument("--use_ema", action="store_true",
                   help="decode with the EMA shadow weights instead of the last iterate")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = args.device
    ds = LatentDataset(args.test_dir, layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())

    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head_w = [v for k, v in ck["model"].items() if k.endswith("state_head.3.weight")]
    if head_w and int(head_w[0].shape[0]) != state_dim:
        state_dim = int(head_w[0].shape[0])       # older decoders were trained on a narrower schema
    del ck
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(args.checkpoint, decoder, map_location=device)
    used_ema = _maybe_load_ema(args.checkpoint, decoder, device, args.use_ema)

    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:  # noqa: BLE001
        print(f"[fidelity] LPIPS unavailable ({e}); reporting PSNR/SSIM only")

    rows = []
    n = min(args.num_clips, len(ds))
    for i in range(n):
        s = ds[i]
        grid = tuple(int(x) for x in s["grid"])
        batch = latent_collate([s])
        lat = {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}
        pred = decoder(lat, grid).frames[0].clamp(0, 1)                     # (T,C,H,W)
        gt = s["frames"]
        gt = (gt if torch.is_tensor(gt) else torch.from_numpy(np.asarray(gt))).float().clamp(0, 1)
        gt = gt.to(device)
        if gt.shape != pred.shape:
            gt = torch.nn.functional.interpolate(gt, size=pred.shape[-2:], mode="bilinear",
                                                 align_corners=False)
        luma = (gt * torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)).sum(
            1, keepdim=True)
        fg = luma < args.fg_thresh
        row = {"clip": s["id"],
               "psnr": _psnr(pred, gt),
               "psnr_fg": _psnr(pred, gt, fg),
               "ssim": float(ssim_fn(pred.unsqueeze(0), gt.unsqueeze(0))),
               "fg_frac": float(fg.float().mean())}
        if lp is not None:
            row["lpips"] = float(lp(pred * 2 - 1, gt * 2 - 1).mean())
        rows.append(row)
        if i < 3 or i % 16 == 0:
            print(f"  {row['clip']}: psnr {row['psnr']:.2f} dB  fg {row['psnr_fg']:.2f} dB  "
                  f"ssim {row['ssim']:.4f}" + (f"  lpips {row['lpips']:.4f}" if lp else ""))

    keys = [k for k in ("psnr", "psnr_fg", "ssim", "lpips") if k in rows[0]]
    summary = {k: {"mean": float(np.nanmean([r[k] for r in rows])),
                   "median": float(np.nanmedian([r[k] for r in rows]))} for k in keys}
    out = {"label": args.label, "checkpoint": args.checkpoint, "test_dir": args.test_dir,
           "config": args.config, "n_clips": len(rows),
           "frame_head": getattr(cfg.decoder, "frame_head", "patch"), "used_ema": used_ema,
           "fg_thresh": args.fg_thresh, "summary": summary, "per_clip": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n[fidelity] {args.label} over {len(rows)} clips: " +
          "  ".join(f"{k}={summary[k]['mean']:.4f}" for k in keys))
    print(f"[fidelity] wrote {args.out}")


if __name__ == "__main__":
    main()
