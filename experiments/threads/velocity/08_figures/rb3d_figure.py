"""Visual proof of 3D rolling-ball velocity steering: decode H_a, H_a + cmd-U8 edit, and the targets.

The steering numbers (angle error vs a target velocity) are the evidence, but they are abstract. This
renders what actually happened, per scene, as a filmstrip with four rows:

    GT clip a        the real video the latent H_a came from (the ball rolls one way)
    DECODE(H_a)      reconstruct H_a untouched -> the decoder's baseline, still rolling v_a
    DECODE(H_a+e)    H_a edited by the COMMAND-ONLY operator e = g * (phi(v_a,v_b) @ W_U) @ U
                     -> the ball should now roll along v_b, having never seen H_b
    GT clip b        the real video for v_b -- the counterfactual we are trying to synthesize

Row 3 vs row 4 is the result; row 3 vs row 2 is the intervention's effect. Arrows/tracks are overlaid
from the same darkness-centroid tracker that produces the metric, so the picture and the number cannot
disagree.

    python experiments/threads/velocity/08_figures/rb3d_figure.py --scenes 4 --gain 2.0 [--cmd_ku 16]
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
from pathlib import Path

import numpy as np
import torch

from src.analysis import velocity_ops as vo
from src.analysis.ball_tracking import ball_centroids, measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

BASE = "."
LAT_BASE = "."


def _strip(frames: torch.Tensor, idx: list[int]) -> np.ndarray:
    """(T,C,H,W) in [0,1] -> a horizontal uint8 filmstrip of the chosen frames."""
    fr = frames.detach().cpu().clamp(0, 1).numpy()
    return np.concatenate([(fr[t].transpose(1, 2, 0) * 255).astype(np.uint8) for t in idx], axis=1)


def _overlay_track(strip: np.ndarray, frames: torch.Tensor, idx: list[int], rgb: tuple) -> np.ndarray:
    """Mark the tracked ball centre in each panel, so the motion is legible in a still image."""
    c = ball_centroids(frames)
    h, w = frames.shape[-2:]
    out = strip.copy()
    for panel, t in enumerate(idx):
        if not np.isfinite(c[t]).all():
            continue
        cx = int(c[t][0] * w) + panel * w
        cy = int(c[t][1] * h)
        r = 5
        y0, y1 = max(0, cy - r), min(out.shape[0], cy + r + 1)
        x0, x1 = max(0, cx - r), min(out.shape[1], cx + r + 1)
        out[y0:y1, x0:x1] = (0.35 * out[y0:y1, x0:x1] + 0.65 * np.array(rgb)).astype(np.uint8)
    return out


def _label_bar(width: int, text: str, height: int = 22) -> np.ndarray:
    """A simple text bar (drawn with PIL if available, else a plain separator)."""
    bar = np.full((height, width, 3), 245, np.uint8)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(bar)
        ImageDraw.Draw(im).text((6, 5), text, fill=(20, 20, 20))
        return np.array(im)
    except Exception:  # noqa: BLE001 - the figure is still readable without labels
        return bar


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--gain", type=float, default=2.0, help="cmd-U8 gain (the calibrated one)")
    p.add_argument("--cmd_ku", type=int, default=8)
    p.add_argument("--out", default="scratchpad/rb3d_steer_proof.png")
    p.add_argument("--dump", default=None,
                   help="directory to also write the FULL 16-frame clips per scene as npz "
                        "(gt_a / dec_a / dec_s / gt_b) -- what the slide GIFs are built from")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = load_config("configs/train/rolling_ball3d_decoder.yaml", [])
    art = Path(f"{BASE}/outputs/analysis/rolling_ball3d/subspace")
    ds = LatentDataset(f"{LAT_BASE}/outputs/latents/rolling_ball3d/test/vjepa2_large",
                       layers=cfg.encoder.layers)
    layers = sorted(int(k) for k in ds[0]["layers"].keys())
    scenes = vo.group_scenes(ds)

    tag = "" if args.cmd_ku == 8 else f"_ku{args.cmd_ku}"
    Wu = {L: np.load(art / f"cmd_Wu{tag}_L{L}.npy").astype(np.float64) for L in layers}
    U = {L: np.load(art / f"global_basis_L{L}.npy").astype(np.float64) for L in layers}

    rec0 = ds.records[0]
    cfg.decoder.state_dim = int(rec0["state_dim"])
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    dec = build_decoder(cfg.decoder, int(rec0["hidden_dim"]), int(rec0["state_dim"])).to(args.device).eval()
    if hasattr(dec, "prime_layers"):
        dec.prime_layers([int(x) for x in ds.available_layers()])
    load_checkpoint(f"{LAT_BASE}/outputs/runs/rolling_ball3d_decoder_fp/checkpoints/last.pt",
                    dec, map_location=args.device)

    idx = [0, 5, 10, 15]
    rows: list[np.ndarray] = []
    for s in sorted(scenes)[: args.scenes]:
        ranks = sorted(scenes[s])
        sa, sb = ds[scenes[s][ranks[0]]], ds[scenes[s][ranks[-1]]]
        grid = tuple(int(x) for x in sa["grid"])
        va, vb = vo.clip_velocity(sa), vo.clip_velocity(sb)
        phi = vo.command_features(va, vb)

        Ha = {L: sa["layers"][L].to(args.device) for L in layers}
        edit = {L: args.gain * ((phi @ Wu[L]) @ U[L][: Wu[L].shape[1]]) for L in layers}
        Hs = {L: (Ha[L].reshape(-1) + torch.tensor(edit[L], device=args.device,
                                                   dtype=Ha[L].dtype)).reshape(Ha[L].shape)
              for L in layers}

        dec_a = dec({L: Ha[L].unsqueeze(0) for L in layers}, grid).frames[0].cpu()
        dec_s = dec({L: Hs[L].unsqueeze(0) for L in layers}, grid).frames[0].cpu()
        gt_a, gt_b = sa["frames"], sb["frames"]

        def _ang(clip):
            """Angle (deg) between the velocity tracked out of `clip` and the commanded v_b."""
            mm = measured_velocity(clip)
            v = np.array([mm["vel_x"], mm["vel_y"]])
            c = v @ vb / (np.linalg.norm(v) * np.linalg.norm(vb) + 1e-12)
            return v, float(np.degrees(np.arccos(np.clip(c, -1, 1))))

        v_dec, ang = _ang(dec_s)
        v_dec_a, ang_a = _ang(dec_a)          # the BEFORE panel's error, for the GIF caption

        if args.dump:
            d = Path(args.dump)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                d / f"viz_scene{s:05d}.npz",
                gt_a=gt_a.numpy().astype(np.float16), dec_a=dec_a.numpy().astype(np.float16),
                dec_s=dec_s.numpy().astype(np.float16), gt_b=gt_b.numpy().astype(np.float16),
                v_a=np.asarray(va, np.float32), v_b=np.asarray(vb, np.float32),
                v_dec=v_dec.astype(np.float32), ang_deg=np.float32(ang),
                v_dec_a=v_dec_a.astype(np.float32), ang_deg_a=np.float32(ang_a),
                gain=np.float32(args.gain))

        w = gt_a.shape[-1] * len(idx)
        rows += [
            _label_bar(w, f"scene {s:05d}   v_a=({va[0]:+.4f},{va[1]:+.4f})  ->  "
                          f"v_b=({vb[0]:+.4f},{vb[1]:+.4f})   steered decode err = {ang:.1f} deg"),
            _overlay_track(_strip(gt_a, idx), gt_a, idx, (60, 60, 200)),
            _overlay_track(_strip(dec_a, idx), dec_a, idx, (60, 60, 200)),
            _overlay_track(_strip(dec_s, idx), dec_s, idx, (220, 40, 40)),
            _overlay_track(_strip(gt_b, idx), gt_b, idx, (220, 40, 40)),
        ]
        print(f"scene {s:05d}: steered decode vs target = {ang:5.1f} deg")

    fig = np.concatenate(rows, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(out, fig)
    except Exception:  # noqa: BLE001
        from PIL import Image

        Image.fromarray(fig).save(out)
    print(f"\nrows per scene: [label] GT a | DECODE(H_a) | DECODE(H_a + cmd edit) | GT b")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
