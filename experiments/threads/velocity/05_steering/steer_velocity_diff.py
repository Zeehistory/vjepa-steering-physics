

#!/usr/bin/env python
"""Difference-vector velocity steering (supervisor's 2026-06-26 proposal).

Instead of pushing the latents along an *abstract probe direction* (the approach that moved the readout
but not the pixels), this steers with a **data-derived difference vector** between two
real encoded clips of the *same scene* that differ only in speed:

    H_a = E(slow clip),  H_b = E(fast clip)        # identical first frame + direction, only speed differs
    Delta = H_b - H_a                              # isolates the velocity factor, per token, per layer
    H_alpha = H_a + alpha * Delta                  # decode this; alpha=0 -> v_a, alpha=1 -> v_b

Because alpha in [0, 1] interpolates between two latents the decoder was trained to reconstruct, the
edit stays ON-MANIFOLD: the endpoints must render correctly and intermediate alphas stay near real
data. Sweeping alpha (incl. extrapolation alpha<0 / alpha>1) should make the decoded ball visibly speed
up / slow down. We re-track the ball in the decoded pixels and check the measured speed rises
monotonically with alpha and tracks the ground-truth interpolated speed.

Two variants:
  * **per-pair** (primary): each scene steered by its OWN Delta (the on-manifold, same-scene edit).
  * **mean-Delta** (LLM-style single vector): one Delta averaged across many scene pairs, applied to
    each held-out scene's H_a — tests whether a single global "speed-up" vector generalizes.

Example
-------
    python experiments/threads/velocity/05_steering/steer_velocity_diff.py \
        --config configs/train/moving_ball_scene_decoder.yaml \
        --latent_dir outputs/latents/moving_ball_scene/test/vjepa2_large \
        --checkpoint outputs/runs/moving_ball_scene_decoder/checkpoints/last.pt \
        --alphas -0.5,0,0.5,1.0,1.5 --num_scenes 12 \
        --output_dir outputs/analysis/moving_ball_scene/diff_steer --device cuda
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
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from src.analysis import visualization as viz
from src.analysis.ball_tracking import measured_velocity
from src.decoders import build_decoder
from src.encoders.feature_extractor import LatentDataset, latent_collate
from src.training.checkpoints import load_checkpoint
from src.utils.config import load_config

_SCENE_RE = re.compile(r"^scene(\d+)_v(\d+)$")


def _scene_rank(sample_id: str) -> tuple[int, int] | None:
    m = _SCENE_RE.match(sample_id)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _clip_speed(sample: dict) -> float:
    """Ground-truth speed of a clip = mean of its obj0_speed state column."""
    keys = list(sample["state_keys"])
    if "obj0_speed" in keys:
        col = keys.index("obj0_speed")
        return float(sample["state"][:, col].mean())
    # fallback: empirical speed from the stored frames
    return measured_velocity(sample["frames"])["speed"]


def _to_dev(sample: dict, layers: list[int], device: str) -> dict[int, torch.Tensor]:
    batch = latent_collate([sample])
    return {int(k): v.to(device) for k, v in batch["layers"].items() if int(k) in layers}


@torch.no_grad()
def _decode_speed(decoder, latents, grid, want_frames=False):
    out = decoder(latents, grid)
    if out.frames is None:
        return {"speed": float("nan"), "vel_x": float("nan"), "vel_y": float("nan"), "n_valid": 0}, None
    fr = out.frames[0].cpu()
    meas = measured_velocity(fr)
    return meas, (fr if want_frames else None)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--latent_dir", required=True, help="scene-paired test cache to steer + decode")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alphas", default="-0.5,0,0.5,1.0,1.5",
                   help="absolute interpolation coefficients; 0->v_a, 1->v_b, <0/>1 extrapolate")
    p.add_argument("--num_scenes", type=int, default=12, help="how many test scenes to steer")
    p.add_argument("--mean_delta", action="store_true",
                   help="also steer every scene by a SINGLE Delta averaged across scene pairs")
    p.add_argument("--device", default="cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    device = args.device

    ds = LatentDataset(args.latent_dir, layers=cfg.encoder.layers)
    primed = [int(x) for x in ds.available_layers()]
    loaded = sorted(int(k) for k in ds[0]["layers"].keys())

    # --- group cached samples into scenes -> {rank: dataset_index} ----------------------------------
    scenes: dict[int, dict[int, int]] = defaultdict(dict)
    for i in range(len(ds)):
        sr = _scene_rank(ds._ids[i])
        if sr is not None:
            scene, rank = sr
            scenes[scene][rank] = i
    full = {s: r for s, r in scenes.items() if len(r) >= 2}
    if not full:
        raise RuntimeError(f"No multi-clip scenes found in {args.latent_dir}; is this a scene_velocity cache?")
    scene_ids = sorted(full)[: args.num_scenes]
    print(f"[diff_steer] {len(scenes)} scenes in cache, steering {len(scene_ids)}; layers={loaded}")

    # --- build decoder ------------------------------------------------------------------------------
    rec0 = ds.records[0]
    enc_dim, state_dim = int(rec0["hidden_dim"]), int(rec0["state_dim"])
    cfg.decoder.state_dim = state_dim
    if cfg.decoder.out_num_frames <= 0:
        cfg.decoder.out_num_frames = cfg.data.num_frames
    decoder = build_decoder(cfg.decoder, enc_dim, state_dim).to(device).eval()
    if hasattr(decoder, "prime_layers"):
        decoder.prime_layers(primed)
    load_checkpoint(args.checkpoint, decoder, map_location=device)

    # --- per-pair steering --------------------------------------------------------------------------
    per_scene: dict[str, dict] = {}
    decoded_by_alpha: dict[float, list[float]] = {a: [] for a in alphas}
    gt_by_alpha: dict[float, list[float]] = {a: [] for a in alphas}
    deltas: dict[int, list[torch.Tensor]] = {L: [] for L in loaded}
    Ha_store: list[tuple[int, dict, tuple, float, float]] = []

    for n, scene in enumerate(scene_ids):
        ranks = sorted(full[scene])
        a_idx, b_idx = full[scene][ranks[0]], full[scene][ranks[-1]]
        sa, sb = ds[a_idx], ds[b_idx]
        grid = tuple(int(x) for x in sa["grid"])
        Ha = _to_dev(sa, loaded, device)
        Hb = _to_dev(sb, loaded, device)
        speed_a, speed_b = _clip_speed(sa), _clip_speed(sb)
        delta = {L: Hb[L] - Ha[L] for L in loaded}
        for L in loaded:
            deltas[L].append(delta[L])
        Ha_store.append((scene, Ha, grid, speed_a, speed_b))

        curve, frames_by_alpha = [], {}
        for a in alphas:
            Hal = {L: Ha[L] + a * delta[L] for L in loaded}
            want = a in (min(alphas), 0.0, 1.0, max(alphas))
            meas, fr = _decode_speed(decoder, Hal, grid, want_frames=want)
            gt_speed = speed_a + a * (speed_b - speed_a)
            curve.append({"alpha": a, "decoded_speed": meas["speed"], "gt_speed": gt_speed,
                          "n_valid": meas["n_valid"]})
            decoded_by_alpha[a].append(meas["speed"])
            gt_by_alpha[a].append(gt_speed)
            if fr is not None:
                frames_by_alpha[a] = fr
        per_scene[f"scene{scene:05d}"] = {
            "speed_a": speed_a, "speed_b": speed_b, "ranks": ranks, "curve": curve}

        if frames_by_alpha:
            viz.steering_filmstrip(frames_by_alpha, out / f"scene{scene:05d}_filmstrip.png")
            base = frames_by_alpha.get(0.0, next(iter(frames_by_alpha.values())))
            if max(alphas) in frames_by_alpha:
                viz.panel_video(base, frames_by_alpha[max(alphas)],
                                out / f"scene{scene:05d}_a0_to_a{max(alphas):g}.mp4", fps=cfg.data.fps)
        dec_str = "  ".join(f"a{c['alpha']:g}={c['decoded_speed']:.4f}" for c in curve)
        print(f"  scene{scene:05d}: v_a={speed_a:.4f} v_b={speed_b:.4f} | decoded {dec_str}")

    # --- aggregate per-pair -------------------------------------------------------------------------
    def _mean_curve(d):
        return [{"alpha": a, "value": float(np.nanmean(v))} for a, v in sorted(d.items())]
    dec_curve = _mean_curve(decoded_by_alpha)
    gt_curve = _mean_curve(gt_by_alpha)
    fin = [(c["alpha"], c["value"]) for c in dec_curve if np.isfinite(c["value"])]
    rho = float(spearmanr([a for a, _ in fin], [v for _, v in fin]).statistic) if len(fin) >= 2 else float("nan")
    # decoded-vs-GT linearity across ALL (scene, alpha) decoded points
    alld, allg = [], []
    for a in alphas:
        for ds_, gs_ in zip(decoded_by_alpha[a], gt_by_alpha[a]):
            if np.isfinite(ds_):
                alld.append(ds_); allg.append(gs_)
    pear = float(np.corrcoef(allg, alld)[0, 1]) if len(alld) >= 2 else float("nan")

    sweeps = {"decoded (pixel-tracked)": [{"alpha": c["alpha"], "readout": c["value"]} for c in dec_curve],
              "ground-truth interp": [{"alpha": c["alpha"], "readout": c["value"]} for c in gt_curve]}

    # --- optional mean-Delta steering ---------------------------------------------------------------
    mean_delta_summary = None
    if args.mean_delta and Ha_store:
        mean_delta = {L: torch.stack(deltas[L], 0).mean(0) for L in loaded}
        md_decoded: dict[float, list[float]] = {a: [] for a in alphas}
        for scene, Ha, grid, speed_a, speed_b in Ha_store:
            for a in alphas:
                Hal = {L: Ha[L] + a * mean_delta[L] for L in loaded}
                meas, _ = _decode_speed(decoder, Hal, grid)
                md_decoded[a].append(meas["speed"])
        md_curve = _mean_curve(md_decoded)
        mfin = [(c["alpha"], c["value"]) for c in md_curve if np.isfinite(c["value"])]
        md_rho = float(spearmanr([a for a, _ in mfin], [v for _, v in mfin]).statistic) if len(mfin) >= 2 else float("nan")
        sweeps["mean-Delta decoded"] = [{"alpha": c["alpha"], "readout": c["value"]} for c in md_curve]
        mean_delta_summary = {"monotonicity_spearman": round(md_rho, 4), "curve": md_curve}

    viz.steering_sweep_plot(
        sweeps, out / "diff_steering_controllability.png",
        title="Difference-vector velocity steering (decoded ball speed vs α)",
        ylabel="ball speed (norm units / frame)")

    summary = {
        "method": "difference_vector (H_a + alpha*(H_b - H_a))",
        "latent_dir": args.latent_dir, "checkpoint": args.checkpoint,
        "layers": loaded, "alphas": alphas, "n_scenes": len(scene_ids),
        "per_pair": {
            "decoded_monotonicity_spearman": round(rho, 4),
            "decoded_vs_gt_pearson": round(pear, 4),
            "decoded_curve": dec_curve, "gt_curve": gt_curve,
        },
        "mean_delta": mean_delta_summary,
        "per_scene": per_scene,
    }
    (out / "diff_steering_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[diff_steer] per-pair decoded monotonicity ρ={rho:.3f}  decoded-vs-GT r={pear:.3f}  "
          f"over {len(scene_ids)} scenes -> {out}")
    if mean_delta_summary:
        print(f"[diff_steer] mean-Delta monotonicity ρ={mean_delta_summary['monotonicity_spearman']:.3f}")


if __name__ == "__main__":
    main()
