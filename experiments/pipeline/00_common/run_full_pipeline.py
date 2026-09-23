

#!/usr/bin/env python
"""End-to-end pipeline: extract latents -> train decoder -> probe -> evaluate -> visualize.

Runs entirely offline on CPU/MPS with the mock encoder when given the smoke config:

    python experiments/pipeline/00_common/run_full_pipeline.py --config configs/train/smoke_synthetic.yaml \
        --output_dir outputs/smoke

Produces under ``--output_dir``:
    latents/   runs/decoder/   probes/   eval/metrics.json   viz/*.png  viz/*.mp4
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

from eval_decoder import evaluate_decoder
from visualize_reconstructions import visualize

from src.analysis.visualization import layerwise_probe_plot
from src.data import build_dataset
from src.encoders import build_encoder
from src.encoders.feature_extractor import extract_latents
from src.training import probe_layers, train_decoder
from src.utils.config import load_config, save_config, to_container
from src.utils.gpu import resolve_device
from src.utils.reproducibility import run_metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--skip_probe", action="store_true")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, root / "pipeline_config.yaml")

    device = resolve_device(cfg.encoder.device)
    print(f"[pipeline] device={device}")

    # 1. extract latents (mock encoder, offline)
    latent_dir = root / "latents"
    encoder = build_encoder(cfg.encoder)
    dataset = build_dataset(cfg.data, encoder_image_size=cfg.encoder.image_size,
                            encoder_frames=cfg.encoder.num_frames)
    extract_latents(
        encoder, dataset, latent_dir, layers=cfg.encoder.layers,
        batch_size=cfg.train.batch_size, device=device,
        store_frames_size=cfg.decoder.out_image_size,
        extract_meta={"config": to_container(cfg.encoder), "provenance": run_metadata()},
    )
    print(f"[pipeline] latents -> {latent_dir}")

    # 2. train decoder
    cfg.latent_dir = str(latent_dir)
    cfg.output_dir = str(root / "runs" / "decoder")
    summary = train_decoder(cfg)
    checkpoint = summary["checkpoint"]
    print(f"[pipeline] trained decoder -> {checkpoint}")

    # 3. layerwise probes (Experiment 1)
    if not args.skip_probe:
        probes_dir = root / "probes"
        records = probe_layers(latent_dir, layers="all", seed=cfg.train.seed,
                               output_csv=probes_dir / "layerwise_decodability.csv")
        if records:
            layerwise_probe_plot(records, probes_dir / "layerwise_probe.png")
        print(f"[pipeline] probes -> {probes_dir}")

    # 4. evaluate (reconstruction + physics + baselines/controls)
    res = evaluate_decoder(cfg, str(latent_dir), checkpoint, root / "eval", device=str(device))
    rec = res["reconstruction"]["model"]
    print(f"[pipeline] eval -> {root/'eval'/'metrics.json'} | PSNR={rec['psnr']:.2f} SSIM={rec['ssim']:.3f}")

    # 5. visualize
    visualize(cfg, str(latent_dir), checkpoint, root / "viz", num_samples=3, device=str(device))
    print(f"[pipeline] DONE. Artifacts under {root}")


if __name__ == "__main__":
    main()
