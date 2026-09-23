#!/bin/bash
#SBATCH --job-name=tprobe
#SBATCH --output=scratchpad/tprobe_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
cd .
PY=python
L=outputs/latents/moving_ball_scene_accel2d_mixed
OUT=outputs/analysis/moving_ball_accel2d_mixed/temporal_probe
$PY experiments/pipeline/06_probes/latent_temporal_probe.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $L/train/vjepa2_large --test_dir $L/test/vjepa2_large \
  --output_dir $OUT
echo "DONE_TPROBE"
