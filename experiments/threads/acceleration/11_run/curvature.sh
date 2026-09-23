#!/bin/bash
#SBATCH --job-name=curv
#SBATCH --output=scratchpad/curv_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
cd .
PY=python
L=outputs/latents/moving_ball_scene_accel2d_mixed
OUT=outputs/analysis/moving_ball_accel2d_mixed/curvature
$PY experiments/threads/acceleration/11_run/curvature_predictability.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $L/train/vjepa2_large --test_dir $L/test/vjepa2_large \
  --output_dir $OUT
echo "DONE_CURV"
