#!/bin/bash
#SBATCH --job-name=fitcurv
#SBATCH --output=scratchpad/fitcurv_%j.log
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
cd .
PY=python
BASE=outputs
$PY experiments/threads/acceleration/04_operators/fit_curvature_operator.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $BASE/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
  --artifacts_dir $BASE/analysis/moving_ball_accel2d_mixed/subspace \
  --save_k ${SAVE_K:-32}
echo "DONE_FITCURV"
