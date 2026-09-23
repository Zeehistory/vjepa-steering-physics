#!/bin/bash
#SBATCH --job-name=plan_d3
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=scratchpad/plan_d3_%j.log
set -euo pipefail
cd .
PY=python
BASE=outputs
ANA=$BASE/analysis/moving_ball_accel2d_mixed
$PY -u experiments/threads/planning-tracks/11_run/plan_d3_warped_svd.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $BASE/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
  --edits_dir $ANA/decopt_edits \
  --output_dir $ANA/plan_d3 \
  --n_edits ${N_EDITS:-500} --n_truedh ${N_TRUEDH:-150}
echo "DONE_D3 exit=$?"
