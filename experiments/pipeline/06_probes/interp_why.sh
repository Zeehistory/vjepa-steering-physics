#!/bin/bash
#SBATCH --job-name=interp_why
#SBATCH --output=logs/interp_why_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
set -euo pipefail
module load miniconda
source activate vjepa-physics-decoder
cd .
export PYTHONPATH=$PWD:$PWD/scripts:${PYTHONPATH:-}
BASE=outputs
python experiments/pipeline/05_steering/interp_why_steer.py \
  --train_dir $BASE/latents/moving_ball_scene_v2d/train/vjepa2_large \
  --test_dir  $BASE/latents/moving_ball_scene_v2d/test/vjepa2_large \
  --artifacts_dir $BASE/analysis/moving_ball_v2d/subspace \
  --output_dir $BASE/analysis/moving_ball_v2d/interp \
  --layers 12,23 --num_scenes 100
