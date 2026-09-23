#!/bin/bash
#SBATCH --job-name=interp_xobj
#SBATCH --output=logs/interp_xobj_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
set -uo pipefail
module load miniconda
source activate vjepa-physics-decoder
cd .
export PYTHONPATH=$PWD:$PWD/scripts:${PYTHONPATH:-}
B=outputs/latents
python experiments/pipeline/06_probes/interp_cross_object.py \
  --disk_train $B/moving_ball_scene_v2d/train/vjepa2_large \
  --disk_test  $B/moving_ball_scene_v2d/test/vjepa2_large \
  --sq_train   $B/moving_ball_scene_v2d_square/train/vjepa2_large \
  --sq_test    $B/moving_ball_scene_v2d_square/test/vjepa2_large \
  --artifacts_dir outputs/analysis/moving_ball_v2d/subspace \
  --output_dir outputs/analysis/moving_ball_v2d/interp \
  --layers 12,23 --num_scenes 100
