#!/bin/bash
#SBATCH --job-name=plan_d1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=scratchpad/plan_d1_%j.log
set -euo pipefail
cd .
PY=python
BASE=outputs
ANA=$BASE/analysis/moving_ball_accel2d_mixed
$PY -u experiments/threads/planning-tracks/11_run/plan_d1_identifiability.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --test_dir $BASE/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
  --artifacts_dir $ANA/subspace \
  --checkpoint $BASE/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt \
  --output_dir $ANA/plan_d1 \
  --num_scenes ${NUM_SCENES:-15} --n_random ${N_RANDOM:-3} \
  --steps ${STEPS:-300} --lr ${LR:-0.08} --anchor ${ANCHOR:-1.0} --device cuda
echo "DONE_D1 exit=$?"
