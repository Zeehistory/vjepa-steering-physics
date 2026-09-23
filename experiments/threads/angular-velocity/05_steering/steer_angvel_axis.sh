#!/bin/bash
#SBATCH --job-name=angvel_axis
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=logs/angvel_axis_%j.out
#SBATCH --error=logs/angvel_axis_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_axis.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/last.pt \
  --num_test ${NUM_TEST:-25} --num_cal ${NUM_CAL:-6} --gains ${GAINS:-0.5,1,1.5,2,3,4} \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/steer_command_axis.json
echo "[angvel_axis] exit=$?"
