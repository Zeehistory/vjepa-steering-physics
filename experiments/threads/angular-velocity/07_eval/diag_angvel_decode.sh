#!/bin/bash
#SBATCH --job-name=angvel_diag
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/angvel_diag_%j.out
#SBATCH --error=logs/angvel_diag_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/07_eval/diag_angvel_decode.py \
  --config configs/train/moving_ball_scene_angvel_decoder.yaml \
  --test_dir $BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angvel2d_decoder_fp/checkpoints/last.pt \
  --num_scenes 12 --device cuda
echo "[angvel_diag] exit=$?"
