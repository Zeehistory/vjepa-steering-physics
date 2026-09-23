#!/bin/bash
#SBATCH --job-name=vitg_diag_a
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vitg_diag_a_%j.out
#SBATCH --error=logs/vitg_diag_a_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/07_eval/diag_vitg_angaccel.py \
  --config configs/train/moving_ball_scene_angvel_vitg_decoder.yaml \
  --test_dir $BASE/outputs/latents/moving_ball_scene_angaccel2d_vitg_test/test/vjepa2_giant \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angaccel2d_vitg_decoder/checkpoints/last.pt \
  --num_scenes 30 --device cuda
echo "[vitg_diag_a] exit=$?"
