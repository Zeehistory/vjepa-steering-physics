#!/bin/bash
#SBATCH --job-name=angvel_read
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=01:00:00
#SBATCH --output=logs/angvel_read_%j.out
#SBATCH --error=logs/angvel_read_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/06_probes/probe_angvel_read.py \
  --config configs/train/moving_ball_scene_angvel_decoder_orient.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d/train/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large \
  --max_train_clips 1200
echo "[angvel_read] exit=$?"
