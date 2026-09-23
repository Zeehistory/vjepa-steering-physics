#!/bin/bash
#SBATCH --job-name=angvel_subspace
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/angvel_subspace_%j.out
#SBATCH --error=logs/angvel_subspace_%j.err
# Fit the angular-velocity PCA subspace U (save_k 16) + ridge operators on TRAIN angvel latents; preview
# held-out TEST generalization. Latent-only (no GPU). --quantity angvel reads obj0_omega as [omega,0].
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/accel_subspace.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large \
    --layers 6,12,18,23 --quantity angvel \
    --output_dir $BASE/outputs/analysis/moving_ball_angvel2d/subspace \
    --save_k 16
echo "[angvel_subspace] exit=$?"
