#!/bin/bash
#SBATCH --job-name=accel_subspace
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accel_subspace_%j.out
#SBATCH --error=logs/accel_subspace_%j.err
# Fit the acceleration PCA subspace U (save_k 16 so U8 AND U16 coexist with no refit) + ridge operators
# on the TRAIN accel latents; preview held-out TEST generalization. Latent-only (no GPU). bigmem because
# global PCA buffers ~7GB/layer.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/accel_subspace.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d/test/vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d/subspace \
    --save_k 16
echo "[accel_subspace] exit=$?"
