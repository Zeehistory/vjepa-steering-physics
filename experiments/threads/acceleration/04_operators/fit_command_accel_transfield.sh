#!/bin/bash
#SBATCH --job-name=amix_tffit
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/accelmix_tffit_%j.out
#SBATCH --error=logs/accelmix_tffit_%j.err
# Fit the 2nd-order TRANSLATION-FIELD accel operator: per-token map [Dx || Dx(x)posbasis(x_a)] -> slab,
# driven by the exact 1/2*Da*t^2 displacement. Latent-only, no GPU. 256G (shard cache ~160G no eviction).
# 3h wall: per-token full-slab outer-product accumulation is heavy (~1e12 flops, ~60-70min observed).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_transfield.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 --ridge 1.0 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace
echo "[amix_tffit] exit=$?"
