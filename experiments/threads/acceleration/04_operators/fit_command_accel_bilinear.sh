#!/bin/bash
#SBATCH --job-name=amix_blfit
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/accelmix_blfit_%j.out
#SBATCH --error=logs/accelmix_blfit_%j.err
# Fit the global BILINEAR reference-conditioned accel operator (the recon-cos winner: 0.28->0.34-0.46).
# Reuses the accel subspace U (KU=16); light KU-coord fit like the base cmd operator (~10min). No GPU.
# 256G: LatentDataset._shard_cache holds all 32 train shards (~160G, no eviction).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_bilinear.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 --ku 16 --ridge 1.0 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace
echo "[amix_blfit] exit=$?"
