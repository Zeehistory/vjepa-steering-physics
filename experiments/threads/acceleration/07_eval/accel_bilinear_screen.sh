#!/bin/bash
#SBATCH --job-name=amix_bilin
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/accelmix_bilin_%j.out
#SBATCH --error=logs/accelmix_bilin_%j.err
# Latent-only screen: does a BILINEAR command(x)reference operator reconstruct the true accel delta better
# than the linear command-only floor (recon cos ~0.28 = the ~14.5deg steer ceiling)? No GPU. 256G because
# LatentDataset._shard_cache holds all 32 train shards (4/24 layers) ~160G with no eviction (same as the fits).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/07_eval/accel_bilinear_screen.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 --ku 16 --ridge 1.0 \
    --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace/bilinear_screen.json
echo "[amix_bilin] exit=$?"
