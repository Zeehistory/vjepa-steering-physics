#!/bin/bash
#SBATCH --job-name=v2d_subspace
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2d_subspace_%j.out
#SBATCH --error=logs/v2d_subspace_%j.err
# Phase 2 + operator fitting (latent-only, CPU): PCA of Delta H (within-scene + global), principal
# angles, ridge F_U (raw + canonicalized), saved as artifacts + a TEST latent-space generalization
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/velocity/04_operators/velocity_subspace.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_v2d/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_v2d/test/vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d/subspace \
    --ridge 1.0 --save_k 8 --max_global_pairs 800
echo "[v2d_subspace] exit=$?"
