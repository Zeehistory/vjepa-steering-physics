#!/bin/bash
#SBATCH --job-name=v2d_dirop
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/v2d_dirop_%j.out
#SBATCH --error=logs/v2d_dirop_%j.err
# Fit the DIRECTION-CONDITIONED canonicalized velocity operator (the open lever from canon_ridge).
# CPU-only, big RAM. Streams the v2d train cache once, saves per-(n_bins,bin,layer) ridge artifacts
# alongside the existing subspace artifacts.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/pipeline/04_operators/fit_dir_operator.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_v2d/train/vjepa2_large \
    --layers 6,12,18,23 --bins 4,8,16 --ridge 1.0 \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d/subspace
echo "[v2d_dirop] exit=$?"
