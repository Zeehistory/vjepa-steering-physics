#!/bin/bash
#SBATCH --job-name=v2d_disent
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/v2d_disent_%j.out
#SBATCH --error=logs/v2d_disent_%j.err
# Velocity-vs-nuisance disentanglement: principal angles between the velocity subspace (v2d global PCA)
# and the size/colour/background subspaces. CPU-only, big RAM.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
L=$BASE/outputs/latents
python -u experiments/pipeline/06_probes/disentangle_nuisance.py \
    --velocity_basis_dir $BASE/outputs/analysis/moving_ball_v2d/subspace \
    --size_dir  $L/moving_ball_scene_size/train/vjepa2_large \
    --color_dir $L/moving_ball_scene_color/train/vjepa2_large \
    --bg_dir    $L/moving_ball_scene_background/train/vjepa2_large \
    --layers 6,12,18,23 --k 8 --max_pairs 500 \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d/disentangle
echo "[v2d_disent] exit=$?"
