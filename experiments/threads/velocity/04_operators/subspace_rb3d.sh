#!/bin/bash
#SBATCH --job-name=rb3d_subspace
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rb3d_subspace_%j.out
#SBATCH --error=logs/rb3d_subspace_%j.err
# PCA of Delta H (within-scene + global), principal angles, ridge F_U (raw + canon), TEST preview, for
# the 3D rolling-ball cache. save_k=16 so BOTH U8 and a U16 fallback are available without a refit.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAT_BASE=${LAT_BASE:-.}  # latents on scratch (see slurm_extract_rb3d.sh)
python -u experiments/threads/velocity/04_operators/velocity_subspace.py \
    --train_dir $LAT_BASE/outputs/latents/rolling_ball3d/train/vjepa2_large \
    --test_dir  $LAT_BASE/outputs/latents/rolling_ball3d/test/vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/rolling_ball3d/subspace \
    --ridge 1.0 --save_k 16 --max_global_pairs 800
echo "[rb3d_subspace] exit=$?"
