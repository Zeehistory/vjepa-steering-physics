#!/bin/bash
#SBATCH --job-name=v2dmix_subspace
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2dmix_subspace_%j.out
#SBATCH --error=logs/v2dmix_subspace_%j.err
# PCA of Delta H (within-scene + global), principal angles, ridge F_U (raw + canon), TEST preview, for
# the appearance-mixed cache. save_k=16 so BOTH U8 and a U16 fallback are available without a refit.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/velocity/04_operators/velocity_subspace.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_v2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_v2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d_mixed/subspace \
    --ridge 1.0 --save_k 16 --max_global_pairs 800
echo "[v2dmix_subspace] exit=$?"
