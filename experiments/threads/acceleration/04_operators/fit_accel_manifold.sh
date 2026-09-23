#!/bin/bash
#SBATCH --job-name=aman_fit
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/aman_fit_%j.out
#SBATCH --error=logs/aman_fit_%j.err
# Fit the CONCEPT-SPACE activation manifold for 2-D acceleration (arXiv:2605.05115 port).
# CPU only: the fit lives in the pooled (T=8, D=1024) profile space and a 64-dim PCA of it, so the
# linear algebra is trivial. Cost is IO (500 train scenes + 100 test scenes x 8 clips x 4 layers).
# Latents are on SCRATCH (project is at 94% quota); artifacts are a few MB and go to project.
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 96000 day devel week mpi priority)
#   sbatch -p <partition> experiments/threads/acceleration/04_operators/fit_accel_manifold.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
LAT=outputs/latents/moving_ball_scene_accel2d_mixed
OUT=outputs/analysis/moving_ball_accel2d_mixed/manifold

set -euo pipefail
python -u experiments/threads/acceleration/04_operators/fit_accel_manifold.py \
    --train_dir "$LAT/train/vjepa2_large" \
    --test_dir  "$LAT/test/vjepa2_large" \
    --layers 6,12,18,23 \
    --k_pca 32,64 --n_centers 32,64,128 --ridge 1e-2,1e-3,1e-4,1e-5 \
    --val_scene_frac 0.2 \
    --output_dir "$OUT"
echo "[aman_fit] done -> $OUT"
