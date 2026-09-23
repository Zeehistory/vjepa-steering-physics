#!/bin/bash
#SBATCH --job-name=ares_fit
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ares_fit_%j.out
#SBATCH --error=logs/ares_fit_%j.err
# Streamed rank-r model of the acceleration edit's SPATIAL RESIDUAL (2026-08-29). CPU only, two passes
# over the 500-scene train split; peak RAM is the r x 2.1M sketch plus its orthonormalization (~13 GB at
# r=192, x4 layers). Read capture_cos (the ceiling) before anything else.
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 160000 day week mpi priority)
#   sbatch -p <partition> experiments/threads/acceleration/04_operators/fit_accel_residual.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
LAT=outputs/latents/moving_ball_scene_accel2d_mixed
OUT=outputs/analysis/moving_ball_accel2d_mixed/residual
set -euo pipefail
python -u experiments/threads/acceleration/04_operators/fit_accel_residual.py \
    --train_dir "$LAT/train/vjepa2_large" --test_dir "$LAT/test/vjepa2_large" \
    --layers 6,12,18,23 --rank ${RANK:-192} --ridge 1.0 --output_dir "$OUT"
echo "[ares_fit] done -> $OUT"
