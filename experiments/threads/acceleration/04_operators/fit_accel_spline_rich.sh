#!/bin/bash
#SBATCH --job-name=aspl_rich
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/aspl_rich_%j.out
#SBATCH --error=logs/aspl_rich_%j.err
#
# Richer conditioning for the spline accel operator (2026-08-26).
#
# The K-sweep already showed the basis is not the binding constraint: at K=8 the spline basis is exact
# (proj_cos == 1) and the fitted operator still only reaches pred_cos ~0.62. The gap is SYNTHESIS -- a
# 13-dim function of (a_a, a_b) cannot predict the edit -- so this widens the map's conditioning while
# holding the basis fixed. All variants stay command-only: appc reads the ANCHOR latent H_a, never H_b.
#
#   FEATURES=appc     sbatch --partition="$(experiments/pipeline/00_common/pick_partition.sh 250000 bigmem day week)" \
#                            experiments/threads/acceleration/04_operators/fit_accel_spline_rich.sh
#
# NOTE the 96G in the sibling _fit_accel_spline.sh header is not what that job actually ran on -- job
# 19848912 was submitted to bigmem with 256G. LatentDataset holds shards in RAM; size for 256G.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAT=outputs/latents/moving_ball_scene_accel2d_mixed
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed

set -euo pipefail
python -u experiments/threads/acceleration/04_operators/fit_accel_spline_rich.py \
    --train_dir "$LAT/train/vjepa2_large" \
    --test_dir  "$LAT/test/vjepa2_large" \
    --appc_dir  "$ANA/subspace" \
    --output_dir "$ANA/spline_rich" \
    --features "${FEATURES:-appc}" --ka "${KA:-16}" \
    --layers 6,12,18,23 --knots "${KNOTS:-1,2,3,8}" --ridge "${RIDGE:-1.0}"
echo "[aspl_rich] done features=${FEATURES:-appc}"
