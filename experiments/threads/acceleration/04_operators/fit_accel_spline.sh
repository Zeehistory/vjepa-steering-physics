#!/bin/bash
#SBATCH --job-name=aspl_fit
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/aspl_fit_%j.out
#SBATCH --error=logs/aspl_fit_%j.err
# Fit the command -> spline-control-point acceleration operators (K = 1,2,3,4,6,8) on TRAIN.
# CPU only: the fit lives in the spatially-pooled (T=8, D=1024) profile space, so the normal equations
# are 13 x K*D at most -- trivial. Cost is IO (500 scenes x 8 clips x 4 layers).
# ALWAYS submit via the queue-aware picker:
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 96000 day devel week mpi priority)
#   sbatch -p <partition> experiments/threads/acceleration/04_operators/fit_accel_spline.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
DATA=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed
OUT=$BASE/outputs/analysis/moving_ball_accel2d_mixed/spline

python -u experiments/threads/acceleration/04_operators/fit_accel_spline_operator.py \
    --train_dir $DATA/train/vjepa2_large \
    --test_dir  $DATA/test/vjepa2_large \
    --layers 6,12,18,23 --knots 1,2,3,4,6,8 --ridge 1.0 \
    --output_dir $OUT
STATUS=$?
echo "[aspl_fit] exit=$STATUS out=$OUT"
exit $STATUS
