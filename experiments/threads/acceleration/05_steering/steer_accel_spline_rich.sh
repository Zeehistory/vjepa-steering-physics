#!/bin/bash
#SBATCH --job-name=aspl_rich_dec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=10:00:00   # 3h was too tight: slow nodes ran ~0.6 pair/min vs 1.2 on fast ones (2026-08-26)
#SBATCH --output=logs/aspl_rich_dec_%j.out
#SBATCH --error=logs/aspl_rich_dec_%j.err
#
# Decode the RICHER-CONDITIONING spline operators, all-pairs protocol (2026-08-26).
#
# The K-sweep localised the bottleneck: at K=8 the spline basis is exact and the fitted operator still
# only reaches pred_cos ~0.62, so what is missing is the MAP, not the basis. These arms hold the basis
# fixed and condition on the anchor latent H_a as well as the command -- still command-only, still no
# H_b. Also carries the layer-subset arms, since one scalar gain has always scaled all four layers even
# though zeroing L23 costs 4x what zeroing L6 costs.
#
# The `cmd` rich variant is the CONTROL: same script, same 7-pair enrichment, same ridge, 13 features --
# it must land on the existing spline arm, or the difference is the harness rather than the features.
#
#   for i in 0 1 2 3; do
#       experiments/threads/acceleration/05_steering/steer_accel_spline_rich.sh
#   done
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAT=outputs/latents/moving_ball_scene_accel2d_mixed
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt

SHARD=${SHARD:-0}; NSHARD=${NSHARD:-4}; TOTAL=${TOTAL:-100}
PER=$(( TOTAL / NSHARD ))
START=$(( SHARD * PER ))

set -euo pipefail
python -u experiments/threads/acceleration/05_steering/steer_accel_spline.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir "$LAT/test/vjepa2_large" \
    --spline_dir "$ANA/spline" \
    --rich_dir "$ANA/spline_rich" --rich_features "${RICH:-cmd,appc,bilinear}" \
    --appc_dir "$ANA/subspace" \
    --checkpoint "$CKPT" \
    --output_dir "$ANA/steer_spline_rich$SHARD" \
    --pairs all --scene_start "$START" --num_scenes "$PER" \
    --knots 2,3 \
    --gains 1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0 \
    --layer_subsets "23;18,23;12,18,23" --subset_knot 3 \
    --oracle_gains "" --family_scenes 0 --traj_scenes 0 --device cuda
echo "[aspl_rich_dec] shard $SHARD done (scenes $START..$((START+PER-1)))"
