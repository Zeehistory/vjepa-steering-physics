#!/bin/bash
#SBATCH --job-name=aspl_ap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=10:00:00   # 3h was too tight: slow nodes ran ~0.6 pair/min vs 1.2 on fast ones (2026-08-26)
#SBATCH --output=logs/aspl_allpairs_%j.out
#SBATCH --error=logs/aspl_allpairs_%j.err
#
# ALL-PAIRS spline decode + GAIN-SWEPT ORACLES (2026-08-26).
#
# Two protocol fixes, both cheap, both required before any claim about the operator band:
#
#   --pairs all      The protocol through 2026-07-27 scored ONE pair per scene (rank0 -> rank7) and threw
#                    away the other 6, even though the operator is FIT on all 7. At n=100 the paired CI on
#                    spline_K3 - spline_K8 is [-1.35, +0.79] -- the whole 13.4-14.5deg band is unresolvable
#                    and even the headline K=1 -> K=3 "temporal shape is worth 18%" is p=0.068. n=700 is
#                    the only way any of it becomes a claim rather than an anecdote.
#
#   --oracle_gains   full_delta / prof_full / proj_K* were reported at gain 1 while every FITTED arm got
#                    its own sweep. That asymmetry is the sole evidence for "acceleration NEEDS spatial
#                    placement" (prof_full 31.3 vs full_delta 10.07) -- a claim already incoherent, since
#                    the fitted spline arms live in prof_full's own spatially-uniform function class and
#                    decode at 13.8.
#
# Sharded by scene: SHARD/NSHARD split the 100 test scenes, summaries merge in calibrate_spline_gain.py.
#
#   for i in 0 1 2 3; do
#     SHARD=$i NSHARD=4 sbatch -p <partition> experiments/threads/acceleration/05_steering/steer_accel_spline_allpairs.sh
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
    --checkpoint "$CKPT" \
    --output_dir "$ANA/steer_spline_ap$SHARD" \
    --pairs all --scene_start "$START" --num_scenes "$PER" \
    --knots 1,2,3,4,6,8 \
    --gains 1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0 \
    --oracle_gains 1.5,2.0,2.5,3.0,4.0,6.0 \
    --oracle_arms full_delta,prof_full,proj_K2,proj_K3,proj_K8 \
    --family_scenes 0 --traj_scenes 3 --device cuda
echo "[aspl_ap] shard $SHARD done (scenes $START..$((START+PER-1)))"
