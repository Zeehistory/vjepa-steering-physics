#!/bin/bash
#SBATCH --job-name=ahyb_ap
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ahyb_ap_%j.out
#SBATCH --error=logs/ahyb_ap_%j.err
#
# ATTRIBUTION of the one arm that went under the oracle (2026-08-29).
#
# At n=700, `oman` = (true delta's spatial residual) + (manifold-predicted temporal profile) decoded at
# 10.46deg against full_delta's 11.23 -- the first arm in this thread ever to go below the decode of the
# real target latent. Before that is called a manifold result it has to survive attribution, so this run
# sweeps the SAME hybrid with the profile swapped out:
#
#   ohybtrue    the TRUE profile, gain-swept from 0 -- the SHRINKAGE control. g=1 IS full_delta and
#               g=0 is residual-only. If some g < 1 here matches oman, the mechanism is plain
#               attenuation of the true profile and the manifold contributed nothing.
#   ohybspline8 the standing temporal-spline operator's profile -- if this matches oman, the win is
#               "any smooth regressed profile beats the true one", not the concept-space geometry.
#   ohybman / ohybmanlin  the manifold profile and its ZERO-CURVATURE control, in the hybrid.
#   ohybmansp8  the manifold/spline blend, which was the best command-only arm at n=700.
#
# spline_K8 (g 2.5) and full_delta (g 1) are decoded here too so the paired bootstrap stays on
# identical rows when this merges with the wave-1/2 shards.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAT=outputs/latents/moving_ball_scene_accel2d_mixed
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed

SHARD=${SHARD:-0}; NSHARD=${NSHARD:-4}; TOTAL=${TOTAL:-100}
PER=$(( TOTAL / NSHARD )); START=$(( SHARD * PER ))

set -euo pipefail
python -u experiments/threads/acceleration/05_steering/steer_accel_spline.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir "$LAT/test/vjepa2_large" \
    --spline_dir "$ANA/spline" \
    --manifold_dir "$ANA/manifold" --manifold_variants man,manlin \
    --manifold_knots "" --manifold_blend "" \
    --oracle_hybrid_srcs true,spline8,man,manlin,mansp8 \
    --hybrid_gains 0,0.25,0.5,0.75,1.0,1.25,1.5 \
    --checkpoint "$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt" \
    --output_dir "$ANA/steer_hyb_ap$SHARD" \
    --pairs all --scene_start "$START" --num_scenes "$PER" \
    --knots 8 --gains 2.5,3.0 \
    --oracle_gains 1.0 --oracle_arms full_delta \
    --family_scenes 0 --traj_scenes 0 --device cuda
echo "[ahyb_ap] shard $SHARD done (scenes $START..$((START+PER-1)))"
