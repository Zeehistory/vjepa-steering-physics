#!/bin/bash
#SBATCH --job-name=amsk_ap
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/amsk_ap_%j.out
#SBATCH --error=logs/amsk_ap_%j.err
#
# ALL-PAIRS decode of CONCEPT-SPACE MANIFOLD steering (arXiv:2605.05115) against the standing
# temporal-spline operator, n=700, leakage-free gain selection (2026-08-29).
#
# What each new family is for:
#
#   man        s(a_b) - s(a_a) with s a thin-plate-spline surface through the acceleration centroids,
#              fitted in a PCA subspace of the pooled temporal profile. THE PAPER'S METHOD, ported.
#              Command-only: it reads the two acceleration labels and never H_b.
#   manlin     the same estimator with the RBF block removed -- the ZERO-CURVATURE control. man - manlin
#              is the only honest measurement of whether concept-space curvature does any work here.
#   mandense   twice the knots, everything else fixed -- curvature-capacity ablation.
#   manK3      the manifold edit additionally projected onto the 3-knot TEMPORAL spline basis, i.e. the
#              paper's geometry composed with this repo's.
#   mansp8     mean of the manifold edit and the standing spline_K8 edit -- only useful if the two
#              operators' errors are partly independent, which is itself worth knowing.
#   oprojU{k}  the TRUE delta H projected onto its own top-k global subspace, gain-swept. This is the
#              only arm class that can beat full_delta, and it is worth being precise about why:
#              full_delta at gain 1 IS the decode of the real target latent, so its 11.24deg is the
#              decoder+tracker noise floor, not a steering ceiling. Getting under it means the
#              synthesized latent reads CLEANER than reality -- which is exactly the claim on-manifold
#              projection makes.
#   oman       the true delta's SPATIAL placement with its temporal profile replaced by the on-manifold
#              one. prof_full (true profile, no placement) is 64deg ungained while full_delta is 11.24,
#              so placement carries most of the oracle's advantage; this isolates the profile half.
#
# Sharded by scene; summaries merge in calibrate_spline_gain.py, which splits val/test by SCENE.
#   for i in 0 1 2 3; do
#     SHARD=$i NSHARD=4 sbatch -p <partition> experiments/threads/acceleration/05_steering/steer_accel_manifold.sh
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
VARIANTS=${VARIANTS:-man,manlin,mandense}
MANDIR=${MANDIR:-$ANA/manifold}

set -euo pipefail
python -u experiments/threads/acceleration/05_steering/steer_accel_spline.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir "$LAT/test/vjepa2_large" \
    --spline_dir "$ANA/spline" \
    --manifold_dir "$MANDIR" --manifold_variants man \
    --manifold_knots "" --manifold_blend "" \
    --manifold_mask_sigmas 1.5,3.0 --mask_control \
    --checkpoint "$CKPT" \
    --output_dir "$ANA/steer_mask_ap$SHARD" \
    --pairs all --scene_start "$START" --num_scenes "$PER" \
    --knots 8 \
    --gains 1.5,2.0,2.5,3.0,4.0,6.0 \
    --oracle_gains 1.0 \
    --oracle_arms full_delta,prof_full \
    --family_scenes 0 --traj_scenes 2 --device cuda
echo "[amsk_ap] shard $SHARD done (scenes $START..$((START+PER-1)))"
