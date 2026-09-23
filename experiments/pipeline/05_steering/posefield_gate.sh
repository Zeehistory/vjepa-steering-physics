#!/bin/bash
#SBATCH --job-name=pf_gate
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=03:00:00
#SBATCH --output=logs/posefield_gate_%j.out
#SBATCH --error=logs/posefield_gate_%j.err
# POSE-FIELD acceleration operator -- LATENT GATE (CPU, no decoder, no new disk).
# Fits the whole --bases ladder in ONE streaming pass over the accel2d_mixed train latents and scores each
# by cos(synthesized dH, TRUE H_b-H_a) on held-out test scenes vs a wrong-accel/right-v0 control.
# Decode only what this clears. Submit:
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 300000 bigmem mpi day week); sbatch -p <partition> experiments/pipeline/05_steering/posefield_gate.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/05_steering/steer_accel_posefield.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
  --gate_only \
  --n_train_scenes ${NTRAIN:-150} --n_test_scenes ${NTEST:-20} \
  --bases "${BASES:-lin_a,lin_dv,rbf_d,rbf_dv,four_dv,poly_dv}" \
  --rbf_grid ${RBF_GRID:-8} --four_k ${FOUR_K:-4} --ridge ${RIDGE:-10.0} \
  --chunk ${CHUNK:-64} \
  --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/posefield/gate_${TAG:-pilot}.json
echo "[pf_gate] exit=$?"
