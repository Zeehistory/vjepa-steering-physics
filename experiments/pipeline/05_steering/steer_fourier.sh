#!/bin/bash
#SBATCH --job-name=steer_fourier
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/steer_fourier_%j.out
#SBATCH --error=logs/steer_fourier_%j.err
# Fourier-in-orientation command-only steer for a rotational quantity.
#
#   QUANTITY=angaccel  FIT=angvel   -> ZERO-SHOT across kinematic order (the headline)
#   QUANTITY=angaccel  FIT=angaccel -> matched/in-domain control (upper bound)
#   QUANTITY=angvel    FIT=angvel   -> regression check (must reproduce held-out rho ~0.94)
#
# PTG=1 adds the per-TEMPORAL-TOKEN least-squares rescaling of the edit (fit on TRAIN, mean-1, so the
# global gain sweep is unchanged). Motivated by the kinematic-order asymmetry: the angular ACCELERATION
# edit's orientation displacement spans ~841x across the 8 tokens against ~29x for angular velocity, so
# one scalar gain is the wrong SHAPE. With FIT=angvel the scale is fit on constant-omega data too, so
# the zero-shot claim is preserved.
#
# GATE_ONLY=1 runs the CPU latent gate with no decoder (submit to bigmem); otherwise a GPU decode.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CKPT=${CKPT:-$BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/step_7000.pt}
QUANTITY=${QUANTITY:-angaccel}
FIT=${FIT:-angvel}
case "$FIT" in
  angvel)   TRAIN_DIR=${TRAIN_DIR:-$BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large} ;;
  angaccel) TRAIN_DIR=${TRAIN_DIR:-$BASE/outputs/latents/moving_ball_scene_angaccel2d_train/test/vjepa2_large} ;;
esac
case "$QUANTITY" in
  angaccel) TEST_DIR=${TEST_DIR:-$BASE/outputs/latents/moving_ball_scene_angaccel2d_test/test/vjepa2_large} ;;
  angvel)   TEST_DIR=${TEST_DIR:-$BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large} ;;
esac
TAG=${TAG:-${QUANTITY}_fit${FIT}}
OUT=${OUT:-$BASE/outputs/analysis/moving_ball_angaccel2d/fourier_${TAG}.json}
# NB: `${GATE_ONLY:---checkpoint $CKPT}` would expand to GATE_ONLY's VALUE when it is set, passing a stray
# "1" as a config override. Branch explicitly.
if [ -n "${GATE_ONLY:-}" ]; then MODE_ARGS="--gate_only"; else MODE_ARGS="--checkpoint $CKPT"; fi
echo "[steer_fourier] QUANTITY=$QUANTITY FIT=$FIT order=${ORDER:-4} harmonics=${HARM:-all} basis=${BASIS:-orientation} canon=$([ -n "${NO_CANON:-}" ] && echo off || echo on)"
echo "[steer_fourier] TRAIN=$TRAIN_DIR"
echo "[steer_fourier] TEST =$TEST_DIR"
# CONFIG must match the BACKBONE: it supplies encoder.layers, and ViT-g latents carry layers 10/20/30/38
# (matched relative depth) while the ViT-L config names 6/12/18/23 -> a mismatch is a KeyError, not a
# silent wrong answer.
CONFIG=${CONFIG:-configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml}
echo "[steer_fourier] CONFIG=$CONFIG"
python -u experiments/pipeline/05_steering/steer_fourier.py \
  --config "$CONFIG" \
  --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" \
  --quantity "$QUANTITY" \
  $MODE_ARGS \
  --n_train_scenes ${NTRAIN:-125} --n_test_scenes ${NTEST:-30} \
  --order ${ORDER:-4} --harmonics ${HARM:-all} --basis ${BASIS:-orientation} \
  ${NO_CANON:+--no_canon} \
  --ridge ${RIDGE:-10} --gains ${GAINS:-0.5,1,1.5,2,3,4} \
  ${PTG:+--per_token_gain} --ptg_scenes ${PTG_SCENES:-40} \
  --out "$OUT"
echo "[steer_fourier] exit=$? -> $OUT"
