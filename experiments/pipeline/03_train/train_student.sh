#!/bin/bash
#SBATCH --job-name=student
#SBATCH --output=scratchpad/student_%j.log
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:b200:1
set -euo pipefail
cd .
PY=python
BASE=outputs
CKPT=$BASE/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
RESID_ARGS=""
if [ "${RESIDUAL:-0}" = "1" ]; then
  RESID_ARGS="--residual_canon --artifacts_dir $BASE/analysis/moving_ball_accel2d_mixed/subspace"
fi
$PY -u experiments/threads/acceleration/03_train/train_decopt_student.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $BASE/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
  --test_dir  $BASE/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
  --checkpoint "$CKPT" \
  --edits_dir $BASE/analysis/moving_ball_accel2d_mixed/decopt_edits \
  --output_dir $BASE/analysis/moving_ball_accel2d_mixed/decopt_student_${TAG:-plain} \
  --epochs ${EPOCHS:-80} --batch ${BATCH:-4} --lr ${LR:-1e-3} \
  --weight_decay ${WD:-1e-4} --hidden ${HIDDEN:-512} --cos_weight ${COSW:-0.0} \
  --decode_weight ${DECODEW:-0.0} --anchor ${ANCHOR:-1.0} \
  --eval_every ${EVAL_EVERY:-5} --eval_n ${EVAL_N:-30} $RESID_ARGS --device cuda
echo "DONE_STUDENT tag=${TAG:-plain}"
