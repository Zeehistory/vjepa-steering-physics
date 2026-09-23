#!/bin/bash
#SBATCH --job-name=gravdec
#SBATCH --output=scratchpad/gravdec_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:b200:1
set -euo pipefail
cd .
PY=python
BASE=outputs
CKPT=$BASE/runs/moving_ball_scene_gravity_decoder_fp/checkpoints/last.pt
ART=$BASE/analysis/moving_ball_gravity/subspace
OUT=$BASE/analysis/moving_ball_gravity/steer_decopt
$PY -u experiments/threads/acceleration/05_steering/steer_accel_decopt.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --test_dir $BASE/latents/moving_ball_scene_gravity/test/vjepa2_large \
  --artifacts_dir $ART --checkpoint "$CKPT" --output_dir "$OUT" \
  --mode free --init zero --steps ${STEPS:-300} --lr ${LR:-0.08} --anchor ${ANCHOR:-1.0} \
  --num_scenes ${NUM_SCENES:-30} --viz_scenes ${VIZ_SCENES:-4} --device cuda
echo "DONE_GRAVDEC"
