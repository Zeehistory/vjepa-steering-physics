#!/bin/bash
#SBATCH --job-name=stepsweep
#SBATCH --output=scratchpad/stepsweep_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:b200:1
set -euo pipefail
cd .
PY=python
BASE=outputs
CKPT=$BASE/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ART=$BASE/analysis/moving_ball_accel2d_mixed/subspace
# Fix A: canon-init warm-start test-time optimization. How few steps to reach the 300-step 5.07deg?
for S in 20 40 80 160; do
  $PY -u experiments/threads/acceleration/05_steering/steer_accel_decopt.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --artifacts_dir $ART --checkpoint "$CKPT" \
    --output_dir $BASE/analysis/moving_ball_accel2d_mixed/steer_stepsweep_$S \
    --mode free --init canon --steps $S --lr 0.08 --anchor 1.0 --num_scenes ${NUM_SCENES:-25} --device cuda
  echo "STEPSWEEP steps=$S done"
done
echo "DONE_STEPSWEEP"
