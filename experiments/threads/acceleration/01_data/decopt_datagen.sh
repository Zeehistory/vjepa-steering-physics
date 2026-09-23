#!/bin/bash
#SBATCH --job-name=decgen
#SBATCH --output=scratchpad/decgen_%j.log
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:b200:1
set -euo pipefail
cd .
PY=python
BASE=outputs
CKPT=$BASE/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ART=$BASE/analysis/moving_ball_accel2d_mixed/subspace
DUMP=$BASE/analysis/moving_ball_accel2d_mixed/decopt_edits
# TRAIN latents (distillation targets); shard via SCENE_START/SCENE_END across parallel jobs.
$PY -u experiments/threads/acceleration/05_steering/steer_accel_decopt.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --test_dir $BASE/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
  --artifacts_dir $ART --checkpoint "$CKPT" \
  --output_dir $BASE/analysis/moving_ball_accel2d_mixed/decopt_datagen_${SLURM_JOB_ID:-x} \
  --mode free --init canon --steps ${STEPS:-300} --lr ${LR:-0.08} --anchor ${ANCHOR:-1.0} \
  --scene_start ${SCENE_START:-0} --scene_end ${SCENE_END:-0} --num_scenes ${NUM_SCENES:-100} \
  --dump_dir $DUMP --device cuda
echo "DONE_DECGEN shard=${SCENE_START:-0}:${SCENE_END:-0}"
