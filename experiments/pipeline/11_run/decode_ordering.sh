#!/bin/bash
#SBATCH --job-name=torder
#SBATCH --output=scratchpad/torder_%j.log
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:b200:1
set -euo pipefail
cd .
PY=python
L=outputs/latents/moving_ball_scene_accel2d_mixed
CKPT=outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
OUT=outputs/analysis/moving_ball_accel2d_mixed/temporal_order
MODE=${MODE:-gaussian}
$PY experiments/pipeline/11_run/decode_temporal_ordering.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --test_dir $L/test/vjepa2_large --checkpoint $CKPT --output_dir $OUT \
  --num_clips 10 --mode $MODE --scale ${SCALE:-1.0} --seeds 3
echo "DONE_TORDER"
