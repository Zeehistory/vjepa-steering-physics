#!/bin/bash
#SBATCH --job-name=plan_diagfast
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=scratchpad/plan_diagfast_%j.log
#SBATCH --requeue
set -uo pipefail
cd .
PY=python
BASE=outputs
ANA=$BASE/analysis/moving_ball_accel2d_mixed
CFG=configs/train/moving_ball_scene_decoder.yaml
CKPT=$BASE/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
TRAIN=$BASE/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large
TEST=$BASE/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
SUB=$ANA/subspace
echo "===== D5 trackers =====";  $PY -u experiments/threads/planning-tracks/11_run/plan_d5_trackers.py --config $CFG --train_dir $TRAIN --edits_dir $ANA/decopt_edits --checkpoint $CKPT --output_dir $ANA/plan_d5 --num_scenes 30 --device cuda
echo "===== D2 support =====";   $PY -u experiments/threads/planning-tracks/11_run/plan_d2_support.py --config $CFG --train_dir $TRAIN --edits_dir $ANA/decopt_edits --checkpoint $CKPT --output_dir $ANA/plan_d2 --num_scenes 16 --device cuda
echo "===== Track T =====";      $PY -u experiments/threads/planning-tracks/04_operators/plan_trackT_transport.py --config $CFG --test_dir $TEST --checkpoint $CKPT --output_dir $ANA/plan_trackT --variant both --polish 20 --num_scenes 20 --device cuda
echo "===== D4/F critic =====";  $PY -u experiments/threads/planning-tracks/11_run/plan_d4F_critic.py --config $CFG --train_dir $TRAIN --edits_dir $ANA/decopt_edits --artifacts_dir $SUB --checkpoint $CKPT --output_dir $ANA/plan_d4F --num_scenes 70 --device cuda
echo "DONE_DIAGFAST"
