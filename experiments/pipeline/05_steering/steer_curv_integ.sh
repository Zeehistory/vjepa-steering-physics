#!/bin/bash
#SBATCH --job-name=curvsteer
#SBATCH --output=scratchpad/curvsteer_%j.log
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:b200:1
set -euo pipefail
cd .
PY=python
BASE=outputs
CKPT=$BASE/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
OUT=$BASE/analysis/moving_ball_accel2d_mixed/steer_curv_integ
$PY -u experiments/threads/acceleration/05_steering/steer_accel2d.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --test_dir $BASE/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
  --artifacts_dir $BASE/analysis/moving_ball_accel2d_mixed/subspace \
  --checkpoint "$CKPT" --output_dir "$OUT" \
  --ks 2,4,8,16 --num_scenes ${NUM_SCENES:-100} \
  --cmd_scales "${CMD_SCALES:-0.5,1.0,1.5,2.0,2.5,3.0,4.0}" \
  --features curv_integ --viz_scenes ${VIZ_SCENES:-6} --viz_gain ${VIZ_GAIN:-2.0} \
  --device cuda
echo "[curvsteer] steer exit=$?"
SUM=$OUT/steer2d_summary.json
if [ -f "$SUM" ]; then
  $PY -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 --out $OUT/calib_cmd_gain.json
fi
echo "DONE_CURVSTEER"
