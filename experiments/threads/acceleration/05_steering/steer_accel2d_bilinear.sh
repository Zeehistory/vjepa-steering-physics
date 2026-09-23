#!/bin/bash
#SBATCH --job-name=amix_blst
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accelmix_blsteer_%j.out
#SBATCH --error=logs/accelmix_blsteer_%j.err
# DECODE the 2nd-order translation-field accel operator + leakage-free gain calibration.
# Read steer_bilinear/calib_cmd_gain.json HELDOUT vs canon 14.46 + full_delta 10.07 ceiling.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_bilinear \
    --ks 8 --num_scenes ${NUM_SCENES:-100} \
    --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0,4.0}" \
    --features bilinear --viz_scenes 4 --viz_gain 2.0 --device cuda
SUM=$BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_bilinear/steer2d_summary.json
[ -f "$SUM" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
    --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_bilinear/calib_cmd_gain.json
echo "[amix_blsteer] DONE"
