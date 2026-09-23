#!/bin/bash
#SBATCH --job-name=amix_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accelmix_steer_%j.out
#SBATCH --error=logs/accelmix_steer_%j.err
# Decode + track the 2D-ACCELERATION steer: full_delta / subspace_U[k] / random[k] / ridge_global /
# cmd_U8 (gain sweep) / ridge_rich on held-out test scenes, then leakage-free gain calibration. CMD_KU
# selects U8 (default) or the U16 operator. Run AFTER the decoder ckpt + cmd-fit both exist (submit
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CK="${1:-last}"
OUTTAG="${2:-$CK}"
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/${CK}.pt
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_${OUTTAG} \
    --ks 2,4,8,16 --num_scenes ${NUM_SCENES:-100} --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
    --cmd_ku ${CMD_KU:-8} --features ${FEATURES:-base} \
    --viz_scenes ${VIZ_SCENES:-6} --viz_gain ${VIZ_GAIN:-2.1} \
    --device cuda
RC=$?
echo "[amix_steer] steer exit=$RC ckpt=$CKPT out=steer_${OUTTAG}"

SUM=$BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_${OUTTAG}/steer2d_summary.json
if [ -f "$SUM" ]; then
    python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
        --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_${OUTTAG}/calib_cmd_gain.json
fi
echo "[amix_steer] done (exit $RC)"
