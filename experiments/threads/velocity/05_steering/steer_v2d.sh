#!/bin/bash
#SBATCH --job-name=v2d_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2d_steer_%j.out
#SBATCH --error=logs/v2d_steer_%j.err
# Phases 3-5 pixel proof: decode + track the 2D-velocity steer for full_delta / transport{,_oracle,
# _shuffle} / subspace_U[k] / random[k] / ridge_global / canon_ridge on held-out test scenes.
# The decode loads only the small TEST cache (~80 clips), so a short GPU partition suffices:
#   sbatch -p <partition> experiments/threads/velocity/05_steering/steer_v2d.sh [CKPT]
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CK="${1:-last}"
OUTTAG="${2:-$CK}"   # output dir suffix (lets parallel decodes not clobber each other)
CKPT=$BASE/outputs/runs/moving_ball_scene_v2d_decoder_fp/checkpoints/${CK}.pt
python -u experiments/threads/velocity/05_steering/steer_velocity2d.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_v2d/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_v2d/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/moving_ball_v2d/steer_${OUTTAG} \
    --ks 2,4,8,16 --num_scenes ${NUM_SCENES:-40} --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
    --device cuda
echo "[v2d_steer] exit=$? ckpt=$CKPT out=steer_${OUTTAG}"
