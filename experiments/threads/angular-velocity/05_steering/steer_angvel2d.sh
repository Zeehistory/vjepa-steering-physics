#!/bin/bash
#SBATCH --job-name=angvel_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/angvel_steer_%j.out
#SBATCH --error=logs/angvel_steer_%j.err
# Decode + track the ANGULAR-VELOCITY steer on held-out test scenes, with built-in leakage-free gain
# calibration (scalar). CMD_KU selects U8 (default) or U16. Run AFTER decoder ckpt + cmd-fit both exist:
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CK="${1:-last}"
OUTTAG="${2:-$CK}"
CKPT=$BASE/outputs/runs/moving_ball_scene_angvel2d_decoder_fp/checkpoints/${CK}.pt
python -u experiments/threads/angular-velocity/05_steering/steer_angvel2d.py \
    --config configs/train/moving_ball_scene_angvel_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_angvel2d/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/moving_ball_angvel2d/steer_${OUTTAG} \
    --ks 2,4,8,16 --num_scenes ${NUM_SCENES:-100} --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
    --cmd_ku ${CMD_KU:-8} --viz_scenes ${VIZ_SCENES:-6} \
    --device cuda
RC=$?
echo "[angvel_steer] done (exit $RC) ckpt=$CKPT out=steer_${OUTTAG}"
