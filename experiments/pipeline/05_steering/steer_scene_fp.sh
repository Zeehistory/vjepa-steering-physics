#!/bin/bash
#SBATCH --job-name=scene_fp_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/scene_fp_steer_%j.out
#SBATCH --error=logs/scene_fp_steer_%j.err
# Re-run the difference-vector velocity-steering MEASUREMENT against the frame-position decoder, on the
# 20 held-out test scenes. Proves (or disproves) the magnitude-gap fix: decoded-vs-GT r and the
# per-scene alpha=0/alpha=1 reconstruction speeds should now match GT (was r~0.34, decoded ~4x slow).
# Usage: sbatch experiments/pipeline/05_steering/steer_scene_fp.sh [CKPT_NAME]   (default last)
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

BASE=.
CK="${1:-last}"
CKPT=$BASE/outputs/runs/moving_ball_scene_decoder_fp/checkpoints/${CK}.pt
OUT=$BASE/outputs/analysis/moving_ball_scene/diff_steer_fp_${CK}

echo "[scene_fp_steer] checkpoint=$CKPT"
python experiments/threads/velocity/05_steering/steer_velocity_diff.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --latent_dir $BASE/outputs/latents/moving_ball_scene/test/vjepa2_large \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --alphas=-0.5,0,0.5,1.0,1.5 \
    --num_scenes 20 \
    --mean_delta \
    --device cuda
echo "[scene_fp_steer] exit=$? -> $OUT"
