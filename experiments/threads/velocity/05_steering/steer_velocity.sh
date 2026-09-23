#!/bin/bash
#SBATCH --job-name=vjepa_steer_vel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=logs/steer_vel_%j.out
#SBATCH --error=logs/steer_vel_%j.err

# Step 2 (velocity-first): steer the velocity subspace and decode to pixels.
# The key interim milestone: add alpha*d_v to latents, decode, and visually verify the ball's
# decoded speed/direction changes with alpha. Writes filmstrips + mp4s + controllability plots.
#
# Must run AFTER:
#   1. slurm_extract_ball.sh (moving_ball_velocity) — needs the labelled latent cache
#   2. slurm_train_decoder.sh — needs a trained transformer decoder checkpoint
#
# To steer speed:
#   TARGET=speed  sbatch experiments/threads/velocity/05_steering/steer_velocity.sh
# To steer horizontal velocity:
#   TARGET=vel_x  sbatch experiments/threads/velocity/05_steering/steer_velocity.sh
# To steer vertical velocity:
#   TARGET=vel_y  sbatch experiments/threads/velocity/05_steering/steer_velocity.sh
TARGET=${TARGET:-"speed"}
BASE_DIR=${BASE_DIR:-"."}
SOURCE_LATENT_DIR=${SOURCE_LATENT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_velocity/vjepa2_large"}
TARGET_LATENT_DIR=${TARGET_LATENT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_velocity/vjepa2_large"}
CHECKPOINT=${CHECKPOINT:-"${BASE_DIR}/outputs/runs/moving_ball_decoder/checkpoints/last.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/analysis/moving_ball_velocity/steer_${TARGET}"}
CONFIG=${CONFIG:-"configs/train/physics_iq_transformer_large.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/threads/velocity/05_steering/steer_velocity.py \
    --config "$CONFIG" \
    --source_latent_dir "$SOURCE_LATENT_DIR" \
    --target_latent_dir "$TARGET_LATENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET" \
    --all_layers \
    --alphas="-1.5,-1.0,-0.5,0,0.5,1.0,1.5" \
    --num_samples 6 \
    --device cuda \
    data.image_size=128 \
    data.num_frames=32 \
    data.fps=8 \
    decoder.out_image_size=128 \
    decoder.out_num_frames=32
rc=$?

# Don't mask failures: a crashed python (e.g. missing arg) must surface as a failed job,
# not a COMPLETED 0:0 that the collector mistakes for a successful steering run.
if [ "$rc" -ne 0 ]; then
    echo "[steer_vel] target=$TARGET FAILED (exit $rc)"
    exit "$rc"
fi
echo "[steer_vel] target=$TARGET done -> $OUTPUT_DIR"
