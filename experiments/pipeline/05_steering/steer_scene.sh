#!/bin/bash
#SBATCH --job-name=scene_steer
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scene_steer_%j.out
#SBATCH --error=logs/scene_steer_%j.err

# Difference-vector velocity steering on the held-out test scenes: decode H_a + alpha*(H_b - H_a)
# and verify the decoded ball's speed tracks alpha.
#   CKPT=.../checkpoints/last.pt sbatch experiments/pipeline/05_steering/steer_scene.sh

BASE_DIR=${BASE_DIR:-"."}
TEST_DIR=${TEST_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene/test/vjepa2_large"}
RUN_DIR=${RUN_DIR:-"${BASE_DIR}/outputs/runs/moving_ball_scene_decoder"}
CKPT=${CKPT:-"${RUN_DIR}/checkpoints/last.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/analysis/moving_ball_scene/diff_steer"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}
ALPHAS=${ALPHAS:-"-0.5,0,0.5,1.0,1.5"}
NUM_SCENES=${NUM_SCENES:-12}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[scene_steer] CKPT=$CKPT TEST_DIR=$TEST_DIR -> $OUTPUT_DIR"
python experiments/threads/velocity/05_steering/steer_velocity_diff.py \
    --config "$CONFIG" \
    --latent_dir "$TEST_DIR" \
    --checkpoint "$CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --alphas="$ALPHAS" \
    --num_scenes "$NUM_SCENES" \
    --mean_delta \
    --device cuda
STATUS=$?

echo "[scene_steer] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
