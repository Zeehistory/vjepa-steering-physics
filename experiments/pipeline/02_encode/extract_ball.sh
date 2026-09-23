#!/bin/bash
#SBATCH --job-name=vjepa_extract_ball
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/extract_ball_%j.out
#SBATCH --error=logs/extract_ball_%j.err

# Step 2 (velocity-first): extract V-JEPA2-large latents for the three moving-ball datasets.
# Dataset is generated on-the-fly (no download needed).
#
# Submit all three in sequence after ball demos look good:
#   DATASET=moving_ball_velocity    sbatch experiments/pipeline/02_encode/extract_ball.sh
#   DATASET=moving_ball_occlusion   sbatch experiments/pipeline/02_encode/extract_ball.sh
#   DATASET=moving_ball_equivariance sbatch experiments/pipeline/02_encode/extract_ball.sh
#
# Or submit all three in parallel (they are independent):
#   for D in moving_ball_velocity moving_ball_occlusion moving_ball_equivariance; do
#       DATASET=$D sbatch experiments/pipeline/02_encode/extract_ball.sh
#   done
DATASET=${DATASET:-"moving_ball_velocity"}
BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/${DATASET}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/physics_iq_transformer_large.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --encoder vjepa2_large \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --shard_size 128 \
    data.image_size=128 \
    data.num_frames=32 \
    encoder.image_size=128 \
    encoder.num_frames=32

echo "[extract_ball] $DATASET -> $OUTPUT_DIR"
