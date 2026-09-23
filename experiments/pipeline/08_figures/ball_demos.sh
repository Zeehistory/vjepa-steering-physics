#!/bin/bash
#SBATCH --job-name=vjepa_ball_demos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/ball_demos_%j.out
#SBATCH --error=logs/ball_demos_%j.err

# Step 2 (velocity-first): generate qualitative demo videos for the three moving-ball scenarios.
# CPU only — no encoder, no GPU needed. Run this FIRST and inspect the videos before full-scale.
#
#   sbatch experiments/pipeline/08_figures/ball_demos.sh
#
# Outputs: outputs/demos/moving_ball/{constant_velocity,occlusion,rotated}/*.mp4 + *_strip.png
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/demos/moving_ball"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/pipeline/01_data/generate_ball_demos.py \
    --output_dir "$OUTPUT_DIR" \
    --n 8 \
    --image_size 128 \
    --num_frames 32 \
    --fps 8

echo "[ball_demos] done -> $OUTPUT_DIR"
