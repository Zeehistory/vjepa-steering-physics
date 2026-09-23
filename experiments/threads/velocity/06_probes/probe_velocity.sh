#!/bin/bash
#SBATCH --job-name=vjepa_vel_probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=logs/vel_probe_%j.out
#SBATCH --error=logs/vel_probe_%j.err

# Step 2 (velocity-first): temporal velocity probe (clip_pool vs temporal 8x1024 vs temporal_diff).
# CPU/sklearn. Must run after slurm_extract_ball.sh for moving_ball_velocity.
#
#   sbatch experiments/threads/velocity/06_probes/probe_velocity.sh
BASE_DIR=${BASE_DIR:-"."}
LATENT_DIR=${LATENT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_velocity/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/analysis/moving_ball_velocity/velocity_probe"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/threads/velocity/06_probes/probe_velocity.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layers all

echo "[vel_probe] done -> $OUTPUT_DIR"
