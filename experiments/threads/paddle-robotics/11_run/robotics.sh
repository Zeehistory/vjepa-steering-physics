#!/bin/bash
#SBATCH --job-name=vjepa_robotics
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/robotics_%j.out
#SBATCH --error=logs/robotics_%j.err

# Step 4: detect achievable-vs-non-achievable latent structure, then steer failures toward success.
# Works on DROID (set LATENT_DIR to a DROID cache) or the robot_toy fallback. Omit CHECKPOINT to run
# detection only (no decoding).
LATENT_DIR=${LATENT_DIR:-"outputs/latents/robot_toy/vjepa2_large"}
CHECKPOINT=${CHECKPOINT:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/analysis/robotics"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

CKPT_ARG=""
if [ -n "$CHECKPOINT" ]; then CKPT_ARG="--checkpoint $CHECKPOINT"; fi

python experiments/threads/paddle-robotics/11_run/robotics_analysis.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $CKPT_ARG \
    --device cuda
