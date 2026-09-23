#!/bin/bash
#SBATCH --job-name=vjepa_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/steer_%j.out
#SBATCH --error=logs/steer_%j.err

# Step 3: learn a physical direction on a labelled (synthetic) cache, steer + decode a target cache
# (e.g. real Physics-IQ), and verify monotonic transfer of an independent readout.
# Requires: synthetic latents (slurm_extract_synthetic.sh), Physics-IQ latents, a trained decoder.
SOURCE_LATENT_DIR=${SOURCE_LATENT_DIR:-"outputs/latents/synthetic_solid/vjepa2_large"}
TARGET_LATENT_DIR=${TARGET_LATENT_DIR:-"outputs/latents/physics_iq/vjepa2_large"}
CHECKPOINT=${CHECKPOINT:-"outputs/runs/physics_iq_decoder_large/checkpoints/last.pt"}
VARIABLE=${VARIABLE:-"vel"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/analysis/steer/$VARIABLE"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/pipeline/05_steering/steer_decode.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --source_latent_dir "$SOURCE_LATENT_DIR" \
    --target_latent_dir "$TARGET_LATENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --variable "$VARIABLE" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
