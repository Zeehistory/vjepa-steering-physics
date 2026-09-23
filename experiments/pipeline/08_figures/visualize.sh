#!/bin/bash
#SBATCH --job-name=vjepa_viz
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err

LATENT_DIR=${LATENT_DIR:-"outputs/latents/physics_iq_test/vjepa2_large"}
CHECKPOINT=${CHECKPOINT:-"outputs/runs/physics_iq_decoder_large/checkpoints/last.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/viz/physics_iq_decoder_large_test"}
NUM_SAMPLES=${NUM_SAMPLES:-10}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

python experiments/pipeline/08_figures/visualize_reconstructions.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --latent_dir "$LATENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --device cuda
