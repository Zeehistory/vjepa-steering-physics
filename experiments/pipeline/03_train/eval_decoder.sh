#!/bin/bash
#SBATCH --job-name=vjepa_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

LATENT_DIR=${LATENT_DIR:-"outputs/latents/physics_iq/vjepa2_large"}
CHECKPOINT=${CHECKPOINT:-"outputs/runs/physics_iq_decoder_large/checkpoints/last.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/eval/physics_iq_decoder_large"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

python experiments/pipeline/07_eval/eval_decoder.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --latent_dir "$LATENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
