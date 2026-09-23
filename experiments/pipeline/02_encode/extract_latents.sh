#!/bin/bash
#SBATCH --job-name=vjepa_extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# Usage:
#   sbatch experiments/pipeline/02_encode/extract_latents.sh
#
# Set PHYSICS_IQ_ROOT below to your Physics-IQ dataset path on the cluster,
# or pass it as an env var:  PHYSICS_IQ_ROOT=/path/to/data sbatch ...

PHYSICS_IQ_ROOT=${PHYSICS_IQ_ROOT:-"data/physics_iq"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/latents/physics_iq/vjepa2_large"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

python experiments/pipeline/02_encode/extract_latents.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --shard_size 128 \
    "data.root=$PHYSICS_IQ_ROOT"
