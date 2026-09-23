#!/bin/bash
#SBATCH --job-name=vjepa_qprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/qprobe_%j.out
#SBATCH --error=logs/qprobe_%j.err

# Step 2: layerwise quantity probes (velocity / acceleration / gravity / ...) on a SYNTHETIC latent
# cache that has exact ground-truth state. CPU/sklearn work, but run on a compute node (logins are
# memory-capped). Submit from a shell with the conda env active.
#   LATENT_DIR=.../latents/synthetic_solid/vjepa2_large sbatch experiments/pipeline/03_train/train_probe.sh
LATENT_DIR=${LATENT_DIR:-"outputs/latents/synthetic_solid/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/analysis/synthetic_solid/quantity_probe"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/pipeline/03_train/train_probe.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layers all
