#!/bin/bash
#SBATCH --job-name=vjepa_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Usage:
#   sbatch experiments/pipeline/03_train/train_decoder.sh
#
# Run extract first:  sbatch experiments/pipeline/02_encode/extract_latents.sh
# Or chain them:      sbatch --dependency=afterok:<extract_job_id> experiments/pipeline/03_train/train_decoder.sh

LATENT_DIR=${LATENT_DIR:-"outputs/latents/physics_iq/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/runs/physics_iq_decoder_large"}
CONFIG=${CONFIG:-"configs/train/physics_iq_transformer_large.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# one so we never lose training progress. step_*.pt are written every ckpt_every; last.pt at the end.
RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then RESUME="train.resume=$LATEST"; echo "[train] resuming from $LATEST"; fi
fi

# accelerate launch uses torchrun under the hood; one process per GPU
accelerate launch \
    --num_processes $SLURM_GPUS_ON_NODE \
    --mixed_precision bf16 \
    experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $RESUME
