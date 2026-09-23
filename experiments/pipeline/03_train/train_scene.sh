#!/bin/bash
#SBATCH --job-name=scene_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/scene_train_%j.out
#SBATCH --error=logs/scene_train_%j.err

# Train the scene-paired moving-ball RGB decoder (256x256, 16 frames).
#   LATENT_DIR=.../moving_ball_scene/train/vjepa2_large \
#   OUTPUT_DIR=.../runs/moving_ball_scene_decoder  sbatch experiments/pipeline/03_train/train_scene.sh
# Pilot:  MAX_STEPS=1500 sbatch --mem=96G -t 03:00:00 experiments/pipeline/03_train/train_scene.sh

BASE_DIR=${BASE_DIR:-"."}
LATENT_DIR=${LATENT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene/train/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/runs/moving_ball_scene_decoder"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}
MAX_STEPS=${MAX_STEPS:-6000}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Resume from the newest checkpoint if one exists (requeue-safe).
RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then RESUME="train.resume=$LATEST"; echo "[scene_train] resuming from $LATEST"; fi
fi

echo "[scene_train] CONFIG=$CONFIG LATENT_DIR=$LATENT_DIR OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS"
accelerate launch \
    --num_processes ${SLURM_GPUS_ON_NODE:-1} \
    --mixed_precision bf16 \
    experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS \
    $RESUME
STATUS=$?

echo "[scene_train] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
