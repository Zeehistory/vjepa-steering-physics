#!/bin/bash
#SBATCH --job-name=v2dmix_fp_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/v2dmix_fp_train_%j.out
#SBATCH --error=logs/v2dmix_fp_train_%j.err
# Retrain the faithful (frame_position) decoder on the APPEARANCE-MIXED 2D-velocity latents. Same config
# + losses as the v2d frame_position decoder (frame_position loss ties the soft centroid to GT
# obj0_pos_x/y every frame -> direction- AND appearance-agnostic, transfers unchanged). Resume-safe.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
requeue() { echo "[v2dmix_fp] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1

BASE=.
LATENT_DIR=$BASE/outputs/latents/moving_ball_scene_v2d_mixed/train/vjepa2_large
OUTPUT_DIR=$BASE/outputs/runs/moving_ball_scene_v2d_mixed_decoder_fp
CONFIG=configs/train/moving_ball_scene_decoder.yaml
MAX_STEPS=${MAX_STEPS:-8000}

RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[v2dmix_fp] resuming from $LATEST"
fi

echo "[v2dmix_fp] CONFIG=$CONFIG OUT=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS LAT=$LATENT_DIR"
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS train.ckpt_every=500 train.log_every=50 $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[v2dmix_fp] training process exited rc=$RC"
exit $RC
