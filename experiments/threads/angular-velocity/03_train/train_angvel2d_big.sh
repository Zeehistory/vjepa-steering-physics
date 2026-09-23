#!/bin/bash
#SBATCH --job-name=angvel_big_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/angvel_big_train_%j.out
#SBATCH --error=logs/angvel_big_train_%j.err
# Train the faithful (frame_position + frame_orientation) decoder on the BIG-OBJECT angular-velocity latents.
# Resume-safe. Big bar (radius 0.32-0.42) should render crisply -> fixes the small-object rendering wall.
#   sbatch --partition=gpu_devel --gres=gpu:b200:1 experiments/threads/angular-velocity/03_train/train_angvel2d_big.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
requeue() { echo "[angvel_big] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1
BASE=.
LATENT_DIR=$BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large
OUTPUT_DIR=$BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient
CONFIG=configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml
MAX_STEPS=${MAX_STEPS:-8000}
RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[angvel_big] resuming from $LATEST"
fi
echo "[angvel_big] CONFIG=$CONFIG OUT=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS LAT=$LATENT_DIR"
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS train.ckpt_every=500 train.log_every=50 $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[angvel_big] training process exited rc=$RC"
exit $RC
