#!/bin/bash
#SBATCH --job-name=spin_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/spin_train_%j.out
#SBATCH --error=logs/spin_train_%j.err
# Faithful (frame_position) decoder for the velocity x spin CROSSTALK scene. Same config and losses as
# the rolling_ball3d decoder -- only the latents differ -- so the translation half of the crosstalk
# result stays comparable to that baseline. Resume-safe: USR1 (150s before the wall) requeues, and a
# restart picks up the newest checkpoint.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
requeue() { echo "[spin_train] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1

LAT_BASE=${LAT_BASE:-.}
LATENT_DIR=$LAT_BASE/outputs/latents/spin_ball3d/train/vjepa2_large
# Checkpoints on SCRATCH: a long run writes tens of GB of them and the 4TB project quota is shared and
# ~95% full. Only the final model matters and it is regenerable.
OUTPUT_DIR=${OUTPUT_DIR:-$LAT_BASE/outputs/runs/spin_ball3d_decoder}
CONFIG=${CONFIG:-configs/train/spin_ball3d_decoder.yaml}
MAX_STEPS=${MAX_STEPS:-8000}

RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[spin_train] resuming from $LATEST"
fi

echo "[spin_train] CONFIG=$CONFIG OUT=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS LAT=$LATENT_DIR"
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS train.ckpt_every=500 train.log_every=50 $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[spin_train] training process exited rc=$RC"
exit $RC
