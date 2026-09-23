#!/bin/bash
#SBATCH --job-name=scene_fp_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/scene_fp_train_%j.out
#SBATCH --error=logs/scene_fp_train_%j.err
# Train the scene diff-steering decoder WITH the motion-faithfulness loss (frame_position/frame_spread,
# now active in configs/train/moving_ball_scene_decoder.yaml). FRESH output dir so it does NOT resume the
# old smeared (magnitude-gap) checkpoint. ckpt_every=500 for preemption safety + early faithfulness check.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

requeue() { echo "[scene_fp] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1

BASE=.
LATENT_DIR=$BASE/outputs/latents/moving_ball_scene/train/vjepa2_large
OUTPUT_DIR=$BASE/outputs/runs/moving_ball_scene_decoder_fp
CONFIG=configs/train/moving_ball_scene_decoder.yaml
MAX_STEPS=${MAX_STEPS:-6000}

RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[scene_fp] resuming from $LATEST"
fi

echo "[scene_fp] CONFIG=$CONFIG OUT=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS"
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS train.ckpt_every=500 train.log_every=50 $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[scene_fp] training process exited rc=$RC"
exit $RC
