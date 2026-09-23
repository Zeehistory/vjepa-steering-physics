#!/bin/bash
#SBATCH --job-name=angaccel_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/angaccel_train_%j.out
#SBATCH --error=logs/angaccel_train_%j.err
# Train the faithful (frame_position + frame_orientation) decoder on the ANGULAR-ACCELERATION latents, so
# the renderer is in-distribution for a ramping spin and stops capping the alpha steering numbers (the
# constant-omega decoder tops out at ceiling rho=0.809 on alpha clips vs 0.98 on the spins it trained on).
# The decoder is only the measurement instrument -- the steering operator stays fit on constant-omega data.
# Resume-safe (requeue on USR1/timeout), and ckpt_every=2000 from the config keeps the checkpoint footprint
# ~16G: the angvel run's 500-step cadence produced 16 x 3.3G and blew the group quota at step 7000.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
requeue() { echo "[angaccel_train] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1
BASE=.
LATENT_DIR=$BASE/outputs/latents/moving_ball_scene_angaccel2d_train/test/vjepa2_large
OUTPUT_DIR=$BASE/outputs/runs/moving_ball_scene_angaccel2d_decoder_orient
CONFIG=configs/train/moving_ball_scene_angaccel_big_decoder_orient.yaml
MAX_STEPS=${MAX_STEPS:-8000}
RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[angaccel_train] resuming from $LATEST"
fi
echo "[angaccel_train] CONFIG=$CONFIG OUT=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS LAT=$LATENT_DIR"
df -h . | tail -1
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS train.log_every=50 $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[angaccel_train] training process exited rc=$RC"
df -h . | tail -1
exit $RC
