#!/bin/bash
#SBATCH --job-name=rb3d_fp_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/rb3d_fp_train_%j.out
#SBATCH --error=logs/rb3d_fp_train_%j.err
# Retrain the faithful (frame_position) decoder on the 3D MuJoCo rolling-ball latents. Same config
# + losses as the v2d frame_position decoder (frame_position loss ties the soft centroid to GT
# obj0_pos_x/y every frame -> direction- AND appearance-agnostic, transfers unchanged). Resume-safe.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
requeue() { echo "[rb3d_fp] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1

BASE=.
LAT_BASE=${LAT_BASE:-.}  # latents on scratch (see slurm_extract_rb3d.sh)
LATENT_DIR=$LAT_BASE/outputs/latents/rolling_ball3d/train/vjepa2_large
# Checkpoints go to SCRATCH too: at ckpt_every=500 an 8000-step run writes 16 x 3.3G = 53G, which
# on its own would exhaust the shared 4TB project quota mid-run (it got to 14G free before this was
# caught). Only the final model matters and it is regenerable, so scratch is the right home.
#
# SEED trains an INDEPENDENT decoder for the decoder-independence test. It overrides train.seed
# (weight init + batch order) and deliberately leaves data.seed alone, so decoder B sees exactly the
# same clips as decoder A and differs only in its own randomness -- changing the data seed too would
# confound "a different decoder" with "different training data".
#   SEED=1 sbatch experiments/threads/velocity/03_train/train_rb3d_fp.sh    ->  runs/rolling_ball3d_decoder_fp_seed1
SEED=${SEED:-}
if [ -n "$SEED" ]; then
    OUTPUT_DIR=$LAT_BASE/outputs/runs/rolling_ball3d_decoder_fp_seed${SEED}
    SEED_OVERRIDE="train.seed=$SEED"
else
    OUTPUT_DIR=$LAT_BASE/outputs/runs/rolling_ball3d_decoder_fp
    SEED_OVERRIDE=""
fi
CONFIG=configs/train/rolling_ball3d_decoder.yaml
MAX_STEPS=${MAX_STEPS:-8000}

RESUME=""
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[rb3d_fp] resuming from $LATEST"
fi

echo "[rb3d_fp] CONFIG=$CONFIG OUT=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS LAT=$LATENT_DIR"
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    optim.max_steps=$MAX_STEPS train.ckpt_every=500 train.log_every=50 $SEED_OVERRIDE $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[rb3d_fp] training process exited rc=$RC"
exit $RC
