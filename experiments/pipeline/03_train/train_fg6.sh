#!/bin/bash
#SBATCH --job-name=fg6_train
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@150
#SBATCH --open-mode=append
#SBATCH --output=logs/fg6_train_%j.out
#SBATCH --error=logs/fg6_train_%j.err
# fg6: fresh run with the target-compression fix (loss.target_lo/hi=0.05/0.95). gpu_devel has zero
# pending queue -> schedules instantly. Chains 1h chunks, requeuing from the newest checkpoint, until
# step 4000. NOTE: new output dir (fg6) so it does NOT resume fg3's white-collapsed checkpoint.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd .

requeue() { echo "[fg6_train] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1

RESUME=""
CKDIR="outputs/runs/moving_ball_decoder_fg6/checkpoints"
if [ -d "$CKDIR" ]; then
  LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
  [ -n "$LATEST" ] && RESUME="train.resume=$LATEST" && echo "[fg6_train] resuming from $LATEST"
fi

accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
  --config configs/train/moving_ball_decoder_large.yaml \
  --latent_dir outputs/latents/moving_ball_velocity/vjepa2_large \
  --output_dir outputs/runs/moving_ball_decoder_fg6 \
  optim.max_steps=4000 train.ckpt_every=200 train.log_every=50 train.num_workers=0 $RESUME &
CHILD=$!
wait $CHILD
RC=$?
echo "[fg6_train] training process exited rc=$RC"
exit $RC
