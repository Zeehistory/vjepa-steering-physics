#!/bin/bash
#SBATCH --job-name=fg4_diag
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/fg4_diag_%j.out
#SBATCH --error=logs/fg4_diag_%j.err
# Decode the early fg4 checkpoint to confirm the ball is rendered (escapes uniform-collapse).
# Pass the checkpoint step as the first arg, e.g.: sbatch experiments/pipeline/07_eval/diag_fg4.sh step_200
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd .
CK="${1:-last}"
python experiments/pipeline/07_eval/diag_decode.py \
  --config configs/train/moving_ball_decoder_large.yaml \
  --latent_dir outputs/latents/moving_ball_velocity/vjepa2_large \
  --checkpoint outputs/runs/moving_ball_decoder_fg4/checkpoints/${CK}.pt \
  --output_dir outputs/analysis/moving_ball_velocity/diag_fg4_${CK} \
  --num_samples 3 --device cuda
