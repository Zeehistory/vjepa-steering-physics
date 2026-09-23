#!/bin/bash
#SBATCH --job-name=fgval_diag2
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/fgval_diag2_%j.out
#SBATCH --error=logs/fgval_diag2_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd .
python experiments/pipeline/07_eval/diag_decode.py   --config configs/train/moving_ball_decoder_large.yaml   --latent_dir outputs/latents/moving_ball_velocity/vjepa2_large   --checkpoint outputs/runs/moving_ball_decoder_fgval/checkpoints/step_300.pt   --output_dir outputs/analysis/moving_ball_velocity/diag_fgval --num_samples 3 --device cuda
