#!/bin/bash
#SBATCH --job-name=fgval_train
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:55:00
#SBATCH --output=logs/fgval_train_%j.out
#SBATCH --error=logs/fgval_train_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd .
accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py   --config configs/train/moving_ball_decoder_large.yaml   --latent_dir outputs/latents/moving_ball_velocity/vjepa2_large   --output_dir outputs/runs/moving_ball_decoder_fgval   optim.max_steps=600 optim.warmup_steps=50 train.ckpt_every=300 train.log_every=50
