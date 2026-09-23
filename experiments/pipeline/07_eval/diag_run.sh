#!/bin/bash
#SBATCH --job-name=diag_decode
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd .
python experiments/pipeline/07_eval/diag_decode.py   --config configs/train/physics_iq_transformer_large.yaml   --latent_dir outputs/latents/moving_ball_velocity/vjepa2_large   --checkpoint outputs/runs/moving_ball_decoder/checkpoints/last.pt   --output_dir outputs/analysis/moving_ball_velocity/diag --num_samples 3 --device cuda   data.image_size=128 data.num_frames=32 decoder.out_image_size=128 decoder.out_num_frames=32
