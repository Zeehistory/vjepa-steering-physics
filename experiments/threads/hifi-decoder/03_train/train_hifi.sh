#!/bin/bash
#SBATCH --job-name=hifi_train
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hifi_train_%j.out
#SBATCH --error=logs/hifi_train_%j.err

# High-fidelity decoder: convup frame head + LPIPS + a loss rebalanced toward appearance.
#   NAME=v2d_mixed sbatch experiments/threads/hifi-decoder/03_train/train_hifi.sh
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
NAME=${NAME:-v2d_mixed}
python experiments/pipeline/03_train/train_decoder.py --config "configs/train/hifi_${NAME}_decoder.yaml"
