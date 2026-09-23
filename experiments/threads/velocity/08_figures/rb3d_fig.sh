#!/bin/bash
#SBATCH --job-name=rb3d_fig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/rb3d_fig_%j.out
#SBATCH --error=logs/rb3d_fig_%j.err
# Render the steering proof figure (GT a / decode(H_a) / decode(H_a+cmd edit) / GT b).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs scratchpad
python -u experiments/threads/velocity/08_figures/rb3d_figure.py --scenes ${SCENES:-4} --gain ${GAIN:-2.0} --cmd_ku ${CMD_KU:-8} \
    --out ${OUT:-scratchpad/rb3d_steer_proof.png}
echo "[rb3d_fig] exit=$?"
