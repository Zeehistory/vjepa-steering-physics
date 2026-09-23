#!/bin/bash
#SBATCH --job-name=viz_rb3d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/viz_rb3d_%j.out
#SBATCH --error=logs/viz_rb3d_%j.err
# Dump the FULL 16-frame 3D rolling-ball steering clips (not just the 4-frame strip) for the slide GIFs.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs scratchpad
python -u experiments/threads/velocity/08_figures/rb3d_figure.py --scenes ${SCENES:-6} --gain ${GAIN:-2.0} --cmd_ku ${CMD_KU:-8} \
    --out scratchpad/rb3d_steer_proof_viz.png \
    --dump ${DUMP:-scratchpad/viz_dumps/rb3d}
echo "[viz_rb3d] exit=$?"
