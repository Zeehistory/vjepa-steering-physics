#!/bin/bash
#SBATCH --job-name=rb3d_cmdop
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rb3d_cmdop_%j.out
#SBATCH --error=logs/rb3d_cmdop_%j.err
# Fit COMMAND-ONLY subspace-synthesis operators on the 3D rolling-ball cache (W_U: command -> U-coords;
# B_rich: rich command -> dH). KU defaults to 8; set KU=16 for the U16 fallback (needs save_k>=16 basis).
# ALWAYS submit via the queue-aware picker:
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 256000 bigmem mpi week day); sbatch -p <partition> --mem=256G experiments/threads/velocity/04_operators/fit_command_rb3d.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAT_BASE=${LAT_BASE:-.}  # latents on scratch (see slurm_extract_rb3d.sh)
python -u experiments/pipeline/04_operators/fit_command_operators.py \
    --train_dir $LAT_BASE/outputs/latents/rolling_ball3d/train/vjepa2_large \
    --test_dir  $LAT_BASE/outputs/latents/rolling_ball3d/test/vjepa2_large \
    --layers ${LAYERS:-6,12,18,23} --ridge 1.0 --ku ${KU:-8} ${POS_FEATURES:+--pos_features} \
    --artifacts_dir $BASE/outputs/analysis/rolling_ball3d/subspace
echo "[rb3d_cmdop] exit=$? (KU=${KU:-8} POS=${POS_FEATURES:-0})"
