#!/bin/bash
#SBATCH --job-name=actablate
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/actablate_%j.out
#SBATCH --error=logs/actablate_%j.err

# Ablations for the inverse action predictor. Every row is a full closed loop, so this is longer than
# the main run despite reusing the same cached descriptors.
#
#   EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/07_eval/ablate_action_predictor.sh
#
# 120G rather than the main run's 96G: the layer/k ablations hold a sliced COPY of the train-post
# descriptors alongside the original, and the Gram step then needs a centred copy of the slice.

EMBODIMENT=${EMBODIMENT:-"paddle"}
LAM=${LAM:-1.0}
FAMILY=${FAMILY:-"ridge"}
BASE=${BASE:-"outputs/paddle_strike"}

module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "[actablate] embodiment=$EMBODIMENT lam=$LAM family=$FAMILY"
PYTHONPATH=. python experiments/threads/paddle-robotics/07_eval/ablate_action_predictor.py \
    --latent_root "${BASE}/latents/${EMBODIMENT}" \
    --output_dir "${BASE}/action_predictor/${EMBODIMENT}_ablations" \
    --feat_cache "${BASE}/action_loop/featcache" \
    --embodiment "$EMBODIMENT" --lam "$LAM" --family "$FAMILY"
