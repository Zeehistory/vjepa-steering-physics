#!/bin/bash
#SBATCH --job-name=amix_slabvel
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/amix_slabvel_%j.out
#SBATCH --error=logs/amix_slabvel_%j.err
# C: fit the TEMPORAL-COMPOSITION per-slab velocity operator (accel = time-composed velocity edits).
# Writes slabvel_basis_L*.npy + cmd_Wslabvel_L*.npy into the MAIN subspace dir (reuses cmd_Brich there).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_slabvel.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --ku ${KU:-32} --max_slab ${MAX_SLAB:-8000}
echo "[amix_slabvel] exit=$?"
