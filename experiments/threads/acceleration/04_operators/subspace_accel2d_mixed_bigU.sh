#!/bin/bash
#SBATCH --job-name=amix_bigU
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/amix_bigU_%j.out
#SBATCH --error=logs/amix_bigU_%j.err
# B: enlarge the accel subspace. Re-run accel_subspace with save_k 128 (and more global pairs for a robust
# high-rank basis) into a SEPARATE subspace_bigU dir so the committed canon artifacts are untouched.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/accel_subspace.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace_bigU \
    --save_k 128 --max_global_pairs 1600
echo "[amix_bigU] exit=$?"
