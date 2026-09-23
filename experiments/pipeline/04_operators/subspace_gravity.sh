#!/bin/bash
#SBATCH --job-name=grav_subspace
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gravity_subspace_%j.out
#SBATCH --error=logs/gravity_subspace_%j.err
# Fit the GRAVITY acceleration PCA subspace U (save_k 16 so U8 AND U16 coexist) + ridge operators on the
# TRAIN gravity latents; preview held-out TEST generalization. Reuses the dataset-agnostic accel_subspace
# script pointed at the gravity dirs. bigmem because global PCA buffers ~7GB/layer.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/accel_subspace.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_gravity/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_gravity/test/vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/moving_ball_gravity/subspace \
    --save_k 16
echo "[gravity_subspace] exit=$?"
