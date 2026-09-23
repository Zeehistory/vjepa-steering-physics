#!/bin/bash
#SBATCH --job-name=amix_probe
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/accelmix_probe_%j.out
#SBATCH --error=logs/accelmix_probe_%j.err
# Acceleration probe + probe->steer verification. Latent-only, single job (no decoder).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/06_probes/probe_accel.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --layers 6,12,18,23 --steer_gain 2.5 \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/accel_probe
echo "[amix_probe] exit=$?"
