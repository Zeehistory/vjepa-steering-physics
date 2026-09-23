#!/bin/bash
#SBATCH --job-name=amix_appc
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accelmix_appc_%j.out
#SBATCH --error=logs/accelmix_appc_%j.err
# Fit the APPEARANCE-CONDITIONED accel operator (disentanglement push): [command || appc(H_a)] -> dH
# (full D). Reads the mixed accel latents, builds its own appearance PC basis from pooled H_a, writes
# cmd_Bappc_L*.npy + appc_mean/appc_basis into the accel_mixed subspace dir. Consumed by
# steer_accel2d.py --features appc. KA env sets appearance rank (default 16). No decoder / subspace dep.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KA=${KA:-16}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_appc.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --ridge 1.0 --ka $KA
echo "[amix_appc] exit=$? (KA=$KA)"
