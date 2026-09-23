#!/bin/bash
#SBATCH --job-name=accel_cmdop
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accel_cmdop_%j.out
#SBATCH --error=logs/accel_cmdop_%j.err
# Fit the command-only acceleration operator f(a_a,a_b) -> U coords (cmd_U8) + the rich global ridge, on
# TRAIN accel latents, reusing the U basis from accel_subspace.py. KU=8 by default; set KU=16 in the env
# for the U16 fallback (the basis already has 16 rows from accel_subspace --save_k 16).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KU=${KU:-8}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d/subspace \
    --ridge 1.0 --ku $KU
echo "[accel_cmdop] exit=$? (KU=$KU)"
