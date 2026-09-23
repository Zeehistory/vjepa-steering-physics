#!/bin/bash
#SBATCH --job-name=angvel_cmdop
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/angvel_cmdop_%j.out
#SBATCH --error=logs/angvel_cmdop_%j.err
# Fit the command-only angular-velocity operator f(w_a,w_b) -> U coords (cmd_U8) + rich global ridge, on
# TRAIN angvel latents, reusing the U basis from accel_subspace.py --quantity angvel. KU=8 default; KU=16 env.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KU=${KU:-8}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large \
    --layers 6,12,18,23 --quantity angvel \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_angvel2d/subspace \
    --ridge 1.0 --ku $KU
echo "[angvel_cmdop] exit=$? (KU=$KU)"
