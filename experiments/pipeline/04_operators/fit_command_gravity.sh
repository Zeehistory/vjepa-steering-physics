#!/bin/bash
#SBATCH --job-name=grav_cmdop
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gravity_cmdop_%j.out
#SBATCH --error=logs/gravity_cmdop_%j.err
# Fit the command-only GRAVITY operator f(g_a,g_b) -> U coords (cmd_U8) + rich global ridge, on TRAIN
# gravity latents, reusing the U basis from the gravity subspace. KU=8 default; set KU=16 in the env for
# the U16 fallback (the basis has 16 rows). Reuses the dataset-agnostic fit_command_operators_accel script.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KU=${KU:-8}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_gravity/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_gravity/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_gravity/subspace \
    --ridge 1.0 --ku $KU
echo "[gravity_cmdop] exit=$? (KU=$KU)"
