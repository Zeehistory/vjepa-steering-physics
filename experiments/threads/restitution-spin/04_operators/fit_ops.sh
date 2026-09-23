#!/bin/bash
#SBATCH --job-name=spin_fitops
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/spin_fitops_%j.out
#SBATCH --error=logs/spin_fitops_%j.err

# Fit W_V / W_S / W_J on the full spin_ball3d train split, then draw the PCA figure. Latent-only, so
# neither step needs the decoder and both can run while it is still training.
#
# Memory is the real constraint, not compute: the joint operator's normal equations are
# 39 x (8*256*1024) float64 = 655 MB per layer, and there are three operators over four layers.

cd .
PY=python
export PYTHONPATH=.
export MUJOCO_GL=disable      # nothing renders here; keeps the GL import chain out of the way
set -eo pipefail

LAT=outputs/latents/spin_ball3d
ANA=${ANA:-outputs/analysis/spin_ball3d}

"$PY" -u experiments/threads/restitution-spin/04_operators/fit_spin_operators.py \
    --train_dir "$LAT/train/vjepa2_large" \
    --output_dir "$ANA/operators" \
    --layers "${LAYERS:-6,12,18,23}" \
    --num_scenes "${NUM_SCENES:-0}" \
    --max_cached_shards "${MAX_SHARDS:-1}"

test -s "$ANA/operators/operator_vel.npz" || { echo "FAIL: operators missing"; exit 1; }

"$PY" -u experiments/threads/restitution-spin/11_run/pca_viz.py \
    --test_dir "$LAT/test/vjepa2_large" \
    --operators_dir "$ANA/operators" \
    --output_dir "$ANA/pca" \
    --layer "${PCA_LAYER:-18}" --num_scenes 64 --num_arrows 32

echo "[fitops] done -> $ANA"
