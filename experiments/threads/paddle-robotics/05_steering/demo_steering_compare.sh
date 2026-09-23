#!/bin/bash
#SBATCH --job-name=steer_cmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/steer_cmp_%j.out
#SBATCH --error=logs/steer_cmp_%j.err

# Side-by-side "no steering vs after steering" demo videos.
#
# BATCH, not login node, for the reason documented in slurm_demo_paddle_strike.sh: at 768 px with the
# swing gap at 8x this holds a few hundred full-res frames in memory, and the login node's cgroup
# SIGKILLs it in a way that looks like success (exit 0, empty log, empty output dir). This one renders
# TWO rollouts per ratio and concatenates them, so the peak is roughly double -- hence 96G.
#
# Usage:
#   sbatch experiments/threads/paddle-robotics/05_steering/demo_steering_compare.sh
#   EMBODIMENT=franka_dynamic WIDE=1 LOOK=1 sbatch experiments/threads/paddle-robotics/05_steering/demo_steering_compare.sh

cd .

IMAGE_SIZE=${IMAGE_SIZE:-768}
EMBODIMENT=${EMBODIMENT:-paddle}
RATIOS=${RATIOS:-0.5,1.0,2.0,3.0}
NOMINAL=${NOMINAL:-1.0}
OUT=${OUT:-outputs/paddle_strike/steer_compare_${EMBODIMENT}_${IMAGE_SIZE}}

# Absolute interpreter path: a non-interactive batch shell has no `conda` function and no `python` on
# PATH, and sourcing the cluster's /etc/bashrc under `set -u` kills the job (unset BASHRCSOURCED).
PY=python
set -eo pipefail

export PYTHONPATH=.
export MUJOCO_GL=egl

EXTRA=""
if [ "${WIDE:-0}" = "1" ]; then EXTRA="--wide"; fi
if [ "${LOOK:-0}" = "1" ]; then EXTRA="$EXTRA --look"; fi

"$PY" -u experiments/threads/paddle-robotics/05_steering/demo_steering_compare.py \
    --output_dir "$OUT" \
    --embodiment "$EMBODIMENT" \
    --ratios "$RATIOS" \
    --nominal_ratio "$NOMINAL" \
    --image_size "$IMAGE_SIZE" \
    $EXTRA

# The silent-failure guard: an OOM-killed render can leave an empty directory behind a zero exit.
test -s "$OUT/compare_all.mp4" || { echo "FAIL: $OUT/compare_all.mp4 missing or empty"; exit 1; }
echo "wrote $OUT"
ls -la "$OUT"
