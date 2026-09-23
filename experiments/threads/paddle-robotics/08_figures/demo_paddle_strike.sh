#!/bin/bash
#SBATCH --job-name=pstrike_demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/pstrike_demo_%j.out
#SBATCH --error=logs/pstrike_demo_%j.err

# Render the paddle-strike demo figures (filmstrips, contact close-up, contact trace, episode mp4s).
#
# This is a BATCH job rather than a login-node command for one measured reason: at --image_size 512
# and up, with the swing gap rendered at 8x, the script holds a few hundred full-resolution frames in
# memory at once and the login node's cgroup SIGKILLs it. That failure is nearly silent -- the shell
# reports "Killed" but a backgrounded wrapper reported exit 0 with an empty log and an empty output
# directory, which looks exactly like a successful run that happened to print nothing. If you are
# tempted to run this interactively, note that the 256 px dataset render is the only size that fits.
#
# Usage:
#   sbatch experiments/threads/paddle-robotics/08_figures/demo_paddle_strike.sh
#   IMAGE_SIZE=1024 EMBODIMENT=franka OUT=/path sbatch experiments/threads/paddle-robotics/08_figures/demo_paddle_strike.sh

cd .

IMAGE_SIZE=${IMAGE_SIZE:-768}
EMBODIMENT=${EMBODIMENT:-paddle}
RATIOS=${RATIOS:-0.5,1.0,2.0,3.0}
OUT=${OUT:-outputs/paddle_strike/demo_${EMBODIMENT}_${IMAGE_SIZE}}

# Call the env's interpreter by absolute path rather than `source ~/.bashrc; conda activate`. Two
# reasons, both hit on the first two submissions: the cluster's /etc/bashrc reads an unset
# BASHRCSOURCED so sourcing it under `set -u` kills the job immediately, and a non-interactive batch
# shell has no `conda` function on PATH at all, so `conda activate` fails and `python` resolves to
# nothing. The absolute path has neither failure mode.
PY=python
set -eo pipefail

export PYTHONPATH=.
export MUJOCO_GL=egl

"$PY" -u experiments/threads/paddle-robotics/08_figures/demo_paddle_strike.py \
    --output_dir "$OUT" \
    --embodiment "$EMBODIMENT" \
    --ratios "$RATIOS" \
    --image_size "$IMAGE_SIZE"

echo "wrote $OUT"
