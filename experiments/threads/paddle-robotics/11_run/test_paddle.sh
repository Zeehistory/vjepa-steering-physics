#!/bin/bash
#SBATCH --job-name=pstrike_tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/pstrike_tests_%j.out
#SBATCH --error=logs/pstrike_tests_%j.err

# The gated test suite for the scene, on a GPU node so the RENDERING tests actually run instead of
# skipping. Two of them are the ones that matter for the aesthetic work:
#   test_ball_is_the_only_dark_object      -- the palette's hard constraint (ball is sole sub-0.5 object)
#   test_tracker_agrees_with_analytic_projection -- the darkness tracker still reads the right centroid
# Both silently SKIP without a GL context, so running them on the login node proves nothing.

cd .
PY=python
set -eo pipefail
export PYTHONPATH=.
export MUJOCO_GL=egl

"$PY" -m pytest tests/test_paddle_strike.py -v -rs
