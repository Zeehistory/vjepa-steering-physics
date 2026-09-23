#!/bin/bash
#SBATCH --job-name=ridge_sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ridge_sweep_%j.out
#SBATCH --error=logs/ridge_sweep_%j.err

# Re-solve a saved operator under a scaled ridge and measure its reachable-subspace ceiling.
#
# CPU-only, and it must be Slurm: the cond512 operator's XtY is (1563 x 2097152) float32 = 13 GB per
# quantity, and a login-node read of it gets OOM-killed (observed). 160G covers both quantities plus the
# 4.3 GB conditioning projection with room for the float64 Gram chunks.
#
# No refit happens here. The fit persisted XtX/XtY and ridge enters only at solve(), so a different
# penalty is an EXACT refit at zero fitting cost -- which is the only reason a dense lambda sweep is
# affordable at all against the 8 hours the cond512 fit itself took.

cd .
PY=python
export PYTHONPATH=.
export MUJOCO_GL=disable
set -eo pipefail

LAT=outputs/latents/spin_ball3d
ANA=${ANA:-outputs/analysis/spin_ball3d}
OPS=${OPS:?set OPS to an operators dir, e.g. $ANA/operators_cond512}
# The layer is part of the identity of the output: one operators dir can hold several layers, and
# a tag without it makes three per-layer sweeps overwrite one another.
TAG=$(basename "$OPS")_L${LAYERS:-18}

"$PY" -u experiments/threads/restitution-spin/07_eval/resolve_ridge_sweep.py \
    --operators_dir "$OPS" \
    --test_dir "$LAT/test/vjepa2_large" \
    --layers "${LAYERS:-18}" \
    --num_scenes "${NUM_SCENES:-48}" \
    --out "$ANA/ridge_sweep_${TAG}.json"

echo "[ridge_sweep] done -> $ANA/ridge_sweep_${TAG}.json"
