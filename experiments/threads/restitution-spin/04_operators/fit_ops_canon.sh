#!/bin/bash
#SBATCH --job-name=spin_canon
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/spin_canon_%j.out
#SBATCH --error=logs/spin_canon_%j.err

# Fit the POSITION-CANONICALIZED operators and measure them, in one job.
#
# Why this exists as a separate script and a separate output directory: the global fit it replaces is a
# real measurement (gain 0.156, alignment 0.39) and the A/B against it is the evidence that
# canonicalization is what fixed things. Overwriting `operators/` would destroy the control. So canonical
# operators go to `operators_canon/` and both stay on disk.
#
# The chained latent_crosstalk run is the decisive number: it reads `canon` out of operators_meta.json
# and rolls predictions back automatically, and its gt_vel / gt_spin controls must still read (1,0) and
# (0,1). If canonicalization were wired up wrong those controls are unaffected (they are real clips, not
# operator output), so they cannot mask a mistake -- but a botched roll-back would show as gain ~0.
#

cd .
PY=python
export PYTHONPATH=.
export MUJOCO_GL=disable
set -eo pipefail

LAT=outputs/latents/spin_ball3d
ANA=${ANA:-outputs/analysis/spin_ball3d}
LAYERS=${LAYERS:-18}
OPS=${OPS:-$ANA/operators_canon}

"$PY" -u experiments/threads/restitution-spin/04_operators/fit_spin_operators.py \
    --train_dir "$LAT/train/vjepa2_large" \
    --output_dir "$OPS" \
    --layers "$LAYERS" \
    --canon "${CANON:-1}" \
    --cond_dim "${COND_DIM:-0}" \
    ${COND_BASIS:+--cond_basis "$COND_BASIS"} \
    --ridge "${RIDGE:-1.0}" \
    --num_scenes "${NUM_SCENES:-0}" \
    --max_cached_shards "${MAX_SHARDS:-1}"

test -s "$OPS/operator_vel.npz" || { echo "FAIL: operators missing"; exit 1; }

"$PY" -u experiments/threads/restitution-spin/06_probes/latent_crosstalk.py \
    --test_dir "$LAT/test/vjepa2_large" \
    --operators_dir "$OPS" \
    --layers "$LAYERS" \
    --num_scenes "${XTALK_SCENES:-48}" \
    --out "$ANA/latent_crosstalk_${TAG:-canon}_L${LAYERS//,/_}.json"

echo "[canon] done -> $OPS"
