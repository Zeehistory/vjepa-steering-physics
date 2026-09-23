#!/bin/bash
#SBATCH --job-name=act_fit
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/act_fit_%j.out
#SBATCH --error=logs/act_fit_%j.err

# Subspace PCA + command-operator fit on the ACTIVATION (mlp.fc2) caches -- the activation twin of
# _subspace_*.sh followed by _fit_command*.sh, which is what the STEERING and READING sections need
# before they mean anything on this representation. Latent-only, no GPU (bigmem for the global-PCA
# buffers, ~7 GB per layer).
#
#   FAMILY=velocity sbatch experiments/pipeline/11_run/act_fit.sh
#
# Artifacts land in a *_act/subspace dir, never on top of the residual-stream ones.

set -o pipefail
FAMILY=${FAMILY:-"velocity"}
SCR=${SCR:-"."}
BASE=${BASE:-"."}

case "$FAMILY" in
  velocity)
    CACHE="moving_ball_scene_v2d_act"; ART="$BASE/outputs/analysis/moving_ball_v2d_act/subspace"
    SUB_SCRIPT="experiments/threads/velocity/04_operators/velocity_subspace.py"; SUB_ARGS=(--ridge 1.0 --save_k 8 --max_global_pairs 800)
    FIT_SCRIPT="experiments/pipeline/04_operators/fit_command_operators.py"; FIT_ARGS=(--ridge 1.0)
    # velocity's TEST activation cache is the one that lives on project, not scratch
    TEST_ROOT="$BASE" ;;
  angvel)
    CACHE="moving_ball_scene_angvel2d_act"; ART="$BASE/outputs/analysis/moving_ball_angvel2d_act/subspace"
    SUB_SCRIPT="experiments/threads/acceleration/04_operators/accel_subspace.py"; SUB_ARGS=(--quantity angvel --save_k 16)
    FIT_SCRIPT="experiments/threads/acceleration/04_operators/fit_command_operators_accel.py"; FIT_ARGS=(--quantity angvel --ridge 1.0 --ku 16)
    TEST_ROOT="$SCR" ;;
  accel)
    CACHE="moving_ball_scene_accel2d_act"; ART="$BASE/outputs/analysis/moving_ball_accel2d_act/subspace"
    SUB_SCRIPT="experiments/threads/acceleration/04_operators/accel_subspace.py"; SUB_ARGS=(--save_k 16)
    FIT_SCRIPT="experiments/threads/acceleration/04_operators/fit_command_operators_accel.py"; FIT_ARGS=(--ridge 1.0 --ku 16)
    TEST_ROOT="$SCR" ;;
  *) echo "unknown FAMILY=$FAMILY (velocity|angvel|accel)"; exit 1 ;;
esac

TRAIN_DIR="$SCR/outputs/latents/$CACHE/train/vjepa2_large"
TEST_DIR="$TEST_ROOT/outputs/latents/$CACHE/test/vjepa2_large"

module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs "$ART"

for d in "$TRAIN_DIR" "$TEST_DIR"; do
  [ -f "$d/extract_meta.json" ] || { echo "[act_fit] missing or incomplete cache: $d"; exit 2; }
  echo "[act_fit] $d hook_site=$(python -c "import json,sys;print(json.load(open('$d/extract_meta.json'))['config'].get('hook_site','block'))")"
done

# STAGE=ops resumes at the command-operator fit, reusing subspace artifacts already on disk.
# Only safe when $ART/subspace_summary.json is complete -- the operator fit reads the bases, so a
# truncated subspace write would silently poison every operator downstream.
if [ "${STAGE:-all}" = "ops" ]; then
  python -c "import json;json.load(open('$ART/subspace_summary.json'))" \
    || { echo "[act_fit] STAGE=ops but $ART/subspace_summary.json is missing/corrupt"; exit 3; }
  echo "[act_fit] STAGE=ops -- reusing existing subspace artifacts in $ART"
else
echo "[act_fit] $FAMILY subspace -> $ART"
python -u "$SUB_SCRIPT" \
    --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" \
    --layers 6,12,18,23 --output_dir "$ART" "${SUB_ARGS[@]}" || exit $?
fi

echo "[act_fit] $FAMILY command operators -> $ART"
python -u "$FIT_SCRIPT" \
    --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" \
    --layers 6,12,18,23 --artifacts_dir "$ART" "${FIT_ARGS[@]}"
STATUS=$?
echo "[act_fit] done (exit $STATUS): $FAMILY -> $ART"
exit $STATUS
