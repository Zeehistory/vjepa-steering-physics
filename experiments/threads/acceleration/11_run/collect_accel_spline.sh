#!/bin/bash
#SBATCH --job-name=aspl_collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/aspl_collect_%j.out
#SBATCH --error=logs/aspl_collect_%j.err
#
# Merge every all-pairs decode shard and produce the leakage-free table (2026-08-26).
#
# Merging is what makes the shards legitimate: calibrate_spline_gain.py splits by SCENE, and each shard
# holds a disjoint scene range, so the union is exactly the n=700 protocol run in four pieces. Two
# tables are written, differing only in the arm each family is paired against:
#
#   ..._vsK8    baseline spline_K8 = the unconstrained per-token profile == the `cmd_prof` operator that
#               the leakage-free record (14.03deg on the old n=100 protocol) actually belongs to. This
#               is the honest "did the spline basis beat the standing command-only operator" test.
#   ..._vsK1    baseline spline_K1 = the classical single global steering vector, i.e. how much the
#               temporal shape is worth at all.
#
# Read the CI, not the mean: at n=100 the K3-vs-K8 gap was -0.27deg [-1.35, +0.79], i.e. nothing.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
ANA=outputs/analysis/moving_ball_accel2d_mixed

SUMS=$(ls "$ANA"/steer_spline_ap[0-9]/steer2d_summary.json \
          "$ANA"/steer_spline_rich[0-9]/steer2d_summary.json 2>/dev/null)
if [ -z "$SUMS" ]; then echo "[collect] no shard summaries found under $ANA"; exit 1; fi
echo "[collect] merging:"; for f in $SUMS; do echo "    $f"; done

for BASE_ARM in spline_K8_s2.5 spline_K1_s4.5; do
    TAG=$(echo "$BASE_ARM" | cut -d_ -f2)
    echo "===== paired against $BASE_ARM ====="
    python -u experiments/threads/acceleration/04_operators/calibrate_spline_gain.py --summaries $SUMS --val_frac 0.5 \
        --baseline "$BASE_ARM" --out "$ANA/calib_allpairs_vs$TAG.json" || true
done
echo "[collect] done -> $ANA/calib_allpairs_vs*.json"
