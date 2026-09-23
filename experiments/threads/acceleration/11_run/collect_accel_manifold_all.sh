#!/bin/bash
#SBATCH --job-name=amanall_collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/amanall_collect_%j.out
#SBATCH --error=logs/amanall_collect_%j.err
#
# Merge the manifold-steering shards WITH the 2026-08-26 spline shards and emit one leakage-free table.
# Merging is legitimate and desirable: both runs decode the SAME 100 test scenes x 7 pairs with the same
# checkpoint and the same tracker, so per-pair rows union into a single paired comparison -- which is
# what lets manifold arms be bootstrapped against spline_K8 on identical rows.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
ANA=outputs/analysis/moving_ball_accel2d_mixed

# NOTE ON MERGING PARTIAL WAVES: calibrate_spline_gain.py scores every arm on the COMMON rows where
# every arm tracked, so folding in a wave whose shards did not all finish drops the rows that wave is
# missing -- sample size, not validity. Check n_usable_rows in the output before reading the table.
SUMS=$(ls "$ANA"/steer_mask_ap[0-9]/steer2d_summary.json \
          "$ANA"/steer_man_ap[0-9]/steer2d_summary.json \
          "$ANA"/steer_spline_ap[0-9]/steer2d_summary.json \
          "$ANA"/steer_spline_rich[0-9]/steer2d_summary.json 2>/dev/null)
if [ -z "$SUMS" ]; then echo "[collect] no shard summaries under $ANA"; exit 1; fi
echo "[collect] merging:"; for f in $SUMS; do echo "    $f"; done

for BASE_ARM in spline_K8_s2.5 full_delta_s1; do
    TAG=$(echo "$BASE_ARM" | tr -d '._' )
    echo "===== paired against $BASE_ARM ====="
    python -u experiments/threads/acceleration/04_operators/calibrate_spline_gain.py --summaries $SUMS --val_frac 0.5 \
        --baseline "$BASE_ARM" --out "$ANA/calib_manifold_all_vs$TAG.json" || true
done
echo "[collect] done -> $ANA/calib_manifold_all_vs*.json"
