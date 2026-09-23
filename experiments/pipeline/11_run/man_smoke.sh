#!/bin/bash
#SBATCH --job-name=aman_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=logs/aman_smoke_%j.out
#SBATCH --error=logs/aman_smoke_%j.err
# End-to-end plumbing smoke for the manifold-steering arms: tiny 4-layer manifold fit, then a 2-scene
# decode that exercises every new arm (man/manlin/mandense, manK, mansp, oprojU, oman).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
LAT=outputs/latents/moving_ball_scene_accel2d_mixed
BASE=.
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
SMOKE=${SMOKE_DIR:-outputs/analysis/moving_ball_accel2d_mixed/man_smoke}

set -euo pipefail
python -u experiments/threads/acceleration/04_operators/fit_accel_manifold.py \
    --train_dir "$LAT/train/vjepa2_large" --test_dir "$LAT/test/vjepa2_large" \
    --layers 6,12,18,23 --k_pca 32 --n_centers 32 --ridge 1e-3 \
    --max_scenes 24 --output_dir "$SMOKE/manifold"

python -u experiments/threads/acceleration/05_steering/steer_accel_spline.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir "$LAT/test/vjepa2_large" \
    --spline_dir "$ANA/spline" \
    --manifold_dir "$SMOKE/manifold" \
    --bigU_dir "$ANA/subspace_bigU" --bigU_ks 8,32,128 \
    --checkpoint "$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt" \
    --output_dir "$SMOKE/steer" \
    --pairs all --num_scenes 2 --knots 3,8 --gains 1.0,2.5 \
    --oracle_gains 1.0,2.0 --oracle_arms full_delta,prof_full \
    --manifold_knots 3 --manifold_blend 8 \
    --family_scenes 0 --traj_scenes 0 --device cuda
echo "[aman_smoke] done"
