#!/bin/bash
#SBATCH --job-name=amix_canon
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/accelmix_canon_%j.out
#SBATCH --error=logs/accelmix_canon_%j.err
# Position-canonicalized command-only accel operator (canonicalization push). Self-contained: builds
# U_canon from start-centred Delta H then fits command -> U_canon coords. Targets tiny KU coords (NOT
# full-D) BUT the pass-0 PCA buffer (~25G) + LatentDataset caching OOM'd at 64G, so run at 256G. `day`
# Writes global_basis_canon_L*.npy + cmd_Wu_canon_L*.npy. Consumed by
# steer_accel2d.py --features canon. KU env sets canon rank (default 16). No decoder/subspace dep.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KU=${KU:-16}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_canon.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --ridge 1.0 --ku $KU
echo "[amix_canon] exit=$? (KU=$KU)"
