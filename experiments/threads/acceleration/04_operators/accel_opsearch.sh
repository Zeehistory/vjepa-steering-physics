#!/bin/bash
#SBATCH --job-name=amix_opsrch
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accelmix_opsearch_%j.out
#SBATCH --error=logs/accelmix_opsearch_%j.err
# Latent-space SEARCH for a better command-only accel operator. Screens base/cmd_prof/v0_prof/probe_ax
# against the accel PROBE (temporal-pool readout) on held-out test -- NO decoder, CPU, minutes. Writes
# opsearch_summary.json + prof_operators.npz (profile-op B matrices + probe Wp) for the eventual decode.
# 256G on `day` (LatentDataset shard caching); no GPU. Reuses existing latents + canon artifacts.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/04_operators/accel_operator_search.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --layers 6,12,18,23 \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/accel_opsearch
echo "[amix_opsearch] exit=$?"
