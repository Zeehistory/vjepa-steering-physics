#!/bin/bash
#SBATCH --job-name=amix_trajc
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/accelmix_trajcanon_%j.out
#SBATCH --error=logs/accelmix_trajcanon_%j.err
# v0-aware TRAJECTORY-canonicalized command-only accel operator. Rolls EACH latent frame to centre the
# ball's actual path (not just frame 0 like canon), aligning the late-frame curvature footprint across
# scenes where the v0 ramp otherwise scatters it. Builds U_trajcanon + cmd->U map. day/128G (pass-0 PCA
# buffer ~25G + shard cache); no GPU. KU env sets rank (default 16). Consumed by steer_accel2d --features
# trajcanon. Reuses existing latents; no re-encode/decode dep. 256G: LatentDataset._shard_cache holds all
# 32 train shards (4/24 layers) ~160G with no eviction — 128G OOM'd (MaxRSS 134G, job 16963478); matches canon.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KU=${KU:-16}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_trajcanon.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --ridge 1.0 --ku $KU
echo "[amix_trajcanon] exit=$? (KU=$KU)"
