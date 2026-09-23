#!/bin/bash
#SBATCH --job-name=amix_trans
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/accelmix_trans_%j.out
#SBATCH --error=logs/accelmix_trans_%j.err
# 2nd-order screen: is the accel edit a POSITION-CONDITIONED per-frame TRANSLATION by 1/2*Da*t^2 (the exact
# trajectory divergence, pos0/v0 cancel)? Latent-only, no GPU. Reports held-out recon cos of assembled dH for
# base_global vs disp vs disp_pos. disp_pos >> 0.28 -> build the translation-field decode. 256G: shard cache
# (~160G, no eviction) + a full-D command->dH base map (~0.8G).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/07_eval/accel_translation_screen.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 --ridge 1.0 \
    --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace/translation_screen.json
echo "[amix_trans] exit=$?"
