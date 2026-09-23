#!/bin/bash
#SBATCH --job-name=angvel_diag_big
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/angvel_diag_big_%j.out
#SBATCH --error=logs/angvel_diag_big_%j.err
# GATE: does the BIG-OBJECT decoder render rotation faithfully? Decode the TRUE H_b latent for held-out big
# scenes and correlate honest measured_angvel vs GT omega. Small-object decoder failed this (rho ~-0.33 = a
# smear). If the big bar renders crisply, rho should be strongly POSITIVE -> the rendering wall is down and
# the TTO steer has an honest target.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/07_eval/diag_angvel_decode.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --test_dir $BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/last.pt \
  --num_scenes 16 --device cuda
echo "[angvel_diag_big] exit=$?"
