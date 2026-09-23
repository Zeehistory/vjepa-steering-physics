#!/bin/bash
#SBATCH --job-name=a_vanmag
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/a_vanmag_%j.out
#SBATCH --error=logs/a_vanmag_%j.err
# A1: does the VANILLA edit H_a + alpha*(H_b-H_a) install correct accel MAGNITUDE? Latent probe, alpha sweep.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/06_probes/probe_accel_vanilla.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 --alphas 0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0 \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/accel_probe
echo "[a_vanmag] exit=$?"
