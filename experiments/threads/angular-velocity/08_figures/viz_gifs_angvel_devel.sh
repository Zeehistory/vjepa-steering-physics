#!/bin/bash
#SBATCH --job-name=viz_angvel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/viz_angvel_%j.out
#SBATCH --error=logs/viz_angvel_%j.err
# Re-run the Fourier-in-orientation angular-velocity steer (the headline command-only result) and dump
# whole decoded clips at the chosen gain, so the rotation can be shown as a before/after GIF.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CKPT=$BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/step_7000.pt
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_fourier.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large \
  --checkpoint $CKPT \
  --n_train_scenes ${NTRAIN:-125} --n_test_scenes ${NTEST:-30} \
  --order 4 --ridge 10 --gains ${GAINS:-0.5,1,1.5,2,3,4} \
  --viz_scenes ${VIZ:-6} --viz_dir ${DUMP:-scratchpad/viz_dumps/angvel} \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/polar/fourier_decode_o4_viz.json
echo "[viz_angvel] exit=$?"
