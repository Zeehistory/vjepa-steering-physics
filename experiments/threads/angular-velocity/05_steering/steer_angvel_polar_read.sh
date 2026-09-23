#!/bin/bash
#SBATCH --job-name=angvel_steer_polar
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=01:30:00
#SBATCH --output=logs/angvel_steer_polar_%j.out
#SBATCH --error=logs/angvel_steer_polar_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
BIG=$BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large
LAYER=${LAYER:-12}
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_polar_read.py \
  --config configs/train/moving_ball_scene_angvel_decoder.yaml \
  --dir $BIG --layer $LAYER --n_r 8 --n_phi 16 --r_lo 1.5 --r_hi 7.5 \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/diag_polar/steer_read_L${LAYER}.json
echo "[angvel_steer_polar] exit=$?"
