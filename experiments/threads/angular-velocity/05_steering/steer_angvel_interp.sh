#!/bin/bash
#SBATCH --job-name=angvel_interp
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/angvel_interp_%j.out
#SBATCH --error=logs/angvel_interp_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_interp.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --test_dir $BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/last.pt \
  --num_scenes ${NUM_SCENES:-25} --alphas ${ALPHAS:-0,0.25,0.5,0.75,1.0} \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/interp_controllability.json
echo "[angvel_interp] exit=$?"
