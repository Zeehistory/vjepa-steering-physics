#!/bin/bash
#SBATCH --job-name=angvel_polar
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/angvel_polar_%j.out
#SBATCH --error=logs/angvel_polar_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_polar.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/step_7000.pt \
  --n_train_scenes ${NTRAIN:-100} --n_test_scenes ${NTEST:-50} \
  --k_u ${KU:-8} --ridge ${RIDGE:-1.0} \
  --n_r ${NR:-8} --n_phi ${NPHI:-24} --r_lo ${RLO:-0.5} --r_hi ${RHI:-7.5} \
  --gains ${GAINS:-1,2,3,4,6} \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/polar/${TAG:-polar_cmdU8}.json
echo "[angvel_polar] exit=$?"
