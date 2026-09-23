#!/bin/bash
#SBATCH --job-name=angvel_fourier
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/angvel_fourier_%j.out
#SBATCH --error=logs/angvel_fourier_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CKPT=$BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/step_7000.pt
TRAIN_DIR=${TRAIN_DIR:-$BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large}
TEST_DIR=${TEST_DIR:-$BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large}
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_fourier.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --train_dir $TRAIN_DIR \
  --test_dir  $TEST_DIR \
  ${GATE_ONLY:+--gate_only} ${CKPT:+--checkpoint $CKPT} \
  --n_train_scenes ${NTRAIN:-125} --n_test_scenes ${NTEST:-30} \
  --order ${ORDER:-4} --ridge ${RIDGE:-10} --gains ${GAINS:-0.5,1,1.5,2,3,4} \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/polar/fourier_${TAG:-gate}.json
echo "[angvel_fourier] exit=$?"
