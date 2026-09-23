#!/bin/bash
#SBATCH --job-name=angvel_rot
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/angvel_rot_%j.out
#SBATCH --error=logs/angvel_rot_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CKPT=$BASE/outputs/runs/moving_ball_scene_angvel2d_big_decoder_orient/checkpoints/step_7000.pt
TEST=$BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large
OUT=$BASE/outputs/analysis/moving_ball_angvel2d/polar
run () {
  echo "############### VARIANT=$1 ###############"
  python -u experiments/threads/angular-velocity/05_steering/steer_angvel_rot.py \
    --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
    --test_dir $TEST --checkpoint $CKPT \
    --n_test_scenes ${NTEST:-30} --variant "$1" \
    --scales ${SCALES:-0,0.5,0.75,1.0,1.25,1.5,2.0} \
    --r_lo ${RLO:-2.0} --r_hi ${RHI:-8.0} \
    --out $OUT/rot_$1.json
}
run full
run annulus
echo "[angvel_rot] exit=$?"
