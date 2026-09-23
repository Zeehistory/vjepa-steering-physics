#!/bin/bash
#SBATCH --job-name=aspl_gain
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aspl_gain_%j.out
#SBATCH --error=logs/aspl_gain_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
python -u experiments/threads/acceleration/05_steering/steer_accel_spline.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --test_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --spline_dir $ANA/spline \
    --checkpoint $BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt \
    --output_dir $ANA/steer_spline_gainext \
    --knots 1,2,3,8 --gains 3.5,4.0,4.5,5.0,6.0,7.0 \
    --num_scenes 100 --family_scenes 0 --traj_scenes 0 --device cuda
echo "[aspl_gain] exit=$?"
