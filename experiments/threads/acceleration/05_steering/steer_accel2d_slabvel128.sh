#!/bin/bash
#SBATCH --job-name=amix_slabvel128
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/amix_slabvel128_%j.out
#SBATCH --error=logs/amix_slabvel128_%j.err
# C retry: decode the KU=128 slabvel operator with a wider gain grid (KU=32 saturated at gain 3.0).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_decoder.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py --config $CFG --test_dir $TEST --artifacts_dir $ANA/subspace \
    --checkpoint $CKPT --output_dir $ANA/steer_slabvel128 --features slabvel --ks 2,4,8,16 \
    --num_scenes 100 --cmd_scales 1.0,2.0,3.0,4.0,5.0 --device cuda
SUM=$ANA/steer_slabvel128/steer2d_summary.json
[ -f "$SUM" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
    --out $ANA/steer_slabvel128/calib_cmd_gain.json
echo "[amix_slabvel128] DONE"
