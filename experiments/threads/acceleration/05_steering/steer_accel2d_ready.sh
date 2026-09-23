#!/bin/bash
#SBATCH --job-name=amix_ready
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/amix_ready_%j.out
#SBATCH --error=logs/amix_ready_%j.err
# GPU decode for the parts that are READY now (no dependency on the bigU re-run):
#   A2  vanilla alpha sweep (pixels)
#   C   slabvel steer (temporal-composition velocity operator) + calibration
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_decoder.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
ART=$ANA/subspace
NS=${NUM_SCENES:-100}

echo "===== A2: vanilla alpha sweep (pixels) ====="
python -u experiments/threads/acceleration/05_steering/steer_accel_vanilla_alpha.py --config $CFG --test_dir $TEST --checkpoint $CKPT \
    --output_dir $ANA/steer_vanilla --alphas 0.5,1.0,1.5,2.0,2.5,3.0 --num_scenes 60 --device cuda

echo "===== C: slabvel (temporal-composition) ====="
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py --config $CFG --test_dir $TEST --artifacts_dir $ART --checkpoint $CKPT \
    --output_dir $ANA/steer_slabvel --features slabvel --ks 2,4,8,16 --num_scenes $NS \
    --cmd_scales 1.0,1.5,2.0,2.5,3.0 --viz_scenes 4 --viz_gain 2.0 --device cuda
SUM=$ANA/steer_slabvel/steer2d_summary.json
[ -f "$SUM" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
    --out $ANA/steer_slabvel/calib_cmd_gain.json
echo "[amix_ready] DONE"
