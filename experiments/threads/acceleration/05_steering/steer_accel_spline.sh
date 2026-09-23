#!/bin/bash
#SBATCH --job-name=aspl_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/aspl_steer_%j.out
#SBATCH --error=logs/aspl_steer_%j.err
# GPU decode + pixel tracking for spline-in-time acceleration steering (Protocols A and B).
# Loads only the small TEST cache (800 clips) + the accel decoder, so gpu_devel fits and schedules fast.
# ALWAYS submit via the queue-aware picker:
#   sbatch -p <partition> experiments/threads/acceleration/05_steering/steer_accel_spline.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_decoder.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
OUT=$ANA/steer_spline

python -u experiments/threads/acceleration/05_steering/steer_accel_spline.py \
    --config $CFG --test_dir $TEST --spline_dir $ANA/spline --checkpoint $CKPT \
    --output_dir $OUT --knots 1,2,3,4,6,8 --gains 1.0,1.5,2.0,2.5,3.0 \
    --num_scenes ${NUM_SCENES:-100} --family_scenes ${FAMILY_SCENES:-40} --traj_scenes 6 \
    --device cuda
STATUS=$?

# Leakage-free gain pick: choose on half the SCENES, report on the disjoint half.
# This used to call calibrate_cmd_gain.py, which recognises only cmd_U8_s{gain} arm names and so
# hard-exited on every spline run ("no cmd_U8_s{gain} methods found in summary" --
# logs/aspl_steer_19850336.err). The published spline table came from a separate manual invocation of
# the right script; this hook never produced anything.
if [ -f "$OUT/steer2d_summary.json" ]; then
    python -u experiments/threads/acceleration/04_operators/calibrate_spline_gain.py --summaries "$OUT/steer2d_summary.json" --val_frac 0.5 \
        --out "$OUT/calib_spline_gain.json"
fi
echo "[aspl_steer] exit=$STATUS out=$OUT"
exit $STATUS
