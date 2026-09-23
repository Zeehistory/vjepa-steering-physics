#!/bin/bash
#SBATCH --job-name=amix_prof
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/accelmix_prof_%j.out
#SBATCH --error=logs/accelmix_prof_%j.err
# DECODE the temporal-PROFILE accel operators found by accel_operator_search.py, plus canon as a control.
# The decisive test: a probe-axis / broadcast-profile edit reads as perfect accel in the latent -- does it
# also DECODE to accelerated pixels? Runs each --features op through steer_accel2d.py (reduced ks to cut
# decode cost; the subspace/random controls are already established) + leakage-free gain calibration.
# One b200 allocation, features looped sequentially. AFTER: read steer_${op}/calib_cmd_gain.json.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
OPS=$BASE/outputs/analysis/moving_ball_accel2d_mixed/accel_opsearch
ART=$BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
for FEAT in probe_ax prof hybrid canon; do
    echo "===== DECODE features=$FEAT ====="
    python -u experiments/threads/acceleration/05_steering/steer_accel2d.py \
        --config configs/train/moving_ball_scene_decoder.yaml \
        --test_dir $TEST --artifacts_dir $ART --opsearch_dir $OPS --checkpoint "$CKPT" \
        --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_${FEAT} \
        --ks 8 --num_scenes ${NUM_SCENES:-100} \
        --cmd_scales "${CMD_SCALES:-0.5,1.0,1.5,2.0,2.5,3.0,4.0,6.0}" \
        --features $FEAT --viz_scenes 4 --viz_gain 1.5 --device cuda
    SUM=$BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_${FEAT}/steer2d_summary.json
    [ -f "$SUM" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
        --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_${FEAT}/calib_cmd_gain.json
    echo "[amix_prof] done features=$FEAT"
done
echo "[amix_prof] ALL DONE"
