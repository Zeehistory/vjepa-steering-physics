#!/bin/bash
#SBATCH --job-name=spd_axis_a
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=logs/spd_axis_a_%j.out
#SBATCH --error=logs/spd_axis_a_%j.err
# Acceleration analogue of refit_speed_axis.sh: supervised |a| axes appended to U8, standardized cmd
# operator refit, 100-scene decoded steer with plain + split gains.  SPEC=accel2d | amix | rb3d_accel
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
export MUJOCO_GL=egl
SC=.
BASE=.
SPEC=${SPEC:?accel2d|amix|rb3d_accel}
CFG=configs/train/moving_ball_scene_decoder.yaml
case "$SPEC" in
  accel2d)    DSET=moving_ball_scene_accel2d;       ART=$BASE/outputs/analysis/moving_ball_accel2d/subspace
              TR=$SC/outputs/latents/$DSET/train/vjepa2_large; TE=$BASE/outputs/latents/$DSET/test/vjepa2_large
              CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_decoder_fp/checkpoints/last.pt
              OUT=$BASE/outputs/analysis/moving_ball_accel2d/steer_refit_std_spd ;;
  amix)       DSET=moving_ball_scene_accel2d_mixed; ART=$BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace
              TR=$SC/outputs/latents/$DSET/train/vjepa2_large; TE=$SC/outputs/latents/$DSET/test/vjepa2_large
              CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
              OUT=$BASE/outputs/analysis/moving_ball_accel2d_mixed/steer_refit_std_spd ;;
  rb3d_accel) D=$SC/../vjepa_sweep/decoded/rb3d_accel/vjepa2_large; ART=$D/artifacts
              TR=$D/latents/train; TE=$D/latents/test; CKPT=$D/decoder/checkpoints/last.pt
              CFG=configs/train/rolling_ball3d_accel_decoder.yaml
              OUT=$BASE/outputs/analysis/size_sweep_decoded/rb3d_accel/vjepa2_large/steer_refit_std_spd ;;
  *) echo "bad SPEC=$SPEC"; exit 1 ;;
esac
echo "[spd-a] SPEC=$SPEC train=$TR test=$TE art=$ART"
[ -f "$ART/global_basis_spd_L23.npy" ] && echo "[spd-a] axes exist, skip" || \
python -u experiments/threads/velocity/04_operators/speed_axis.py --quantity accel \
    --train_dir "$TR" --layers 6,12,18,23 --artifacts_dir "$ART" --ku 8 || exit 1
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
    --train_dir "$TR" --test_dir "$TE" --layers 6,12,18,23 --artifacts_dir "$ART" \
    --ridge 1.0 --ku 0 --basis_tag spd --quantity accel --standardize || exit 1
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py \
    --config $CFG --test_dir "$TE" --artifacts_dir "$ART" --checkpoint "$CKPT" --output_dir "$OUT" \
    --ks 2,4,8,11 --num_scenes 100 --cmd_scales "0.5,1.0,1.5,2.0,2.5,3.0,4.0" --spd_scales "1.5,2.0,3.0" \
    --cmd_ku 0 --basis_tag spd --cmd_std --device cuda || exit 1
python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py \
    --summary "$OUT/steer2d_summary.json" --val_frac 0.5 --out "$OUT/calib_cmd_gain.json"
echo "[spd-a] DONE $SPEC -> $OUT"
