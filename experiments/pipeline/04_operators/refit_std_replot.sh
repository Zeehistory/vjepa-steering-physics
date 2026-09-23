#!/bin/bash
#SBATCH --job-name=refit_replot
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=logs/refit_replot_%j.out
#SBATCH --error=logs/refit_replot_%j.err
# 2026-09-15 replot sweep: baseline + STANDARDIZED command-operator refit and 100-scene decoded steer for
# the datasets the 09-04 magnitude fix never covered. One SPEC per job:
#   SPEC=v2d        velocity, single appearance     (train latents re-extracted to scratch)
#   SPEC=accel2d    acceleration, single appearance (train latents re-extracted to scratch)
#   SPEC=amix       acceleration, mixed appearance  (train+test re-extracted to scratch)
# The baseline arm must reproduce the published angle ladder; the std arm is the magnitude result.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
export MUJOCO_GL=egl

SC=.
BASE=.
SPEC=${SPEC:?set SPEC=v2d|accel2d|amix}
case "$SPEC" in
  v2d)     Q=velocity; DSET=moving_ball_scene_v2d;         ANDIR=moving_ball_v2d;         RUN=moving_ball_scene_v2d_decoder_fp;         KU=8
           TE=$BASE/outputs/latents/$DSET/test/vjepa2_large ;;
  accel2d) Q=accel;    DSET=moving_ball_scene_accel2d;     ANDIR=moving_ball_accel2d;     RUN=moving_ball_scene_accel2d_decoder_fp;     KU=8
           TE=$BASE/outputs/latents/$DSET/test/vjepa2_large ;;
  amix)    Q=accel;    DSET=moving_ball_scene_accel2d_mixed; ANDIR=moving_ball_accel2d_mixed; RUN=moving_ball_scene_accel2d_mixed_decoder_fp; KU=8
           TE=$SC/outputs/latents/$DSET/test/vjepa2_large ;;
  *) echo "bad SPEC=$SPEC"; exit 1 ;;
esac
TR=$SC/outputs/latents/$DSET/train/vjepa2_large
ART=$BASE/outputs/analysis/$ANDIR/subspace
CKPT=$BASE/outputs/runs/$RUN/checkpoints/last.pt
CFG=configs/train/moving_ball_scene_decoder.yaml
echo "[refit_replot] SPEC=$SPEC Q=$Q KU=$KU train=$TR test=$TE"

FAILED=""
for ARM in baseline standardized; do
    TAG=""; FITX=(); STDX=()
    [ "$ARM" = "standardized" ] && { TAG="_std"; FITX=(--standardize); STDX=(--cmd_std); }
    echo ""; echo "=========== $SPEC : fit $ARM ==========="
    if [ "${STEER_ONLY:-0}" = 1 ]; then
        echo "[refit_replot] STEER_ONLY=1: reusing fitted operators in $ART"
    elif [ "$Q" = velocity ]; then
        python -u experiments/pipeline/04_operators/fit_command_operators.py \
            --train_dir "$TR" --test_dir "$TE" --layers 6,12,18,23 --artifacts_dir "$ART" \
            --ridge 1.0 --ku "$KU" --skip_rich "${FITX[@]}" || { FAILED="$FAILED $SPEC:fit:$ARM"; continue; }
    else
        python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
            --train_dir "$TR" --test_dir "$TE" --layers 6,12,18,23 --artifacts_dir "$ART" \
            --ridge 1.0 --ku "$KU" --quantity accel "${FITX[@]}" || { FAILED="$FAILED $SPEC:fit:$ARM"; continue; }
    fi
    echo ""; echo "=========== $SPEC : steer $ARM ==========="
    OUT=$BASE/outputs/analysis/$ANDIR/steer_refit${TAG}
    if [ "$Q" = velocity ]; then
        python -u experiments/threads/velocity/05_steering/steer_velocity2d.py \
            --config $CFG --test_dir "$TE" --artifacts_dir "$ART" --checkpoint "$CKPT" --output_dir "$OUT" \
            --ks 2,4,8,16 --num_scenes 100 --cmd_scales "1.0,1.5,2.0,2.5,3.0" \
            --cmd_ku "$KU" --dir_bins "" "${STDX[@]}" --device cuda || FAILED="$FAILED $SPEC:steer:$ARM"
    else
        python -u experiments/threads/acceleration/05_steering/steer_accel2d.py \
            --config $CFG --test_dir "$TE" --artifacts_dir "$ART" --checkpoint "$CKPT" --output_dir "$OUT" \
            --ks 2,4,8,16 --num_scenes 100 --cmd_scales "0.5,1.0,1.5,2.0,2.5,3.0,4.0" \
            --cmd_ku "$KU" "${STDX[@]}" --device cuda || FAILED="$FAILED $SPEC:steer:$ARM"
    fi
    [ -f "$OUT/steer2d_summary.json" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py \
        --summary "$OUT/steer2d_summary.json" --val_frac 0.5 --out "$OUT/calib_cmd_gain.json"
done

B=$BASE/outputs/analysis/$ANDIR
[ -f $B/steer_refit/steer2d_summary.json ] && [ -f $B/steer_refit_std/steer2d_summary.json ] && \
  python -u experiments/pipeline/07_eval/compare_magnitude_control.py \
    --base $B/steer_refit/steer2d_summary.json --std $B/steer_refit_std/steer2d_summary.json --label "$SPEC"

echo ""
if [ -n "$FAILED" ]; then echo "[refit_replot] FAILED:$FAILED"; exit 1; fi
echo "[refit_replot] done"
