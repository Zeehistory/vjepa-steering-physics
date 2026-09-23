#!/bin/bash
#SBATCH --job-name=refit_std
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/refit_std_%j.out
#SBATCH --error=logs/refit_std_%j.err

# Refit the command-only velocity operator with a SCALE-FREE ridge, and score it head-to-head
# against a baseline refit on the identical data.
#
# The bug: fit_command_operators.py applied `ridge * I` to raw `command_features`. The bias and
# unit-heading columns have RMS ~1; every speed-carrying column (v_b, v_a, dv, |v_b|, |v_a|) has
# RMS ~0.02. Ridge retention is d_j/(d_j+lambda), so at lambda=1 heading keeps ~98% and speed keeps
# ~2% -- a 47x differential. Result: an operator that steers heading and ignores magnitude
# (corr(|commanded|,|achieved|) = +0.00 on 2-D vs +0.98 for the full_delta oracle).
#
# Both arms are refit here on the SAME re-extracted latents, so the comparison is clean and the
# baseline arm is expected to reproduce the published behaviour.

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
export MUJOCO_GL=egl

SC=.
BASE=.
KU=${KU:-16}

# name | latent dataset | analysis dir | train config | decoder run | steer config | gains
SPECS=(
"v2d_mixed|moving_ball_scene_v2d_mixed|moving_ball_v2d_mixed|moving_ball_scene_decoder|moving_ball_scene_v2d_mixed_decoder_fp"
"rb3d|rolling_ball3d|rolling_ball3d|rolling_ball3d_decoder|rolling_ball3d_decoder_fp_seed1"
)

FAILED=""
for spec in "${SPECS[@]}"; do
    IFS='|' read -r NAME DSET ANDIR CFG RUN <<< "$spec"
    ART=$BASE/outputs/analysis/$ANDIR/subspace
    TR=$SC/outputs/latents/$DSET/train/vjepa2_large
    TE=$SC/outputs/latents/$DSET/test/vjepa2_large

    for ARM in baseline standardized; do
        EXTRA=(); TAG=""
        [ "$ARM" = "standardized" ] && { EXTRA=(--standardize); TAG="_std"; }
        echo ""
        echo "=========== $NAME : fit $ARM (ku=$KU) ==========="
        python -u experiments/pipeline/04_operators/fit_command_operators.py \
            --train_dir "$TR" --test_dir "$TE" \
            --layers 6,12,18,23 --artifacts_dir "$ART" --ridge 1.0 --ku "$KU" --skip_rich \
            "${EXTRA[@]}" || { FAILED="$FAILED $NAME:fit:$ARM"; continue; }

        echo ""
        echo "=========== $NAME : steer $ARM ==========="
        STD_FLAG=(); [ "$ARM" = "standardized" ] && STD_FLAG=(--cmd_std)
        python -u experiments/threads/velocity/05_steering/steer_velocity2d.py \
            --config "configs/train/${CFG}.yaml" \
            --test_dir "$TE" --artifacts_dir "$ART" \
            --checkpoint "$BASE/outputs/runs/$RUN/checkpoints/last.pt" \
            --output_dir "$BASE/outputs/analysis/$ANDIR/steer_refit${TAG}" \
            --ks 2,4,8,16 --num_scenes 100 --cmd_scales "1.0,1.5,2.0,2.5,3.0" \
            --cmd_ku "$KU" --dir_bins "" "${STD_FLAG[@]}" --device cuda \
            || FAILED="$FAILED $NAME:steer:$ARM"

        SUM=$BASE/outputs/analysis/$ANDIR/steer_refit${TAG}/steer2d_summary.json
        [ -f "$SUM" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py \
            --summary "$SUM" --val_frac 0.5 \
            --out $BASE/outputs/analysis/$ANDIR/steer_refit${TAG}/calib_cmd_gain.json
    done
done

echo ""
if [ -n "$FAILED" ]; then echo "[refit_std] FAILED:$FAILED"; exit 1; fi
echo "[refit_std] done"
