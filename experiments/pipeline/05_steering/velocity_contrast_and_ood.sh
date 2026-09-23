#!/bin/bash
#SBATCH --job-name=vel_ood
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/vel_ood_%j.out
#SBATCH --error=logs/vel_ood_%j.err

# Two questions about the velocity steer, both answered by replaying saved operators:
#
#  (1) MAX-CONTRAST FILMSTRIP. The published strips show rank 0 -> rank 7, and because the
#      generators permute speed against heading that pair is an arbitrary MAGNITUDE pair -- rb3d
#      scene00000 goes 0.018 -> 0.019, a 9% speed change dressed up as a 39 degree heading change.
#      Re-dump with --pair_mode max_mag_ratio, which shows the pair whose speeds actually separate
#      (~2x is available inside the existing cache). Illustrative pair, NOT the scored pair.
#
#  (2) OUT-OF-DISTRIBUTION COMMANDS. Sweep the commanded speed from 0.25x below the training band
#      to 3x above it and measure what comes out, in decoded pixels AND off the latent.

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

module purge
module load miniconda
conda activate vjepa-physics-decoder
export PYTHONPATH=.
export MUJOCO_GL=egl

FS=outputs/filmstrip
RUNS=outputs/runs
AN=outputs/analysis

# name | latent dir | analysis dir | config | checkpoint run | gain | band_lo band_hi
SPECS=(
"v2d_mixed|moving_ball_scene_v2d_mixed|moving_ball_v2d_mixed|moving_ball_scene_decoder|moving_ball_scene_v2d_mixed_decoder_fp|2.5|0.012 0.024"
"rb3d|rolling_ball3d|rolling_ball3d|rolling_ball3d_decoder|rolling_ball3d_decoder_fp_seed1|2.0|0.010 0.022"
)

FAILED=""
for spec in "${SPECS[@]}"; do
    IFS='|' read -r NAME DSET ANDIR CFG RUN GAIN BAND <<< "$spec"

    echo ""
    echo "=========== $NAME : max-contrast re-dump (gain=$GAIN) ==========="
    python experiments/pipeline/08_figures/dump_filmstrip.py \
        --config "configs/train/${CFG}.yaml" \
        --test_dir "$FS/latents/$DSET/test/vjepa2_large" \
        --artifacts_dir "$AN/$ANDIR/subspace" \
        --checkpoint "$RUNS/$RUN/checkpoints/last.pt" \
        --output_dir "$FS/frames/${NAME}_maxmag" \
        --quantity velocity --gain "$GAIN" --num_scenes 12 \
        --pair_mode max_mag_ratio --device cuda || FAILED="$FAILED ${NAME}:dump"

    echo ""
    echo "=========== $NAME : OOD command sweep (band=$BAND) ==========="
    python experiments/pipeline/05_steering/steer_velocity_ood.py \
        --config "configs/train/${CFG}.yaml" \
        --test_dir "$FS/latents/$DSET/test/vjepa2_large" \
        --artifacts_dir "$AN/$ANDIR/subspace" \
        --checkpoint "$RUNS/$RUN/checkpoints/last.pt" \
        --output "$AN/$ANDIR/steer_last/ood_speed_sweep.json" \
        --gain "$GAIN" --num_scenes 12 --band $BAND --device cuda || FAILED="$FAILED ${NAME}:ood"
done

echo ""
if [ -n "$FAILED" ]; then echo "[vel_ood] FAILED:$FAILED"; exit 1; fi
echo "[vel_ood] done"
