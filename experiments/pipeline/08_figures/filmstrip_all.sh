#!/bin/bash
#SBATCH --job-name=filmstrip
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/filmstrip_%j.out
#SBATCH --error=logs/filmstrip_%j.err

# Dump Real-vs-Decoded-steered frame stacks for every quantity that has BOTH a trained decoder and a
# fitted command operator. Replays saved artifacts only -- nothing is fit here.
#
# Gains are the ones each quantity's steer_*.py selected on its disjoint validation half:
#   v2d_mixed 2.5 | rb3d 2.0 | accel2d_mixed 2.0 | angvel2d 2.0
# GRAVITY IS THE EXCEPTION: it has cmd_Wu artifacts but no calibrated linear steer was ever run
# (outputs/analysis/moving_ball_gravity/ holds only subspace/ + steer_decopt/), so 2.0 below is an
# UNCALIBRATED default carried over from the other quantities. Label any gravity panel accordingly.

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
NS=${NUM_SCENES:-6}

# name | latent dataset dir | analysis dir | config | checkpoint run | quantity | gain
SPECS=(
"v2d_mixed|moving_ball_scene_v2d_mixed|moving_ball_v2d_mixed|moving_ball_scene_decoder|moving_ball_scene_v2d_mixed_decoder_fp|velocity|2.5"
"rb3d|rolling_ball3d|rolling_ball3d|rolling_ball3d_decoder|rolling_ball3d_decoder_fp_seed1|velocity|2.0"
"accel2d_mixed|moving_ball_scene_accel2d_mixed|moving_ball_accel2d_mixed|moving_ball_scene_decoder|moving_ball_scene_accel2d_mixed_decoder_fp|accel|2.0"
"gravity|moving_ball_scene_gravity|moving_ball_gravity|moving_ball_scene_decoder|moving_ball_scene_gravity_decoder_fp|gravity|2.0"
"angvel2d|moving_ball_scene_angvel2d|moving_ball_angvel2d|moving_ball_scene_angvel_decoder|moving_ball_scene_angvel2d_decoder_fp|angvel|2.0"
)

FAILED=""
for spec in "${SPECS[@]}"; do
    IFS='|' read -r NAME DSET ANDIR CFG RUN QUANT GAIN <<< "$spec"
    echo ""
    echo "=================== $NAME (quantity=$QUANT gain=$GAIN) ==================="
    python experiments/pipeline/08_figures/dump_filmstrip.py \
        --config "configs/train/${CFG}.yaml" \
        --test_dir "$FS/latents/$DSET/test/vjepa2_large" \
        --artifacts_dir "$AN/$ANDIR/subspace" \
        --checkpoint "$RUNS/$RUN/checkpoints/last.pt" \
        --output_dir "$FS/frames/$NAME" \
        --quantity "$QUANT" --gain "$GAIN" --num_scenes "$NS" --device cuda
    if [ $? -ne 0 ]; then FAILED="$FAILED $NAME"; fi
done

echo ""
if [ -n "$FAILED" ]; then echo "[filmstrip] FAILED:$FAILED"; exit 1; fi
echo "[filmstrip] all quantities done -> $FS/frames"
