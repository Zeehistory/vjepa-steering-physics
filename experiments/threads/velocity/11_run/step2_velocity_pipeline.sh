#!/bin/bash
#
# USAGE: source experiments/threads/velocity/11_run/step2_velocity_pipeline.sh   (from repo root, env active)
#
# This submits all jobs with the correct dependencies so you can kick everything off in one go.
# Requires: conda env `vjepa-physics-decoder` active, repo cloned to $BASE_DIR.
#
# Flow:
#   0. Demos (CPU, ~10 min)          → inspect outputs/demos/moving_ball/ before anything else
#   1. Extract latents x3 (GPU, ~1h each, all 3 in parallel after demos look good)
#   2. Velocity probe  (CPU, ~1h)    → needs extract_velocity done
#   3. Occlusion probe (CPU, ~30m)   → needs extract_occlusion done
#   4. Equiv probe     (CPU, ~30m)   → needs extract_equivariance done
#   5. Steer velocity  (GPU, ~30m)   → needs extract_velocity + trained decoder checkpoint
#
# NOTE: step 5 requires a trained decoder. Train one first with:
#   LATENT_DIR=.../moving_ball_velocity/vjepa2_large \
#   sbatch experiments/pipeline/03_train/train_decoder.sh    (update that script's config for 128x128/32f)

set -e
BASE_DIR=${BASE_DIR:-"."}

echo "=== Step 2 velocity pipeline submission ==="
echo "BASE_DIR=$BASE_DIR"

# -- 0. Demos ---------------------------------------------------------------------------------
JID_DEMOS=$(sbatch --parsable experiments/pipeline/08_figures/ball_demos.sh)
echo "0. demos         -> job $JID_DEMOS"

# -- 1. Latent extraction (parallel, depend on demos completing so you can cancel if demos look wrong)
JID_EXT_VEL=$(sbatch --parsable \
    --dependency=afterok:$JID_DEMOS \
    --export=ALL,DATASET=moving_ball_velocity,BASE_DIR=$BASE_DIR \
    experiments/pipeline/02_encode/extract_ball.sh)
echo "1a. extract vel  -> job $JID_EXT_VEL"

JID_EXT_OCC=$(sbatch --parsable \
    --dependency=afterok:$JID_DEMOS \
    --export=ALL,DATASET=moving_ball_occlusion,BASE_DIR=$BASE_DIR \
    experiments/pipeline/02_encode/extract_ball.sh)
echo "1b. extract occ  -> job $JID_EXT_OCC"

JID_EXT_EQV=$(sbatch --parsable \
    --dependency=afterok:$JID_DEMOS \
    --export=ALL,DATASET=moving_ball_equivariance,BASE_DIR=$BASE_DIR \
    experiments/pipeline/02_encode/extract_ball.sh)
echo "1c. extract eqv  -> job $JID_EXT_EQV"

# -- 2. Velocity probe (depends on velocity extraction)
JID_VPROBE=$(sbatch --parsable \
    --dependency=afterok:$JID_EXT_VEL \
    --export=ALL,BASE_DIR=$BASE_DIR \
    experiments/threads/velocity/06_probes/probe_velocity.sh)
echo "2.  vel probe    -> job $JID_VPROBE"

# -- 3. Occlusion probe (depends on occlusion extraction)
JID_OPROBE=$(sbatch --parsable \
    --dependency=afterok:$JID_EXT_OCC \
    --export=ALL,BASE_DIR=$BASE_DIR \
    experiments/pipeline/06_probes/probe_occlusion.sh)
echo "3.  occ probe    -> job $JID_OPROBE"

# -- 4. Equivariance probe (depends on equivariance extraction)
JID_EPROBE=$(sbatch --parsable \
    --dependency=afterok:$JID_EXT_EQV \
    --export=ALL,BASE_DIR=$BASE_DIR \
    experiments/pipeline/06_probes/probe_equivariance.sh)
echo "4.  equiv probe  -> job $JID_EPROBE"

# -- 5. Velocity steering (depends on velocity extraction; decoder must already be trained separately)
CKPT="${BASE_DIR}/outputs/runs/moving_ball_decoder/checkpoints/last.pt"
if [ -f "$CKPT" ]; then
    for TGT in speed vel_x vel_y; do
        JID_STEER=$(sbatch --parsable \
            --dependency=afterok:$JID_EXT_VEL \
            --export=ALL,TARGET=$TGT,BASE_DIR=$BASE_DIR \
            experiments/threads/velocity/05_steering/steer_velocity.sh)
        echo "5.  steer $TGT   -> job $JID_STEER"
    done
else
    echo "5.  steer_velocity SKIPPED: no checkpoint at $CKPT"
    echo "    Train a decoder first, then run:"
    echo "    for TGT in speed vel_x vel_y; do"
    echo "        TARGET=\$TGT sbatch experiments/threads/velocity/05_steering/steer_velocity.sh"
    echo "    done"
fi

echo ""
echo "Watch jobs: squeue -u \$USER"
echo "Outputs:    $BASE_DIR/outputs/{demos,latents,analysis}/moving_ball*"
