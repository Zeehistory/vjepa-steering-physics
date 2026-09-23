#!/bin/bash
#SBATCH --job-name=angvel_decopt_big
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/angvel_decopt_big_%j.out
#SBATCH --error=logs/angvel_decopt_big_%j.err
# BIG-OBJECT decoder-in-the-loop TTO steer for angular velocity (accel's 5.07deg method). Free full-D edit
# optimized against the frozen big-object decoder; honest measured_angvel eval every EVAL_EVERY steps.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large
DECRUN=moving_ball_scene_angvel2d_big_decoder_orient
CKPT=$BASE/outputs/runs/$DECRUN/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_angvel2d
NS=${NUM_SCENES:-30}
STEPS=${STEPS:-300}
LR=${LR:-0.03}
OUT=${OUT:-steer_decopt_big}
VERB=""; [ "${VERBOSE:-0}" = "1" ] && VERB="--verbose"
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_decopt.py --config $CFG --test_dir $TEST \
    --checkpoint $CKPT --output_dir $ANA/$OUT \
    --steps $STEPS --lr $LR --l2 ${L2:-1e-3} \
    --anchor_phase ${ANCHOR_PHASE:-2.0} --anchor_centre ${ANCHOR_CENTRE:-5.0} \
    --anchor_mass ${ANCHOR_MASS:-1.0} --eval_every ${EVAL_EVERY:-20} \
    --viz_scenes ${VIZ_SCENES:-6} --num_scenes $NS $VERB --device cuda
echo "[angvel_decopt_big] exit=$? OUT=$OUT STEPS=$STEPS LR=$LR"
