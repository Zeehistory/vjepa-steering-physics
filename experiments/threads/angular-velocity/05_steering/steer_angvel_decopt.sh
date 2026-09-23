#!/bin/bash
#SBATCH --job-name=angvel_decopt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/angvel_decopt_%j.out
#SBATCH --error=logs/angvel_decopt_%j.err
# Decoder-in-the-loop test-time edit optimization for ANGULAR VELOCITY (free full-D edit, init zero).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=${CFG:-configs/train/moving_ball_scene_angvel_decoder_orient.yaml}
TEST=$BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large
DECRUN=${DECRUN:-moving_ball_scene_angvel2d_decoder_orient}
CKPT=$BASE/outputs/runs/$DECRUN/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_angvel2d
NS=${NUM_SCENES:-30}
STEPS=${STEPS:-300}
LR=${LR:-0.03}
OUT=${OUT:-steer_decopt}
VERB=""; [ "${VERBOSE:-0}" = "1" ] && VERB="--verbose"
python -u experiments/threads/angular-velocity/05_steering/steer_angvel_decopt.py --config $CFG --test_dir $TEST \
    --checkpoint $CKPT --output_dir $ANA/$OUT \
    --steps $STEPS --lr $LR --l2 ${L2:-1e-3} \
    --anchor_phase ${ANCHOR_PHASE:-2.0} --anchor_centre ${ANCHOR_CENTRE:-5.0} \
    --anchor_mass ${ANCHOR_MASS:-1.0} --eval_every ${EVAL_EVERY:-20} \
    --viz_scenes ${VIZ_SCENES:-0} --num_scenes $NS $VERB --device cuda
echo "[angvel_decopt] exit=$? OUT=$OUT STEPS=$STEPS LR=$LR"
