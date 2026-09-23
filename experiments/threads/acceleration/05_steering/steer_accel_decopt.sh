#!/bin/bash
#SBATCH --job-name=amix_decopt
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/amix_decopt_%j.out
#SBATCH --error=logs/amix_decopt_%j.err
# Decoder-in-the-loop test-time edit optimization. MODE=free (full-D, init canon) or subspace.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_decoder.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
MODE=${MODE:-free}
NS=${NUM_SCENES:-30}
STEPS=${STEPS:-200}
LR=${LR:-0.02}
ART=${ART:-$ANA/subspace}
BASIS=${BASIS:-global_basis_canon}
RANK=${RANK:-16}
OUT=${OUT:-steer_decopt_$MODE}
VERB=""; [ "${VERBOSE:-0}" = "1" ] && VERB="--verbose"
BDIR="${BASIS_DIR:-$ART}"
python -u experiments/threads/acceleration/05_steering/steer_accel_decopt.py --config $CFG --test_dir $TEST --artifacts_dir $ART \
    --checkpoint $CKPT --output_dir $ANA/$OUT --mode $MODE --basis $BASIS --rank $RANK --basis_dir $BDIR \
    --init ${INIT:-canon} --steps $STEPS --lr $LR --l2 ${L2:-1e-3} --anchor ${ANCHOR:-0.2} \
    --viz_scenes ${VIZ_SCENES:-0} --num_scenes $NS $VERB --device cuda
echo "[amix_decopt] exit=$? MODE=$MODE OUT=$OUT"
