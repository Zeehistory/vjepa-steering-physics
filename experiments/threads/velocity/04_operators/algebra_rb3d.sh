#!/bin/bash
#SBATCH --job-name=rb3d_algebra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rb3d_algebra_%j.out
#SBATCH --error=logs/rb3d_algebra_%j.err
# Identity / inverse / composition / commutativity for the command-only velocity operator, scored on
# decoded + pixel-tracked velocity as well as latent distance. ~10 decodes per scene, so this is
# roughly 10x a plain steer run per scene -- 40 scenes fits comfortably in the 2h wall.
#
# GAIN must be the gain that was selected on VALIDATION for this dataset (rolling_ball3d: 2.0, from
# analysis/rolling_ball3d/steer_last/calib_cmd_gain.json). Using a different one would test a
# different operator than the one the paper reports.
#
#   sbatch -p <partition> experiments/threads/velocity/04_operators/algebra_rb3d.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

BASE=.
LAT_BASE=${LAT_BASE:-.}
OUTTAG=${OUTTAG:-algebra}

python -u experiments/threads/velocity/04_operators/algebra_velocity2d.py \
    --config configs/train/rolling_ball3d_decoder.yaml \
    --test_dir $LAT_BASE/outputs/latents/rolling_ball3d/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/rolling_ball3d/subspace \
    --checkpoint ${CKPT:-$LAT_BASE/outputs/runs/rolling_ball3d_decoder_fp/checkpoints/last.pt} \
    --output_dir $BASE/outputs/analysis/rolling_ball3d/${OUTTAG} \
    --num_scenes ${NUM_SCENES:-40} \
    --gain ${GAIN:-2.0} \
    --device cuda
RC=$?
echo "[rb3d_algebra] exit=$RC out=${OUTTAG}"
exit $RC
