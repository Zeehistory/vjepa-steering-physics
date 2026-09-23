#!/bin/bash
#SBATCH --job-name=ablate_fourier
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablate_fourier_%j.out
#SBATCH --error=logs/ablate_fourier_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
QUANTITY=${QUANTITY:-angaccel}
FIT=${FIT:-angvel}
case "$FIT" in
  angvel)   TRAIN_DIR=${TRAIN_DIR:-$BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large} ;;
  angaccel) TRAIN_DIR=${TRAIN_DIR:-$BASE/outputs/latents/moving_ball_scene_angaccel2d_train/test/vjepa2_large} ;;
esac
case "$QUANTITY" in
  angaccel) TEST_DIR=${TEST_DIR:-$BASE/outputs/latents/moving_ball_scene_angaccel2d_test/test/vjepa2_large} ;;
  angvel)   TEST_DIR=${TEST_DIR:-$BASE/outputs/latents/moving_ball_scene_angvel2d_big_test/test/vjepa2_large} ;;
esac
python -u experiments/pipeline/07_eval/ablate_fourier.py \
  --config configs/train/moving_ball_scene_angvel_big_decoder_orient.yaml \
  --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR" \
  --quantity "$QUANTITY" --n_train_scenes ${NTRAIN:-125} --n_test_scenes ${NTEST:-30} \
  --axes ${AXES:-order,harmonics,canon,basis,ridge,n_train} \
  --out $BASE/outputs/analysis/moving_ball_angaccel2d/ablation_${TAG:-${QUANTITY}_fit${FIT}}.json
echo "[ablate] exit=$?"
