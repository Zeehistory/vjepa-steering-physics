#!/bin/bash
#SBATCH --job-name=pf_dec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=04:00:00
#SBATCH --output=logs/posefield_decode_%j.out
#SBATCH --error=logs/posefield_decode_%j.err
# POSE-FIELD acceleration operator -- FIT + DECODE against the frozen accel2d_mixed decoder, honest
# parabola tracker. Reports held-out angle error / mag_corr vs the paste-the-truth ceiling and no-op.
# Run only on a basis the latent gate cleared. Submit:
#   BASIS=rbf_dv sbatch -p <partition> --gres=gpu:b200:1 experiments/pipeline/05_steering/posefield_decode.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CK="${1:-last}"
BASIS="${BASIS:-rbf_dv}"
python -u experiments/threads/acceleration/05_steering/steer_accel_posefield.py \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/${CK}.pt \
  --n_train_scenes ${NTRAIN:-500} --n_test_scenes ${NTEST:-30} \
  --bases "$BASIS" --decode_basis "$BASIS" \
  --viz_scenes ${VIZ_SCENES:-6} --viz_tag ${TAG:-v1} \
  --rbf_grid ${RBF_GRID:-8} --four_k ${FOUR_K:-4} --ridge ${RIDGE:-10.0} \
  --gains "${GAINS:-0.5,1,1.5,2,3,4}" --chunk ${CHUNK:-64} --device cuda \
  --out $BASE/outputs/analysis/moving_ball_accel2d_mixed/posefield/decode_${BASIS}_${TAG:-v1}.json
echo "[pf_dec] exit=$?"
