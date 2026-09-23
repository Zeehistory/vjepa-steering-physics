#!/bin/bash
#SBATCH --job-name=angvel_polar
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=01:30:00
#SBATCH --output=logs/angvel_polar_%j.out
#SBATCH --error=logs/angvel_polar_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAYER=${LAYER:-18}
python -u experiments/threads/angular-velocity/07_eval/diag_angvel_polar.py \
  --config configs/train/moving_ball_scene_angvel_decoder_orient.yaml \
  --train_dir $BASE/outputs/latents/moving_ball_scene_angvel2d/train/vjepa2_large \
  --test_dir  $BASE/outputs/latents/moving_ball_scene_angvel2d/test/vjepa2_large \
  --layer $LAYER --n_scenes 80 --n_r 8 --n_phi 16 --max_probe_clips 800 ${EXTRA:-} \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/diag_polar/diag_L${LAYER}${TAG:-}.json
echo "[angvel_polar] exit=$?"
