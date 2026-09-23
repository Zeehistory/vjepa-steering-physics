#!/bin/bash
#SBATCH --job-name=angvel_polar_big
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=01:30:00
#SBATCH --output=logs/angvel_polar_big_%j.out
#SBATCH --error=logs/angvel_polar_big_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
BIG=$BASE/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large
LAYER=${LAYER:-12}
# big object spans ~5-7 cells -> widen the annulus (r 2-7 cells) so it covers the rotating bar
python -u experiments/threads/angular-velocity/07_eval/diag_angvel_polar.py \
  --config configs/train/moving_ball_scene_angvel_decoder.yaml \
  --train_dir $BIG --test_dir $BIG \
  --layer $LAYER --n_scenes 125 --n_r 8 --n_phi 16 --skip_probe \
  --ann_r_lo 1.5 --ann_r_hi 7.5 --ann_n_r 8 --ann_n_phi 32 \
  --out $BASE/outputs/analysis/moving_ball_angvel2d/diag_polar/diag_big_L${LAYER}.json
echo "[angvel_polar_big] exit=$?"
