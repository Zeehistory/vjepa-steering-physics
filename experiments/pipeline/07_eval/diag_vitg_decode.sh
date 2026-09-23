#!/bin/bash
#SBATCH --job-name=vitg_diag
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vitg_diag_%j.out
#SBATCH --error=logs/vitg_diag_%j.err
# GATE: is the ViT-g angaccel decoder a faint SMEAR or a threshold-calibration problem?
# Result #9 (job 18461471): decode(true H_b) ceiling rho=0.111 = CHANCE, yet the decoder's own
# frame_orientation loss converged to 0.0009. Same pathology as the old small-object angvel decoder,
# where default thr(0.5,0.25) gave rho 0.13 but tight thr(0.25,0.08) gave 0.98. This decodes true H_a/H_b
# for held-out ViT-g angaccel scenes and reports measured rotation across 4 threshold pairs + marker
# redness / bar darkness stats. If n_valid jumps and rho recovers at relaxed thresholds -> tracker
# calibration fix (not a decoder retrain); if redness/darkness are ~0 everywhere -> genuine smear.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/angular-velocity/07_eval/diag_angvel_decode.py \
  --config configs/train/moving_ball_scene_angvel_vitg_decoder.yaml \
  --test_dir $BASE/outputs/latents/moving_ball_scene_angaccel2d_vitg_test/test/vjepa2_giant \
  --checkpoint $BASE/outputs/runs/moving_ball_scene_angaccel2d_vitg_decoder/checkpoints/last.pt \
  --num_scenes 16 --device cuda
echo "[vitg_diag] exit=$?"
