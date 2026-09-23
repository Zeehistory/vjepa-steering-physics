#!/bin/bash
#SBATCH --job-name=fid_base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fid_base_%j.out
#SBATCH --error=logs/fid_base_%j.err
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
FS=outputs/filmstrip/latents
RUNS=outputs/runs
OUT=outputs/hifi/fidelity
python experiments/pipeline/07_eval/eval_decoder_fidelity.py --label "baseline_v2d_mixed" \
  --config configs/train/moving_ball_scene_decoder.yaml \
  --test_dir "$FS/moving_ball_scene_v2d_mixed/test/vjepa2_large" \
  --checkpoint "$RUNS/moving_ball_scene_v2d_mixed_decoder_fp/checkpoints/last.pt" \
  --out "$OUT/baseline_v2d_mixed.json"
python experiments/pipeline/07_eval/eval_decoder_fidelity.py --label "baseline_rb3d" \
  --config configs/train/rolling_ball3d_decoder.yaml \
  --test_dir "$FS/rolling_ball3d/test/vjepa2_large" \
  --checkpoint "$RUNS/rolling_ball3d_decoder_fp_seed1/checkpoints/last.pt" \
  --out "$OUT/baseline_rb3d.json"
