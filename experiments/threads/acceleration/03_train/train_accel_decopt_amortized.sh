#!/bin/bash
#SBATCH --job-name=amix_amort
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=05:00:00
#SBATCH --output=logs/amix_amort_%j.out
#SBATCH --error=logs/amix_amort_%j.err
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/threads/acceleration/03_train/train_accel_decopt_amortized.py \
    --config configs/train/moving_ball_scene_decoder.yaml \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --checkpoint $BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt \
    --output_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/decopt_amortized \
    --epochs ${EPOCHS:-30} --batch ${BATCH:-4} --lr ${LR:-1e-3} --anchor ${ANCHOR:-1.0} \
    --out_scale ${OUT_SCALE:-0.05} --eval_n ${EVAL_N:-30} --eval_every ${EVAL_EVERY:-3} \
    --max_train_scenes ${MAX_TRAIN:-0} --n_targets ${N_TARGETS:-3} --device cuda
echo "[amix_amort] exit=$?"
