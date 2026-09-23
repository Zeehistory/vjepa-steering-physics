#!/bin/bash
#SBATCH --job-name=fg6_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fg6_steer_%j.out
#SBATCH --error=logs/fg6_steer_%j.err
# Run all three velocity-steering targets sequentially against the fg6 (state co-training) decoder.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd .

BASE=.
CONFIG=configs/train/moving_ball_decoder_large.yaml
CKPT=$BASE/outputs/runs/moving_ball_decoder_fg6/checkpoints/step_3000.pt
LAT=$BASE/outputs/latents/moving_ball_velocity/vjepa2_large

for TARGET in speed vel_x vel_y; do
  echo "==================== STEER target=$TARGET ===================="
  OUT=$BASE/outputs/analysis/moving_ball_velocity/steer_${TARGET}
  python experiments/threads/velocity/05_steering/steer_velocity.py \
    --config "$CONFIG" \
    --source_latent_dir "$LAT" \
    --target_latent_dir "$LAT" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --target "$TARGET" \
    --all_layers \
    --alphas="-1.5,-1.0,-0.5,0,0.5,1.0,1.5" \
    --num_samples 6 \
    --device cuda \
    data.image_size=128 data.num_frames=32 data.fps=8 \
    decoder.out_image_size=128 decoder.out_num_frames=32
  echo "[fg6_steer] target=$TARGET exit=$?"
done
echo "[fg6_steer] ALL DONE"
