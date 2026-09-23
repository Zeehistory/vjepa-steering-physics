#!/bin/bash
#SBATCH --job-name=amix_bigUsteer
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=logs/amix_bigUsteer_%j.out
#SBATCH --error=logs/amix_bigUsteer_%j.err
# B: enlarged-subspace decode. Comprehensive ku=64 (ks 2..128 -> read-based subspace ceiling free) plus
# lean cmd_U16/32/128. Submit AFTER the bigU cmd fits complete.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_decoder.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
ARTB=$ANA/subspace_bigU
NS=${NUM_SCENES:-100}

calib () { local S=$1/steer2d_summary.json; [ -f "$S" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py \
    --summary "$S" --val_frac 0.5 --out $1/calib_cmd_gain.json; }

echo "===== B: bigU comprehensive ku=64 (ks 2..128) ====="
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py --config $CFG --test_dir $TEST --artifacts_dir $ARTB --checkpoint $CKPT \
    --output_dir $ANA/steer_bigU_ku64 --cmd_ku 64 --ks 2,4,8,16,32,64,128 --num_scenes $NS \
    --cmd_scales 1.0,1.5,2.0,2.5,3.0 --device cuda
calib $ANA/steer_bigU_ku64

for KU in 16 32 128; do
  echo "===== B: bigU lean cmd_U$KU ====="
  python -u experiments/threads/acceleration/05_steering/steer_accel2d.py --config $CFG --test_dir $TEST --artifacts_dir $ARTB --checkpoint $CKPT \
      --output_dir $ANA/steer_bigU_ku$KU --cmd_ku $KU --ks 8 --num_scenes $NS \
      --cmd_scales 1.0,1.5,2.0,2.5,3.0 --device cuda
  calib $ANA/steer_bigU_ku$KU
done
echo "[amix_bigUsteer] ALL DONE"
