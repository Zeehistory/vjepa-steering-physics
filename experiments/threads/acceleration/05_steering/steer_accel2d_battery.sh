#!/bin/bash
#SBATCH --job-name=amix_battery
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=logs/amix_battery_%j.out
#SBATCH --error=logs/amix_battery_%j.err
# GPU decode battery for the parallel accel experiments. Runs on the frozen accel_mixed decoder ckpt:
#   A2  vanilla alpha sweep (does the true-Delta-H edit's MAGNITUDE scale? pixels)
#   B   bigU subspace steer at ku=64 (all ks 2..128) + lean cmd_U16/32/128 runs -> does a bigger
#       subspace help the command operator? (read-based subspace_U{k} ceiling comes free)
#   C   slabvel steer (temporal-composition velocity operator)
# Each command steer is followed by leakage-free gain calibration. Submit AFTER the CPU fits complete.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CFG=configs/train/moving_ball_scene_decoder.yaml
TEST=$BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large
CKPT=$BASE/outputs/runs/moving_ball_scene_accel2d_mixed_decoder_fp/checkpoints/last.pt
ANA=$BASE/outputs/analysis/moving_ball_accel2d_mixed
ART=$ANA/subspace
ARTB=$ANA/subspace_bigU
NS=${NUM_SCENES:-100}

calib () {  # $1 = steer output dir
  local SUM=$1/steer2d_summary.json
  [ -f "$SUM" ] && python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
      --out $1/calib_cmd_gain.json
}

echo "===== A2: vanilla alpha sweep (pixels) ====="
python -u experiments/threads/acceleration/05_steering/steer_accel_vanilla_alpha.py --config $CFG --test_dir $TEST --checkpoint $CKPT \
    --output_dir $ANA/steer_vanilla --alphas 0.5,1.0,1.5,2.0,2.5,3.0 --num_scenes 60 --device cuda

echo "===== C: slabvel (temporal-composition) ====="
python -u experiments/threads/acceleration/05_steering/steer_accel2d.py --config $CFG --test_dir $TEST --artifacts_dir $ART --checkpoint $CKPT \
    --output_dir $ANA/steer_slabvel --features slabvel --ks 2,4,8,16 --num_scenes $NS \
    --cmd_scales 1.0,1.5,2.0,2.5,3.0 --viz_scenes 4 --viz_gain 2.0 --device cuda
calib $ANA/steer_slabvel

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

echo "[amix_battery] ALL DONE"
