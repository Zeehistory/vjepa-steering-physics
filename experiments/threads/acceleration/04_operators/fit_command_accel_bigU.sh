#!/bin/bash
#SBATCH --job-name=amix_cmdbigU
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/amix_cmdbigU_%j.out
#SBATCH --error=logs/amix_cmdbigU_%j.err
# B: fit the command->U operator at several subspace ranks (ku=16,32,64,128) against the bigU basis, so
# steer can test whether a LARGER subspace lets the command operator capture more of the accel footprint.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
ART=$BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace_bigU
for KU in 16 32 64 128; do
  echo "=== fit cmd operator ku=$KU ==="
  python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py \
      --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
      --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
      --layers 6,12,18,23 --artifacts_dir $ART --ridge 1.0 --ku $KU
  echo "[amix_cmdbigU] ku=$KU exit=$?"
done
echo "[amix_cmdbigU] all done"
