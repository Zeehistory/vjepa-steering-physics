#!/bin/bash
#SBATCH --job-name=sq_extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sq_extract_%j.out
#SBATCH --error=logs/sq_extract_%j.err

# Cross-OBJECT control encode: same scene_velocity2d trajectories/velocities as the disk dataset, but the
# object is a SQUARE (shape=square). Identical 256x256/16-frame/4-layer spec + same token grid (8,16,16)
# so the disk-trained probe + operators (global_basis_L*, cmd_Wu_L*, ridge_Bt_L*) apply directly. Tests
# whether the velocity axis is object-AGNOSTIC (physics) or disk-bound (a dataset concept). Both splits in
# one job (gpu_devel allows only one queued job/user). Cache -> moving_ball_scene_v2d_square/<split>.
set -uo pipefail
module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
BASE_DIR=.
CONFIG=configs/train/moving_ball_scene_decoder.yaml

run_split () {
  local SPLIT=$1 NUM=$2 SEED=$3
  local OUT=${BASE_DIR}/outputs/latents/moving_ball_scene_v2d_square/${SPLIT}/vjepa2_large
  echo "[sq_extract] SPLIT=$SPLIT NUM=$NUM SEED=$SEED -> $OUT"
  python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" --encoder vjepa2_large --layers 6,12,18,23 \
    --output_dir "$OUT" --batch_size 8 --shard_size 128 \
    data.scenario=scene_velocity2d data.shape=square data.clips_per_scene=8 \
    data.num_clips=$NUM data.seed=$SEED \
    "data.speed_range=[0.012,0.026]" "data.radius_range=[0.11,0.11]"
}

run_split test 800 2
run_split train 4000 0
echo "[sq_extract] done"
