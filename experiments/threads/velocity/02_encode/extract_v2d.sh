#!/bin/bash
#SBATCH --job-name=v2d_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2d_extract_%j.out
#SBATCH --error=logs/v2d_extract_%j.err

# Extract VJEPA2-L latents for the 2D-VELOCITY scene dataset (scene_velocity2d): each scene = 8 clips
# sharing ONE start position, each with a distinct velocity VECTOR (direction AND speed vary). Same
# 256x256/16-frame/4-layer spec as the speed-only scene cache; clips_per_scene=8 and the capped
# speed_range are overridden on the CLI. Cache lands under moving_ball_scene_v2d/<split>.
#
#   SPLIT=train sbatch experiments/threads/velocity/02_encode/extract_v2d.sh   # 4000 clips (500 scenes x 8), seed 0
#   SPLIT=test  sbatch experiments/threads/velocity/02_encode/extract_v2d.sh   #  800 clips (100 scenes x 8), seed 2

SPLIT=${SPLIT:-"train"}
case "$SPLIT" in
  train) DEF_CLIPS=4000; DEF_SEED=0 ;;
  test)  DEF_CLIPS=800;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene_v2d/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[v2d_extract] SPLIT=$SPLIT NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=scene_velocity2d \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.speed_range=[0.012,0.026]" \
    "data.radius_range=[0.11,0.11]"
STATUS=$?

echo "[v2d_extract] done (exit $STATUS): $SPLIT -> $OUTPUT_DIR"
exit $STATUS