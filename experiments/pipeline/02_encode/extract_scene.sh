#!/bin/bash
#SBATCH --job-name=scene_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/scene_extract_%j.out
#SBATCH --error=logs/scene_extract_%j.err

# Extract VJEPA2-L latents for the scene-paired moving-ball dataset (256x256, 16 frames).
# Scene-disjoint splits via distinct seeds. Generated on-the-fly (no download).
#
#   SPLIT=train sbatch experiments/pipeline/02_encode/extract_scene.sh   # 2000 clips (500 scenes), seed 0
#   SPLIT=val   sbatch experiments/pipeline/02_encode/extract_scene.sh   # 400 clips  (100 scenes), seed 1
#   SPLIT=test  sbatch experiments/pipeline/02_encode/extract_scene.sh   # 400 clips  (100 scenes), seed 2
# Pilot: SPLIT=train NUM_CLIPS=160 sbatch ... ; SPLIT=test NUM_CLIPS=40 sbatch ...

SPLIT=${SPLIT:-"train"}
case "$SPLIT" in
  train) DEF_CLIPS=2000; DEF_SEED=0 ;;
  val)   DEF_CLIPS=400;  DEF_SEED=1 ;;
  test)  DEF_CLIPS=400;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|val|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# NOTE on invocation: the data block comes from --config (which already carries the scene_velocity
# spec), NOT from --dataset, because extract_latents merges a --dataset file AFTER overrides and would
# clobber num_clips/seed. We DO pass --encoder (for device/dtype) and --layers (the vjepa2_large encoder
# file sets layers: all, which --layers overrides so the cache stores only the 4 decoded layers).
echo "[scene_extract] SPLIT=$SPLIT NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED
STATUS=$?

echo "[scene_extract] done (exit $STATUS): $SPLIT -> $OUTPUT_DIR"
exit $STATUS
