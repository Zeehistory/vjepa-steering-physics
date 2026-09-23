#!/bin/bash
#SBATCH --job-name=var_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/var_extract_%j.out
#SBATCH --error=logs/var_extract_%j.err

# Extract VJEPA2-L latents for a NUISANCE-CONTROL moving-ball variant (scene_size/color/background).
# Same 256x256/16-frame/4-layer spec as slurm_extract_scene.sh, but the scenario (and, for size, the
# radius range) is overridden on the CLI and the cache lands under a per-variant directory.
#
#   SCENARIO=scene_size       SPLIT=train sbatch experiments/pipeline/02_encode/extract_variant.sh
#   SCENARIO=scene_color      SPLIT=test  sbatch experiments/pipeline/02_encode/extract_variant.sh
#   SCENARIO=scene_background SPLIT=train sbatch experiments/pipeline/02_encode/extract_variant.sh
# Optional: RADIUS_LO/RADIUS_HI override the ramped radius range (size variant only).

SCENARIO=${SCENARIO:?set SCENARIO=scene_size|scene_color|scene_background}
SPLIT=${SPLIT:-"train"}
case "$SPLIT" in
  train) DEF_CLIPS=2000; DEF_SEED=0 ;;
  val)   DEF_CLIPS=400;  DEF_SEED=1 ;;
  test)  DEF_CLIPS=400;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|val|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

# size ramps radius; color/background hold radius fixed at the scene_velocity value (0.11).
case "$SCENARIO" in
  scene_size)        RADIUS_LO=${RADIUS_LO:-0.07}; RADIUS_HI=${RADIUS_HI:-0.13} ;;
  scene_color|scene_background) RADIUS_LO=${RADIUS_LO:-0.11}; RADIUS_HI=${RADIUS_HI:-0.11} ;;
  *) echo "unknown SCENARIO=$SCENARIO"; exit 1 ;;
esac

# short tag for the cache dir: scene_size -> size, etc.
TAG=${SCENARIO#scene_}
BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene_${TAG}/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[var_extract] SCENARIO=$SCENARIO SPLIT=$SPLIT NUM_CLIPS=$NUM_CLIPS SEED=$SEED radius=[$RADIUS_LO,$RADIUS_HI] -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=$SCENARIO \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.radius_range=[$RADIUS_LO,$RADIUS_HI]"
STATUS=$?

echo "[var_extract] done (exit $STATUS): $SCENARIO/$SPLIT -> $OUTPUT_DIR"
exit $STATUS
