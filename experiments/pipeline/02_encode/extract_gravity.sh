#!/bin/bash
#SBATCH --job-name=grav_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gravity_extract_%j.out
#SBATCH --error=logs/gravity_extract_%j.err

# Extract VJEPA2-L latents for the GRAVITY / projectile dataset (scene_gravity): each scene = 8 clips
# sharing ONE start + ONE roughly-horizontal launch v0, each clip a distinct DOWNWARD gravity |g|.
# Same 256x256/16-frame/4-layer spec as scene_accel2d. Cache -> moving_ball_scene_gravity/<split>.
#
#   SPLIT=train sbatch experiments/pipeline/02_encode/extract_gravity.sh   # 4000 clips (500 scenes x 8), seed 0
#   SPLIT=test  sbatch experiments/pipeline/02_encode/extract_gravity.sh   #  800 clips (100 scenes x 8), seed 2

SPLIT=${SPLIT:-"train"}
case "$SPLIT" in
  train) DEF_CLIPS=4000; DEF_SEED=0 ;;
  test)  DEF_CLIPS=800;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene_gravity/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[gravity_extract] SPLIT=$SPLIT NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=scene_gravity \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.speed_range=[0.008,0.016]" \
    "data.gravity_range=[0.0015,0.0040]" \
    "data.radius_range=[0.08,0.12]"
STATUS=$?

echo "[gravity_extract] done (exit $STATUS): $SPLIT -> $OUTPUT_DIR"
exit $STATUS
