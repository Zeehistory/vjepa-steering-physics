#!/bin/bash
#SBATCH --job-name=v2d_extract_act
# whenever any other job of yours already sits on gpu_devel -- even if the other partitions are free.
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2d_extract_act_%j.out
#SBATCH --error=logs/v2d_extract_act_%j.err

# ACTIVATION (MLP-write) twin of slurm_extract_v2d.sh: identical dataset/seed/layers, but the encoder
# is hooked at `mlp.fc2` instead of the block output, so the cache holds the MLP ACTIVATIONS rather
# than the residual stream. Clip-for-clip aligned with moving_ball_scene_v2d/<split> (same seed), so
# the PCA explorer can be rebuilt on activations and compared panel-for-panel against the latent site.
#
#   SPLIT=test sbatch experiments/threads/velocity/02_encode/extract_v2d_act.sh

SPLIT=${SPLIT:-"test"}
case "$SPLIT" in
  train) DEF_CLIPS=4000; DEF_SEED=0 ;;
  test)  DEF_CLIPS=800;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}
SITE=${SITE:-"mlp.fc2"}

BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene_v2d_act/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[v2d_extract_act] SPLIT=$SPLIT SITE=$SITE NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    encoder.hook_site=$SITE \
    data.scenario=scene_velocity2d \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.speed_range=[0.012,0.026]" \
    "data.radius_range=[0.11,0.11]"
STATUS=$?

echo "[v2d_extract_act] done (exit $STATUS): $SPLIT -> $OUTPUT_DIR"
exit $STATUS
