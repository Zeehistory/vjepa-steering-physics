#!/bin/bash
#SBATCH --job-name=rb3d_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rb3d_extract_%j.out
#SBATCH --error=logs/rb3d_extract_%j.err

# Extract VJEPA2-L latents for the 3D MuJoCo rolling-ball-on-a-table dataset (rolling_ball3d): each
# scene = 8 clips sharing ONE start on the tabletop, each clip a distinct velocity VECTOR, rendered
# through a FIXED perspective camera with real rolling physics. Same 256x256/16-frame/4-layer spec as
# moving_ball_scene_v2d(_mixed), so the steering comparison is apples-to-apples.
#
#   SPLIT=train sbatch experiments/threads/velocity/02_encode/extract_rb3d.sh   # 4000 clips (500 scenes x 8), seed 0
#   SPLIT=test  sbatch experiments/threads/velocity/02_encode/extract_rb3d.sh   #  800 clips (100 scenes x 8), seed 2
#
# 500 train scenes matches the 2D velocity baseline exactly, so the numbers are comparable.

# Storage dtype. fp16 halves the cache (a 4000-clip/4-layer fp32 split is ~172 GB, which
# does not fit the trainer's RAM); the sweep's dtype gate measured fp16 latents at a max
# top-8 subspace principal angle of 0.0012 deg, so this is lossless for our purposes.
#   LATENT_DTYPE=float16 FRAMES_DTYPE=float16 SPLIT=train sbatch ...
SPLIT=${SPLIT:-"train"}
case "$SPLIT" in
  train) DEF_CLIPS=4000; DEF_SEED=0 ;;
  test)  DEF_CLIPS=800;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

# Latents live on SCRATCH, not project: this cache is ~46 MB/clip -> ~220G for the full set, and the
# 4TB project quota is a shared, 97%-full group resource. Scratch has a 10TB group quota (~8TB free)
# and is purged after 60 days, which is exactly right for a regenerable intermediate.
LAT_BASE=${LAT_BASE:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${LAT_BASE}/outputs/latents/rolling_ball3d/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/rolling_ball3d_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# MuJoCo renders offscreen. EGL is the working backend on both the login node (mesa) and the GPU nodes
# (nvidia); osmesa is NOT installed in this env and glfw needs a display, so pin it explicitly.
export MUJOCO_GL=egl

echo "[rb3d_extract] SPLIT=$SPLIT NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    --latent_dtype ${LATENT_DTYPE:-float32} \
    --frames_dtype ${FRAMES_DTYPE:-float32} \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED
STATUS=$?

echo "[rb3d_extract] done (exit $STATUS): $SPLIT -> $OUTPUT_DIR"
exit $STATUS
