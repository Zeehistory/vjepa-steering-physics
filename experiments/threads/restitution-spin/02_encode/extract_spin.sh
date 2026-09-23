#!/bin/bash
#SBATCH --job-name=spin_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/spin_extract_%j.out
#SBATCH --error=logs/spin_extract_%j.err

# VJEPA2-L latents for the velocity x spin CROSSTALK scene (spin_ball3d): a marked ball gliding across
# a table while spinning about the vertical axis, with v and omega sampled INDEPENDENTLY on a complete
# 4x4 factorial per scene. Same 256x256 / 16-frame / 4-layer spec as rolling_ball3d, so the translation
# half of every result is directly comparable to the cmd_U8 velocity baseline measured there.
#
#   SPLIT=train sbatch experiments/threads/restitution-spin/02_encode/extract_spin.sh   # 4096 clips (256 scenes x 16), seed 0
#   SPLIT=test  sbatch experiments/threads/restitution-spin/02_encode/extract_spin.sh   # 1024 clips ( 64 scenes x 16), seed 2
#
# time rather than the run.

SPLIT=${SPLIT:-"train"}
case "$SPLIT" in
  train) DEF_CLIPS=4096; DEF_SEED=0 ;;
  test)  DEF_CLIPS=1024; DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

# Latents live on SCRATCH: ~46 MB/clip -> ~190G for the train split, against a project quota that is a
# shared, 96%-full group resource. Scratch is the right home for a regenerable intermediate (the scene
# is deterministic in (seed, index) and the encoder is frozen, so this cache is reproducible on demand).
LAT_BASE=${LAT_BASE:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${LAT_BASE}/outputs/latents/spin_ball3d/${SPLIT}/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/spin_ball3d_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# MuJoCo renders offscreen; EGL is the backend that works on both login and GPU nodes in this env.
export MUJOCO_GL=egl

echo "[spin_extract] SPLIT=$SPLIT NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
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
echo "[spin_extract] done (exit $STATUS): $SPLIT -> $OUTPUT_DIR"
exit $STATUS
