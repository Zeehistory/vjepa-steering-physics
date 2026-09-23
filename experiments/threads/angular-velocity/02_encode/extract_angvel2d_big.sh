#!/bin/bash
#SBATCH --job-name=angvel_big_extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/angvel_big_extract_%j.out
#SBATCH --error=logs/angvel_big_extract_%j.err
# BIG-OBJECT angular-velocity probe set: SAME scene_angvel2d pipeline but a LARGE object (radius 0.32-0.42
# vs 0.11-0.15) so the rotating bar spans ~5-7 token cells instead of ~2 -> the log-polar phi-structure is
# resolvable. Tests the RESOLUTION hypothesis (diag_polar size-trend predicts best|cos| 0.33 -> ~0.6+). One
# split only (used by the shared-axis pairwise-cosine test, which needs a single set of scenes).
NUM_CLIPS=${NUM_CLIPS:-1000}
SEED=${SEED:-7}
BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene_angvel2d_big/test/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_angvel_decoder.yaml"}
RADIUS=${RADIUS:-"[0.32,0.42]"}
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
echo "[angvel_big] NUM_CLIPS=$NUM_CLIPS SEED=$SEED RADIUS=$RADIUS -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=scene_angvel2d \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.omega_range=[0.06,0.20]" \
    "data.radius_range=$RADIUS"
STATUS=$?
echo "[angvel_big] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
