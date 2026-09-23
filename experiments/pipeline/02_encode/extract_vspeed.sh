#!/bin/bash
#SBATCH --job-name=vspeed_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/vspeed_extract_%j.out
#SBATCH --error=logs/vspeed_extract_%j.err

# SPEED-ONLY velocity scenes (scenario `scene_velocity`) for the basis-dependence comparison.
#
# Why this cache exists. The basis-dependence page compares a spin family whose scenes vary ONE scalar
# (omega, everything else bit-identical) against `scene_velocity2d`, whose scenes vary the velocity
# VECTOR -- direction and speed together. That makes the velocity panel a heading spray rather than the
# slow-vs-fast ladder the spin panel is, and the two sides are not matched. `scene_velocity` is the
# matched control: identical first frame AND identical direction within a scene, ranks ordered slow ->
# fast, so a within-scene pair isolates |v| exactly the way an angvel pair isolates omega.
#
# speed_range is the config's full [0.012, 0.045] rather than v2d's capped [0.012, 0.026]: with one
# shared direction there is no multi-heading feasibility constraint, so the ball can travel ~0.68 of the
# frame over the clip. That is deliberate -- it gives translation the WIDEST configuration sweep we can
# stage, which is the strongest chance for velocity's chart to fold the way spin's does. It does not.
#
# Layer 12 only (the layer every basis-dependence measurement is taken at): the 4-layer caches run ~35G
# per 800 clips and project storage is at 95%, so this lands on scratch and is re-extractable from the
# seed in minutes if the 30-day purge takes it.
#
#   sbatch experiments/pipeline/02_encode/extract_vspeed.sh

NUM_CLIPS=${NUM_CLIPS:-800}          # 100 scenes x 8 ranks
SEED=${SEED:-2}                      # test seed, scene-disjoint from train (0)
SCRATCH=${SCRATCH:-"."}
LAYERS=${LAYERS:-12}                  # 23 = last layer
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH}/outputs/latents/moving_ball_scene_vspeed${SUFFIX}/test/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[vspeed] NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers $LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=scene_velocity \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.speed_range=[0.012,0.045]" \
    "data.radius_range=[0.11,0.11]"
STATUS=$?
echo "[vspeed] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
