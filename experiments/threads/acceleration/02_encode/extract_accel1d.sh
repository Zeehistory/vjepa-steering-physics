#!/bin/bash
#SBATCH --job-name=accel1d_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/accel1d_extract_%j.out
#SBATCH --error=logs/accel1d_extract_%j.err

# 1-D ACCELERATION scenes (scenario `scene_accel1d`) for the basis-dependence comparison.
#
# The second-order twin of `scene_velocity` and the translational twin of `scene_angaccel2d`: every rank
# shares the start, appearance, heading AND initial speed, and differs only in a SIGNED acceleration along
# that one line -- half speeding up, half slowing down, exactly as the angular-acceleration family shares
# omega0 and ramps alpha in both signs. Without it the order-2 comparison would repeat the asymmetry the
# speed ladder was built to remove: `scene_accel2d` varies the acceleration VECTOR (direction as well as
# magnitude) while `scene_angaccel2d` varies one scalar.
#
# Ranges match `scene_accel2d` so the two acceleration families are directly comparable.
# Layer 12 only, onto scratch (project is at 95%); re-extractable from the seed in ~12 min.
#
#   sbatch experiments/threads/acceleration/02_encode/extract_accel1d.sh

NUM_CLIPS=${NUM_CLIPS:-800}          # 100 scenes x 8 ranks
SEED=${SEED:-2}                      # test seed, scene-disjoint from train (0)
SCRATCH_DIR=${SCRATCH_DIR:-"."}
LAYERS=${LAYERS:-12}                  # 23 = last layer
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRATCH_DIR}/outputs/latents/moving_ball_scene_accel1d${SUFFIX}/test/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_decoder.yaml"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[accel1d] NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers $LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=scene_accel1d \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.speed_range=[0.008,0.016]" \
    "data.accel_range=[0.0015,0.0035]" \
    "data.radius_range=[0.11,0.11]"
STATUS=$?
echo "[accel1d] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
