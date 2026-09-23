#!/bin/bash
#SBATCH --job-name=angaccel_extract
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/angaccel_extract_%j.out
#SBATCH --error=logs/angaccel_extract_%j.err
# ANGULAR-ACCELERATION latents: the second-order rotational quantity (theta(t) = theta0 + w0*t + a/2*t^2).
# Deliberately mirrors _extract_angvel2d_big.sh (BIG object, radius 0.32-0.42, same encoder/layers/shards)
# so the angvel operator can be applied to these latents with NOTHING changed but the orientation roll-out
# -- that apples-to-apples match is what makes the ZERO-SHOT transfer claim (fit on constant-omega, steer
# alpha) a clean test rather than a confound.
#
#   SPLIT=test  (default) seed 11, 240 clips / 30 scenes (~11G) -- gate + steer target
#   SPLIT=train           seed  7, 1000 clips / 125 scenes (~43G) -- matched-fit control only
#   SCENARIO=scene_angaccel2d_mixed  for the appearance-transfer variant
SPLIT=${SPLIT:-test}
SCENARIO=${SCENARIO:-scene_angaccel2d}
if [ "$SPLIT" = "train" ]; then
  NUM_CLIPS=${NUM_CLIPS:-1000}; SEED=${SEED:-7}
else
  NUM_CLIPS=${NUM_CLIPS:-240};  SEED=${SEED:-11}
fi
BASE_DIR=${BASE_DIR:-"."}
NAME=${NAME:-"moving_ball_scene_angaccel2d_${SPLIT}"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/${NAME}/test/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_angvel_decoder.yaml"}
ENCODER=${ENCODER:-vjepa2_large}
LAYERS=${LAYERS:-6,12,18,23}
RADIUS=${RADIUS:-"[0.32,0.42]"}
OMEGA0=${OMEGA0:-"[-0.06,0.06]"}
ALPHA=${ALPHA:-"[0.005,0.014]"}
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
echo "[angaccel] SPLIT=$SPLIT SCENARIO=$SCENARIO CLIPS=$NUM_CLIPS SEED=$SEED RADIUS=$RADIUS ALPHA=$ALPHA -> $OUTPUT_DIR"
df -h . | tail -1
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder "$ENCODER" \
    --layers "$LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=$SCENARIO \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.radius_range=$RADIUS" \
    "data.omega0_range=$OMEGA0" \
    "data.alpha_range=$ALPHA"
STATUS=$?
df -h . | tail -1
echo "[angaccel] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
