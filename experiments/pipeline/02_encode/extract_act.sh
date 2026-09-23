#!/bin/bash
#SBATCH --job-name=act_extract
# gpu_devel is deliberately absent: the devel partition has MaxSubmitJobsPerUser=1, and Slurm applies that
# at SUBMIT time to the whole list -- one job of yours already on gpu_devel makes every sbatch that
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/act_extract_%j.out
#SBATCH --error=logs/act_extract_%j.err

# ACTIVATION twin of the three latent extractions behind the PCA explorer site. Same datasets, same
# seeds, same clip-for-clip splits, same layers as slurm_extract_v2d.sh / _angvel2d.sh / _accel2d.sh
# -- the ONLY difference is encoder.hook_site, which taps the MLP write (`mlp.fc2`) instead of the
# block output (the residual stream) the published site was built on. Caches land beside the latent
# ones under <family>_act/, on PROJECT rather than scratch (scratch is purged monthly).
#
#   FAMILY=velocity SPLIT=test sbatch experiments/pipeline/02_encode/extract_act.sh
#   FAMILY=angvel   SPLIT=test sbatch experiments/pipeline/02_encode/extract_act.sh
#   FAMILY=accel    SPLIT=test sbatch experiments/pipeline/02_encode/extract_act.sh
#
# SITE= overrides the tap (e.g. SITE=attn.proj) without touching anything downstream.

FAMILY=${FAMILY:-"velocity"}
SPLIT=${SPLIT:-"test"}
SITE=${SITE:-"mlp.fc2"}

case "$SPLIT" in
  train) DEF_CLIPS=4000; DEF_SEED=0 ;;
  test)  DEF_CLIPS=800;  DEF_SEED=2 ;;
  *) echo "unknown SPLIT=$SPLIT (train|test)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-$DEF_CLIPS}
SEED=${SEED:-$DEF_SEED}

# Per-family dataset block, copied verbatim from that family's latent extraction script so the two
# caches are clip-for-clip aligned. Change one of these ONLY alongside its latent twin.
case "$FAMILY" in
  velocity)
    CACHE="moving_ball_scene_v2d_act"
    CONFIG="configs/train/moving_ball_scene_decoder.yaml"
    DATA_ARGS=(data.scenario=scene_velocity2d
               "data.speed_range=[0.012,0.026]"
               "data.radius_range=[0.11,0.11]") ;;
  angvel)
    CACHE="moving_ball_scene_angvel2d_act"
    CONFIG="configs/train/moving_ball_scene_angvel_decoder.yaml"
    DATA_ARGS=(data.scenario=scene_angvel2d
               "data.omega_range=[0.06,0.20]"
               "data.radius_range=[0.11,0.15]") ;;
  accel)
    CACHE="moving_ball_scene_accel2d_act"
    CONFIG="configs/train/moving_ball_scene_decoder.yaml"
    DATA_ARGS=(data.scenario=scene_accel2d
               "data.speed_range=[0.008,0.016]"
               "data.accel_range=[0.0015,0.0035]"
               "data.radius_range=[0.08,0.12]") ;;
  *) echo "unknown FAMILY=$FAMILY (velocity|angvel|accel)"; exit 1 ;;
esac

BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/${CACHE}/${SPLIT}/vjepa2_large"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[act_extract] FAMILY=$FAMILY SPLIT=$SPLIT SITE=$SITE NUM_CLIPS=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    encoder.hook_site=$SITE \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "${DATA_ARGS[@]}"
STATUS=$?

echo "[act_extract] done (exit $STATUS): $FAMILY/$SPLIT -> $OUTPUT_DIR"
exit $STATUS
