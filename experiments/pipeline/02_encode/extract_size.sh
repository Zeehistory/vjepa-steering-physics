#!/bin/bash
#SBATCH --job-name=size_extract
# gpu_devel is deliberately absent -- see the note in extract_act.sh (its MaxSubmitJobsPerUser=1 is
# applied at submit time to the whole partition LIST, so naming it can fail every sbatch).
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/size_extract_%j.out
#SBATCH --error=logs/size_extract_%j.err

# MODEL-SIZE twin of extract_act.sh: the same three PCA-explorer families, the same datasets, seeds,
# splits and tap (mlp.fc2), encoded by a LARGER backbone. Everything except `--encoder` and
# `--layers` is copied verbatim from extract_act.sh so the caches stay clip-for-clip aligned.
#
#   FAMILY=velocity SIZE=huge  sbatch experiments/pipeline/02_encode/extract_size.sh
#   FAMILY=angvel   SIZE=giant sbatch experiments/pipeline/02_encode/extract_size.sh
#   FAMILY=velocity SIZE=huge TAP=residual sbatch ...   # the residual-stream twin (see TAP below)
#
# TAP (2026-09-12). The sweep was first run on mlp.fc2 only, and the model-size page therefore
# compared against the fc2 explorer (mlp-activations.html) while readers compared it against the
# residual-stream explorer (index.html) -- which is the one everybody remembers. On the residual
# stream the energy-masked velocity path is smooth at EVERY ViT-L depth (0.27-0.31 normalised);
# on fc2 it is smooth at half depth but random at the final block (0.94), so the two pages looked
# contradictory. They are two taps, not two results. TAP=residual encodes the block output
# (hook_site=block) into a *_res_sz cache so the page can show both.
#
# LAYERS ARE NOT COMPARABLE ACROSS BACKBONES. ViT-L's 6,12,18,23 sit at 25/50/75/96% of its 24
# blocks; the same indices in ViT-g's 40 blocks are 15/30/45/58% -- a depth comparison wearing a
# size comparison's clothes. Each size therefore gets the indices at MATCHED RELATIVE DEPTH:
#     large  (24 blocks):  6,12,18,23
#     huge   (32 blocks):  8,16,24,31
#     giant  (40 blocks): 10,20,30,38
#
# NUM_CLIPS defaults to 32 (4 scenes x 8 clips), not the 800 of the full caches: the explorer reads
# 3 scenes x 4 ranks, and a full ViT-g cache would be ~40 GB against a project quota already at 97%.
# The generator is seeded, so these 32 are the FIRST 32 clips of the corresponding full cache --
# SIZE=large reproduces the published ViT-L cache clip-for-clip and exists as exactly that control.

FAMILY=${FAMILY:-"velocity"}
SIZE=${SIZE:-"giant"}
SPLIT=${SPLIT:-"test"}
TAP=${TAP:-fc2}          # fc2 | residual
case "$TAP" in
  fc2)      SITE="mlp.fc2"; SFX="_act_sz" ;;
  residual) SITE="block";   SFX="_res_sz" ;;
  *) echo "unknown TAP=$TAP (fc2|residual)"; exit 1 ;;
esac
NUM_CLIPS=${NUM_CLIPS:-32}
SEED=${SEED:-2}          # test-split seed, as in extract_act.sh

case "$SIZE" in
  large) ENCODER=vjepa2_large; LAYERS="6,12,18,23"; BS=${BS:-8} ;;
  huge)  ENCODER=vjepa2_huge;  LAYERS="8,16,24,31"; BS=${BS:-4} ;;
  giant) ENCODER=vjepa2_giant; LAYERS="10,20,30,38"; BS=${BS:-2} ;;
  *) echo "unknown SIZE=$SIZE (large|huge|giant)"; exit 1 ;;
esac

# --- per-family dataset block, copied verbatim from extract_act.sh ---
case "$FAMILY" in
  velocity)
    CACHE="moving_ball_scene_v2d"
    CONFIG="configs/train/moving_ball_scene_decoder.yaml"
    DATA_ARGS=(data.scenario=scene_velocity2d
               "data.speed_range=[0.012,0.026]"
               "data.radius_range=[0.11,0.11]") ;;
  angvel)
    CACHE="moving_ball_scene_angvel2d"
    CONFIG="configs/train/moving_ball_scene_angvel_decoder.yaml"
    DATA_ARGS=(data.scenario=scene_angvel2d
               "data.omega_range=[0.06,0.20]"
               "data.radius_range=[0.11,0.15]") ;;
  accel)
    CACHE="moving_ball_scene_accel2d"
    CONFIG="configs/train/moving_ball_scene_decoder.yaml"
    DATA_ARGS=(data.scenario=scene_accel2d
               "data.speed_range=[0.008,0.016]"
               "data.accel_range=[0.0015,0.0035]"
               "data.radius_range=[0.08,0.12]") ;;
  angaccel)
    # copied from experiments/threads/angular-velocity/02_encode/extract_angaccel2d.sh (test split):
    # BIG rotor, same radius as the angvel_big set, seed 11. NOT clip-aligned with the 240-clip
    # angaccel2d_test cache unless NUM_CLIPS/SEED match it -- SEED is overridden below for that.
    CACHE="moving_ball_scene_angaccel2d"
    CONFIG="configs/train/moving_ball_scene_angvel_decoder.yaml"
    SEED=${ANGACCEL_SEED:-11}
    DATA_ARGS=(data.scenario=scene_angaccel2d
               "data.radius_range=[0.32,0.42]"
               "data.omega0_range=[-0.06,0.06]"
               "data.alpha_range=[0.005,0.014]") ;;
  *) echo "unknown FAMILY=$FAMILY (velocity|angvel|accel|angaccel)"; exit 1 ;;
esac

BASE_DIR=${BASE_DIR:-"."}
# _act_sz / _res_sz keeps the 32-clip size-sweep caches from colliding with the 800-clip published
# ones and with each other. CACHE_ROOT lets a re-derivable cache live on scratch (project is at 97%).
CACHE_ROOT=${CACHE_ROOT:-"${BASE_DIR}/outputs/latents"}
OUTPUT_DIR=${OUTPUT_DIR:-"${CACHE_ROOT}/${CACHE}${SFX}/${SPLIT}/vjepa2_${SIZE}"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[size_extract] FAMILY=$FAMILY SIZE=$SIZE TAP=$TAP ENCODER=$ENCODER LAYERS=$LAYERS SITE=$SITE"
echo "[size_extract] NUM_CLIPS=$NUM_CLIPS SEED=$SEED BS=$BS -> $OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder "$ENCODER" \
    --layers "$LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BS" \
    --shard_size 128 \
    encoder.hook_site=$SITE \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "${DATA_ARGS[@]}"
STATUS=$?

echo "[size_extract] done (exit $STATUS): $FAMILY/$SIZE -> $OUTPUT_DIR"
exit $STATUS
