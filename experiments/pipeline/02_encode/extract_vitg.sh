#!/bin/bash
#SBATCH --job-name=vitg_extract
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=logs/vitg_extract_%j.out
#SBATCH --error=logs/vitg_extract_%j.err
# MODEL-SIZE axis: encode the rotor sets with V-JEPA2 ViT-g (1408-dim, 40 layers) so the Fourier-in-
# orientation result can be tested in a ~2x-larger backbone -- and, unlike the earlier latent-gate-only
# probe, pixel-verified through a ViT-g decoder.
#
# LAYERS ARE MATCHED BY RELATIVE DEPTH, not by index. ViT-g has 40 blocks to ViT-L's 24, so the ViT-L
# indices 6/12/18/23 (25/50/75/96% depth) correspond to 10/20/30/38, NOT to 6/12/18/23 (which would be
# 15/30/45/58% -- a depth confound masquerading as a size comparison).
#
#   QUANTITY=angvel   SPLIT=train -> operator-fit + ViT-g decoder training set
#   QUANTITY=angaccel SPLIT=test  -> the steer target
QUANTITY=${QUANTITY:-angvel}
SPLIT=${SPLIT:-train}
if [ "$SPLIT" = "train" ]; then NUM_CLIPS=${NUM_CLIPS:-1000}; SEED=${SEED:-7};
else NUM_CLIPS=${NUM_CLIPS:-240}; SEED=${SEED:-11}; fi
BASE_DIR=.
NAME=${NAME:-moving_ball_scene_${QUANTITY}2d_vitg_${SPLIT}}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/latents/${NAME}/test/vjepa2_giant}
CONFIG=configs/train/moving_ball_scene_angvel_decoder.yaml
LAYERS=${LAYERS:-10,20,30,38}
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
case "$QUANTITY" in
  angvel)   SCEN=scene_angvel2d;   EXTRA=("data.omega_range=[0.06,0.20]") ;;
  angaccel) SCEN=scene_angaccel2d; EXTRA=("data.omega0_range=[-0.06,0.06]" "data.alpha_range=[0.005,0.014]") ;;
esac
echo "[vitg] QUANTITY=$QUANTITY SPLIT=$SPLIT NUM=$NUM_CLIPS SEED=$SEED LAYERS=$LAYERS -> $OUTPUT_DIR"
df -h . | tail -1
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_giant \
    --layers "$LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --shard_size 64 \
    data.scenario=$SCEN \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.radius_range=[0.32,0.42]" \
    "${EXTRA[@]}"
STATUS=$?
df -h . | tail -1
echo "[vitg] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
