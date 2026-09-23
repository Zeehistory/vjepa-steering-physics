#!/bin/bash
#SBATCH --job-name=vjepa_catsteer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/catsteer_%j.out
#SBATCH --error=logs/catsteer_%j.err

# Step 3 (category steering): steer real fluid Physics-IQ clips toward "solid" along (w_solid - w_fluid)
# and decode the result. Requires:
#   * the category probe directions (slurm_probe_categories.sh -> category_directions.npz)
#   * Physics-IQ latents and a trained transformer decoder.
#
# Defaults reflect the Phase-A/B/C findings. Steering only the deepest layer's probe weights moved the
# readout but not the pixels (decoder ignored the edit until alpha broke the decode into mush), so the
# defaults now use METHOD=diff_means (class-centroid translation, decoder-renderable) + ALL_LAYERS=1
# (steer every cached layer). ALPHAS are a FRACTION of each layer's per-token norm (1.0 == one token
# norm), so the sweep is comparable across layers. Set METHOD=probe / ALL_LAYERS=0 to reproduce the
# discriminative single-layer variant. For METHOD=probe, DIRECTIONS must point at category_directions.npz.
TARGET_LATENT_DIR=${TARGET_LATENT_DIR:-"outputs/latents/physics_iq/vjepa2_large"}
DIRECTIONS=${DIRECTIONS:-"outputs/analysis/physics_iq/category_probe/raw/category_directions.npz"}
CHECKPOINT=${CHECKPOINT:-"outputs/runs/physics_iq_decoder_large/checkpoints/last.pt"}
METHOD=${METHOD:-"diff_means"}
ALL_LAYERS=${ALL_LAYERS:-"1"}
FROM_CATEGORY=${FROM_CATEGORY:-"fluid_dynamics"}
TO_CATEGORY=${TO_CATEGORY:-"solid_mechanics"}
LAYER=${LAYER:-"-1"}
ALPHAS=${ALPHAS:-"0,0.05,0.1,0.15,0.2,0.3"}
NUM_SAMPLES=${NUM_SAMPLES:-"4"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/analysis/steer_category/${FROM_CATEGORY}_to_${TO_CATEGORY}_${METHOD}"}

# --all_layers is a store_true flag; include it only when ALL_LAYERS=1.
ALL_LAYERS_FLAG=""
[ "$ALL_LAYERS" = "1" ] && ALL_LAYERS_FLAG="--all_layers"

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/pipeline/05_steering/steer_category.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --target_latent_dir "$TARGET_LATENT_DIR" \
    --directions "$DIRECTIONS" \
    --checkpoint "$CHECKPOINT" \
    --method "$METHOD" \
    $ALL_LAYERS_FLAG \
    --from_category "$FROM_CATEGORY" \
    --to_category "$TO_CATEGORY" \
    --layer="$LAYER" \
    --alphas="$ALPHAS" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
