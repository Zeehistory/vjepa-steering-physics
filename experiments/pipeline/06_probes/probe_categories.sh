#!/bin/bash
#SBATCH --job-name=vjepa_catprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/catprobe_%j.out
#SBATCH --error=logs/catprobe_%j.err

# Step 1: category-subspace probing on Physics-IQ latents (CPU-bound sklearn; GPU not required but the
# latent cache is large, so we keep the node's memory). Set LATENT_DIR / OUTPUT_DIR or use defaults.
#
# Runs the probe TWICE for a clean comparison (per the z-score question):
#   * standardized/ : StandardScaler z-scoring (maximizes classification performance)
#   * raw/          : --no_standardize (raw latent values; truer subspace geometry, no per-dim rescaling)
# Both restrict to the 3 viable categories (solid_mechanics, fluid_dynamics, optics); thermodynamics and
# magnetism are dropped explicitly (too few scenarios for valid scenario-grouped CV).
LATENT_DIR=${LATENT_DIR:-"outputs/latents/physics_iq/vjepa2_large"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/analysis/physics_iq/category_probe"}
CATEGORIES=${CATEGORIES:-"solid_mechanics,fluid_dynamics,optics"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "== standardized (z-score) probe =="
python experiments/pipeline/06_probes/probe_categories.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR/standardized" \
    --categories "$CATEGORIES" \
    --layers all

echo "== raw (no z-score) probe =="
python experiments/pipeline/06_probes/probe_categories.py \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR/raw" \
    --categories "$CATEGORIES" \
    --no_standardize \
    --layers all
