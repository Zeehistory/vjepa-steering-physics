#!/bin/bash
#SBATCH --job-name=vjepa_extract_syn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/extract_syn_%j.out
#SBATCH --error=logs/extract_syn_%j.err

# Step 2: extract VJEPA latents for a synthetic-physics dataset with exact GT labels.
# The dataset is generated on the fly on this compute node, so no data.root / download is needed.
#   DATASET=mujoco_solid    sbatch experiments/pipeline/02_encode/extract_synthetic.sh   # MuJoCo rigid-body (recommended)
#   DATASET=genesis_fluid   sbatch experiments/pipeline/02_encode/extract_synthetic.sh   # Genesis fluids (GPU-only)
#   DATASET=synthetic_solid sbatch experiments/pipeline/02_encode/extract_synthetic.sh   # lightweight 2D fallback
#   DATASET=robot_toy       sbatch experiments/pipeline/02_encode/extract_synthetic.sh
#
# One-time engine install into the conda env (run on a compute node, not login):
#   pip install -e .[sim_mujoco]     # MuJoCo
#   pip install -e .[sim_genesis]    # Genesis (CUDA node)
DATASET=${DATASET:-"mujoco_solid"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/latents/$DATASET/vjepa2_large"}

module purge
module load miniconda
conda activate vjepa-physics-decoder

# Headless off-screen rendering for MuJoCo on the cluster (no display). Harmless for other datasets.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python experiments/pipeline/02_encode/extract_latents.py \
    --config configs/train/physics_iq_transformer_large.yaml \
    --dataset "$DATASET" \
    --encoder vjepa2_large \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --shard_size 128
