#!/bin/bash
#SBATCH --job-name=angvel_vitg_extract
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/angvel_vitg_extract_%j.out
#SBATCH --error=logs/angvel_vitg_extract_%j.err
# LARGER-MODEL robustness: encode a SMALL big-object angular-velocity set with V-JEPA2 ViT-g (1408-dim,
# ~2x ViT-L params) to test whether the Fourier-in-orientation command->dH structure PERSISTS in a bigger
# backbone (decoder-free LATENT GATE only -- no ViT-g decoder). Kept small + TRANSIENT (deleted after the
# gate) because the 4TB group quota is ~full. SPLIT/SEED/NUM set by env.
NUM_CLIPS=${NUM_CLIPS:-256}
SEED=${SEED:-7}
SPLIT=${SPLIT:-train}
BASE_DIR=.
OUTPUT_DIR=${BASE_DIR}/outputs/latents/moving_ball_scene_angvel2d_vitg_${SPLIT}/test/vjepa2_giant
CONFIG=configs/train/moving_ball_scene_angvel_decoder.yaml
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
echo "[angvel_vitg] SPLIT=$SPLIT NUM=$NUM_CLIPS SEED=$SEED -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_giant \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --shard_size 64 \
    data.scenario=scene_angvel2d \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.omega_range=[0.06,0.20]" \
    "data.radius_range=[0.32,0.42]"
echo "[angvel_vitg] done (exit $?) -> $OUTPUT_DIR"
