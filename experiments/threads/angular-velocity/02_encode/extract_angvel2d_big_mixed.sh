#!/bin/bash
#SBATCH --job-name=angvel_bigmix_extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/angvel_bigmix_extract_%j.out
#SBATCH --error=logs/angvel_bigmix_extract_%j.err
# BIG-OBJECT + MIXED-APPEARANCE angular-velocity TEST set: robustness slice for the polar cmd-U8 result.
# Same big rotor (radius 0.32-0.42) but per-scene randomized bar colour (grey->dark blue, kept far from red
# so the marker stays separable) + background (white->grey). If polar cmd-U8 still steers omega command-only
# here, the method generalizes across appearance. ~240 clips
# = ~11G (fits free disk, no reclaim). SEED distinct from the big train (7) / big test (11).
NUM_CLIPS=${NUM_CLIPS:-240}
SEED=${SEED:-23}
BASE_DIR=${BASE_DIR:-"."}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/outputs/latents/moving_ball_scene_angvel2d_bigmix_test/test/vjepa2_large"}
CONFIG=${CONFIG:-"configs/train/moving_ball_scene_angvel_decoder.yaml"}
RADIUS=${RADIUS:-"[0.32,0.42]"}
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
echo "[angvel_bigmix] NUM_CLIPS=$NUM_CLIPS SEED=$SEED RADIUS=$RADIUS -> $OUTPUT_DIR"
python experiments/pipeline/02_encode/extract_latents.py \
    --config "$CONFIG" \
    --encoder vjepa2_large \
    --layers 6,12,18,23 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --shard_size 128 \
    data.scenario=scene_angvel2d_mixed \
    data.clips_per_scene=8 \
    data.num_clips=$NUM_CLIPS \
    data.seed=$SEED \
    "data.omega_range=[0.06,0.20]" \
    "data.radius_range=$RADIUS"
STATUS=$?
echo "[angvel_bigmix] done (exit $STATUS) -> $OUTPUT_DIR"
exit $STATUS
