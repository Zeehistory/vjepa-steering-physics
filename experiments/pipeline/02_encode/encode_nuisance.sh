#!/bin/bash
#SBATCH --job-name=nuis_enc
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/nuis_enc_%j.out
#SBATCH --error=logs/nuis_enc_%j.err
# Encode the TRAIN caches for the colour + background nuisance variants (size is already cached). Only
# train is needed for the disentanglement subspace angles. One job, two encodes (respects gpu_devel cap=1
# if run there). Skips an encode whose cache already has shards.
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
CONFIG=configs/train/moving_ball_scene_decoder.yaml
RC=0
for SCN in scene_color scene_background; do
    TAG=${SCN#scene_}
    OUT=$BASE/outputs/latents/moving_ball_scene_${TAG}/train/vjepa2_large
    if ls "$OUT"/*.tar >/dev/null 2>&1; then echo "[nuis_enc] $SCN train already cached, skip"; continue; fi
    echo "[nuis_enc] encoding $SCN train -> $OUT"
    python experiments/pipeline/02_encode/extract_latents.py --config "$CONFIG" --encoder vjepa2_large --layers 6,12,18,23 \
        --output_dir "$OUT" --batch_size 8 --shard_size 128 \
        data.scenario=$SCN data.num_clips=2000 data.seed=0 "data.radius_range=[0.11,0.11]"
    S=$?; echo "[nuis_enc] $SCN exit=$S"; [ $S -ne 0 ] && RC=$S
done
echo "[nuis_enc] done RC=$RC"; exit $RC
