#!/bin/bash
#SBATCH --job-name=v2d_transport
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2d_transport_%j.out
#SBATCH --error=logs/v2d_transport_%j.err
# Fit the masked TRAJECTORY-TRANSPORT operator (latent-only, CPU; big RAM for the train shard cache).
# Per (layer, temporal token) ridge of the 6-dim mask-weighted velocity feature -> 1024-dim dH token,
# plus a held-out TEST latent gate (deployable forward-sim + oracle target masks) and a sigma sweep.
#
# ALWAYS pick a non-backlogged partition at submit time: do NOT `sbatch` this directly,
# submit via:
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 192000 mpi priority bigmem day week)
#   sbatch -p <partition> experiments/pipeline/04_operators/fit_transport.sh
# SMOKE=1 runs a 6-scene shakeout first (catches runtime bugs in ~2 min before the full fit).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
TRAIN=$BASE/outputs/latents/moving_ball_scene_v2d/train/vjepa2_large
TEST=$BASE/outputs/latents/moving_ball_scene_v2d/test/vjepa2_large
OUT=$BASE/outputs/analysis/moving_ball_v2d/subspace
LAYERS=${LAYERS:-6,12,18,23}
SIGMAS=${SIGMAS:-0.75,1.0,1.5}

if [ "${SMOKE:-0}" = "1" ]; then
    echo "[v2d_transport] SMOKE: 6-scene shakeout"
    python -u experiments/pipeline/04_operators/fit_transport_operator.py --train_dir "$TRAIN" --test_dir "$TEST" \
        --layers 12 --sigmas 1.0 --output_dir "$OUT/_smoke" --max_scenes 6 || exit 1
    echo "[v2d_transport] smoke OK"
fi

python -u experiments/pipeline/04_operators/fit_transport_operator.py \
    --train_dir "$TRAIN" --test_dir "$TEST" \
    --layers "$LAYERS" --sigmas "$SIGMAS" --ridge 1.0 --output_dir "$OUT"
echo "[v2d_transport] exit=$?"
