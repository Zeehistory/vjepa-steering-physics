#!/bin/bash
#SBATCH --job-name=pstrike_extract
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pstrike_extract_%j.out
#SBATCH --error=logs/pstrike_extract_%j.err

# Extract all four VJEPA2-L latent caches the action-conditioned loop needs, in ONE job.
#
#   PILOT=1 sbatch experiments/threads/paddle-robotics/02_encode/extract_paddle_strike.sh   # 64/8 clips, minutes -- validates
#           sbatch experiments/threads/paddle-robotics/02_encode/extract_paddle_strike.sh   # the full 4000/500/800/100
#
# interactive session already holds one, so four parallel submissions fail outright with
# one-off costs -- conda import, HF weight load -- across all four caches instead of paying them
# four times.
#
# Submit a PARTITION LIST, never a single -p gpu, so the scheduler places the job wherever a GPU frees
# account, and including either makes the whole job sit in PENDING/(PartitionConfig) rather than
# falling through to a partition that would have run it.
#
# EMBODIMENT=franka extracts the arm rendering for the appearance-transfer arm of the study.

EMBODIMENT=${EMBODIMENT:-"paddle"}
PILOT=${PILOT:-0}
BASE=${BASE:-"outputs/paddle_strike/latents"}

if [ "$PILOT" = "1" ]; then
  # 64 post clips = 8 complete scenes, so scene pairing and the 8-action sweep are both exercised --
  # a smaller pilot would validate only the forward pass, which is not the part that breaks.
  SPEC="train:post:64:0 train:pre:8:0 test:post:16:2 test:pre:2:2"
  ROOT="${BASE}/pilot/${EMBODIMENT}"
else
  SPEC="train:post:4000:0 train:pre:500:0 test:post:800:2 test:pre:100:2"
  ROOT="${BASE}/${EMBODIMENT}"
fi

module purge
module load miniconda
conda activate vjepa-physics-decoder

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# The dataset renders offscreen on the GPU node. paddle_strike sets this at import, but DataLoader
# workers inherit the environment, so setting it here removes any question of ordering.
export MUJOCO_GL=egl

OVERALL=0
for entry in $SPEC; do
  IFS=':' read -r SPLIT WINDOW CLIPS SEED <<< "$entry"
  case "$WINDOW" in
    post) SCENARIO=scene_paddle_strike ;;
    pre)  SCENARIO=scene_paddle_strike_pre ;;
  esac
  OUT="${ROOT}/${SPLIT}_${WINDOW}"
  echo "=============================================================="
  echo "[pstrike_extract] $EMBODIMENT $SPLIT/$WINDOW scenario=$SCENARIO clips=$CLIPS seed=$SEED"
  echo "[pstrike_extract] -> $OUT"
  python experiments/pipeline/02_encode/extract_latents.py \
      --config configs/train/paddle_strike_latents.yaml \
      --encoder vjepa2_large \
      --layers 6,12,18,23 \
      --output_dir "$OUT" \
      --batch_size 8 \
      --shard_size 128 \
      data.scenario=$SCENARIO \
      data.num_clips=$CLIPS \
      data.seed=$SEED \
      data.split=$SPLIT \
      data.embodiment=$EMBODIMENT
  ST=$?
  echo "[pstrike_extract] $SPLIT/$WINDOW exit=$ST"
  # Keep going on failure so one bad cache does not hide the state of the other three, but remember it.
  [ $ST -ne 0 ] && OVERALL=$ST
done

echo "[pstrike_extract] ALL DONE (worst exit $OVERALL) -> $ROOT"
exit $OVERALL
