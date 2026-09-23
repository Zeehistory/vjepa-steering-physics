#!/bin/bash
#SBATCH --job-name=pstrike_loop
# cpu=1000,mem=15000G.
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pstrike_loop_%j.out
#SBATCH --error=logs/pstrike_loop_%j.err

# Close the action loop, on a compute node.
#
#   EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/11_run/close_action_loop.sh
#   EMBODIMENT=franka sbatch experiments/threads/paddle-robotics/11_run/close_action_loop.sh
#   TRANSFER=1        sbatch experiments/threads/paddle-robotics/11_run/close_action_loop.sh   # fit paddle, test franka
#
# NO GPU. simulate(render=False) never builds a renderer and the fits are numpy, so this is pure CPU --
# asking for a GPU would only queue behind jobs that need one.
#
# The memory is the whole reason this is a batch job. Run on an 8 GB interactive allocation it was
# SIGKILLed twice: the pooled descriptors alone are ~3.4 GB for train post plus ~1.9 GB for the other
# three splits, and the Gram step needs a centred copy on top. 96 GB covers the full 4000-clip train
# split with room for the copy, so the reduction never has to be coarsened to fit the node.

EMBODIMENT=${EMBODIMENT:-"paddle"}
TRANSFER=${TRANSFER:-0}
K=${K:-64}
POOL=${POOL:-4}
MAX_TRAIN=${MAX_TRAIN:-4000}

BASE=${BASE:-"outputs/paddle_strike"}
LAT="${BASE}/latents"

module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

EXTRA=""
if [ "$TRANSFER" = "1" ]; then
  OUT="${BASE}/action_loop/transfer_paddle_to_franka"
  EXTRA="--transfer_root ${LAT}/franka"
  ROOT="${LAT}/paddle"
else
  OUT="${BASE}/action_loop/${EMBODIMENT}"
  ROOT="${LAT}/${EMBODIMENT}"
fi

echo "[pstrike_loop] embodiment=$EMBODIMENT transfer=$TRANSFER k=$K pool=$POOL max_train=$MAX_TRAIN"
echo "[pstrike_loop] latents=$ROOT -> $OUT"
python experiments/threads/paddle-robotics/11_run/close_action_loop.py --lam_grid "${LAM_GRID:-0.1,1.0,10.0,100.0,300.0,1000.0,3000.0,10000.0,30000.0}" \
    --latent_root "$ROOT" \
    --output_dir "$OUT" \
    --feat_cache "${BASE}/action_loop/featcache" \
    --embodiment "$EMBODIMENT" \
    --k "$K" --pool "$POOL" --max_train_post "$MAX_TRAIN" $EXTRA
STATUS=$?
echo "[pstrike_loop] done (exit $STATUS) -> $OUT"
exit $STATUS
