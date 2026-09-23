#!/bin/bash
#SBATCH --job-name=actpred
# tightest member, so including it would reject the job outright. day allows what this needs.
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/actpred_%j.out
#SBATCH --error=logs/actpred_%j.err

# Train the inverse action predictor and score it by executing what it commands.
#
#   EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/03_train/train_action_predictor.sh
#   EMBODIMENT=franka sbatch experiments/threads/paddle-robotics/03_train/train_action_predictor.sh
#   TRANSFER=1        sbatch experiments/threads/paddle-robotics/03_train/train_action_predictor.sh   # fit paddle, test franka
#
# NO GPU, deliberately. The design matrix is a few thousand rows of ~200 columns, so the MLP trains
# in minutes on CPU threads -- asking for a GPU would add queue time behind jobs that actually need
# one and buy nothing. simulate(render=False) never builds a renderer either.
#
# 96G is set by the PCA, not the predictor: the Gram step needs the 4000 x 524288 float32 train-post
# descriptors (8.4 GB) plus a mean-centred copy, and the other three splits on top. Run smaller and
# it is SIGKILLed rather than slowed.

EMBODIMENT=${EMBODIMENT:-"paddle"}
TRANSFER=${TRANSFER:-0}
K=${K:-64}
POOL=${POOL:-4}
MAX_TRAIN=${MAX_TRAIN:-4000}
FAMILIES=${FAMILIES:-"ridge,mlp"}
SOURCES=${SOURCES:-"real,synth,both"}
MEMBERS=${MEMBERS:-5}

BASE=${BASE:-"outputs/paddle_strike"}
LAT="${BASE}/latents"

module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# FOUR threads, not one-per-core, and PASSIVE waiting. Both matter and the second one is the trap:
# OpenMP's default wait policy spins, so on a shared node the idle threads burn CPU fighting the
# other jobs for cores instead of yielding. The franka run drew 12 busy cores for 2h50m and produced
# nothing while training an MLP that takes ~2 minutes -- AveCPU said "working hard", the log said
# "no progress", and the difference was entirely spin-wait. The design matrix here is a few thousand
# rows of ~200 columns, so 4 threads is already past the point where more helps.
export OMP_NUM_THREADS=${TORCH_THREADS:-4}
export MKL_NUM_THREADS=${TORCH_THREADS:-4}
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

EXTRA=""
if [ "$TRANSFER" = "1" ]; then
  OUT="${BASE}/action_predictor/transfer_paddle_to_franka"
  EXTRA="--transfer_root ${LAT}/franka"
  ROOT="${LAT}/paddle"
else
  OUT="${BASE}/action_predictor/${EMBODIMENT}"
  ROOT="${LAT}/${EMBODIMENT}"
fi

# EXEC=franka_dynamic runs the ACTUATED arm as the executor while perception stays on the kinematic
# rendering. That split is mandatory, not a convenience: the dynamic variant starts its swing at
# frame 4, inside the context window, so perceiving it would leak the action.
if [ -n "$EXEC" ]; then
  OUT="${OUT}_exec_${EXEC}"
  # 2.9 s/rollout against the kinematic 0.12 s, so the calibration sweep is capped rather than run
  # over all 50 val scenes.
  EXTRA="$EXTRA --exec_embodiment ${EXEC} --cal_scenes ${CAL_SCENES:-12}"
  EXTRA="$EXTRA --max_test_scenes ${MAX_TEST_SCENES:-25}"
fi

echo "[actpred] embodiment=$EMBODIMENT transfer=$TRANSFER families=$FAMILIES sources=$SOURCES"
echo "[actpred] latents=$ROOT -> $OUT"
PYTHONPATH=. python experiments/threads/paddle-robotics/03_train/train_action_predictor.py \
    --latent_root "$ROOT" \
    --output_dir "$OUT" \
    --feat_cache "${BASE}/action_loop/featcache" \
    --embodiment "$EMBODIMENT" \
    --k "$K" --pool "$POOL" --max_train_post "$MAX_TRAIN" \
    --families "$FAMILIES" --target_sources "$SOURCES" --n_members "$MEMBERS" $EXTRA
