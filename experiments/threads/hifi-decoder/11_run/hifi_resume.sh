#!/bin/bash
#SBATCH --job-name=hifi_resume
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hifi_resume_%j.out
#SBATCH --error=logs/hifi_resume_%j.err

# The 16k-step schedule needs ~20h at the measured 13.7 steps/min, but the queue caps us at 12h.
# This picks up from the newest checkpoint and keeps going; chain as many as the schedule needs.
# It is a no-op (exit 0) once the run has reached max_steps, so an over-long chain costs nothing.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
NAME=${NAME:-v2d_mixed}
CFG="configs/train/hifi_${NAME}_decoder.yaml"
CKDIR="outputs/runs/hifi_${NAME}_decoder/checkpoints"

# Prefer last.pt (written on a clean finish); otherwise the highest-numbered step_N.pt.
CK="$CKDIR/last.pt"
if [ ! -f "$CK" ]; then
    CK=$(ls -1 "$CKDIR"/step_*.pt 2>/dev/null | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
fi
if [ -z "${CK:-}" ] || [ ! -f "$CK" ]; then
    echo "[hifi_resume:$NAME] no checkpoint in $CKDIR -- nothing to resume"; exit 1
fi

# max_steps lives under `optim`, not `train` -- reading it from the wrong node yields a traceback,
# not an error, so both values are checked for digits before they are compared as numbers.
MAXS=$(python -c "import sys;sys.path.insert(0,'.');from src.utils.config import load_config;print(load_config('$CFG').optim.max_steps)" 2>/dev/null | tail -1)
HAVE=$(python -c "import torch;print(int(torch.load('$CK',map_location='cpu',weights_only=False).get('step',0)))" 2>/dev/null | tail -1)
echo "[hifi_resume:$NAME] $CK @ step ${HAVE:-?} / ${MAXS:-?}"
case "${MAXS:-}" in ''|*[!0-9]*) echo "[hifi_resume:$NAME] bad max_steps '${MAXS:-}'"; exit 1;; esac
case "${HAVE:-}" in ''|*[!0-9]*) echo "[hifi_resume:$NAME] bad step '${HAVE:-}'"; exit 1;; esac
if [ "$HAVE" -ge "$MAXS" ]; then
    echo "[hifi_resume:$NAME] already at max_steps -- no-op"; exit 0
fi

# Project storage sits at ~95% of the group quota and each checkpoint is 3.4GB, so drop the older
# step_N.pt files before this job writes another 12h worth. The one we resume from is kept.
for stale in $(ls -1 "$CKDIR"/step_*.pt 2>/dev/null); do
    if [ "$stale" != "$CK" ]; then
        echo "[hifi_resume:$NAME] pruning $stale"
        rm -f "$stale"
    fi
done

python experiments/pipeline/03_train/train_decoder.py --config "$CFG" "train.resume=$CK"
