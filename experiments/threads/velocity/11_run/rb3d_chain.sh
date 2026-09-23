#!/bin/bash
# Driver for the 3D rolling-ball steering pipeline.
#
# user (the devel partition MaxSubmitJobsPU=1), so a dependency chain cannot be pre-queued there -- and
# and preempted the v2d_mixed decoder at the finish line last time). So: submit one stage at a time,
# wait, submit the next. CPU stages go to bigmem (also instant) and overlap the decoder train.
#
#   nohup experiments/threads/velocity/11_run/rb3d_chain.sh > logs/rb3d_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
BASE=.
LAT=${LAT_BASE:-.}/outputs/latents/rolling_ball3d
say() { echo "[chain $(date +%H:%M:%S)] $*"; }

gpu_submit() {
    local out
    while true; do
        out=$("$@" 2>&1)
        if echo "$out" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
            sleep 60; continue
        fi
        echo "$out" | tail -1; return 0
    done
}
wait_for() {   # wait_for <jobid>
    while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done
    sacct -n -j "$1" --format=State 2>/dev/null | head -1 | tr -d ' '
}

# -- 1. encode test (800 clips ~35G), then train (4000 clips ~184G) -- both on SCRATCH -----------
for SPLIT in test train; do
    if [ -d "$LAT/$SPLIT/vjepa2_large" ] && [ -n "$(ls -A $LAT/$SPLIT/vjepa2_large 2>/dev/null)" ]; then
        say "encode $SPLIT already present, skipping"; continue
    fi
    J=$(SPLIT=$SPLIT gpu_submit sbatch --parsable --partition=gpu_devel experiments/threads/velocity/02_encode/extract_rb3d.sh)
    say "encode $SPLIT -> job $J"
    S=$(wait_for "$J"); say "encode $SPLIT finished: $S"
    case "$S" in COMPLETED) ;; *) say "ABORT: encode $SPLIT failed ($S) — see logs/rb3d_extract_$J.err"; exit 1;; esac
done

# -- 2. decoder train (gpu_devel, long pole) + subspace (bigmem, parallel) ------------------------
PART=$(experiments/pipeline/00_common/pick_partition.sh 256000 bigmem mpi week day)
JSUB=$(sbatch --parsable --partition="$PART" --mem=256G experiments/threads/velocity/04_operators/subspace_rb3d.sh)
say "subspace -> job $JSUB on $PART (runs in parallel with the decoder)"

JTR=$(gpu_submit sbatch --parsable --partition=gpu_devel experiments/threads/velocity/03_train/train_rb3d_fp.sh)
say "decoder train -> job $JTR"

S=$(wait_for "$JSUB"); say "subspace finished: $S"
case "$S" in COMPLETED) ;; *) say "ABORT: subspace failed ($S)"; exit 1;; esac

# -- 3. command operator fit (bigmem; needs the subspace basis) -----------------------------------
PART=$(experiments/pipeline/00_common/pick_partition.sh 256000 bigmem mpi week day)
JFIT=$(sbatch --parsable --partition="$PART" --mem=256G experiments/threads/velocity/04_operators/fit_command_rb3d.sh)
say "cmd fit -> job $JFIT on $PART"
S=$(wait_for "$JFIT"); say "cmd fit finished: $S"
case "$S" in COMPLETED) ;; *) say "ABORT: cmd fit failed ($S)"; exit 1;; esac

S=$(wait_for "$JTR"); say "decoder train finished: $S"
CK=${LAT_BASE:-.}/outputs/runs/rolling_ball3d_decoder_fp/checkpoints/last.pt
[ -f "$CK" ] || { say "ABORT: no decoder checkpoint at $CK"; exit 1; }

# -- 4. steer + leakage-free gain calibration -----------------------------------------------------
JST=$(gpu_submit sbatch --parsable --partition=gpu_devel experiments/threads/velocity/05_steering/steer_rb3d.sh last)
say "steer -> job $JST"
S=$(wait_for "$JST"); say "steer finished: $S"
say "RESULT: $BASE/outputs/analysis/rolling_ball3d/steer_last/calib_cmd_gain.json"
cat "$BASE/outputs/analysis/rolling_ball3d/steer_last/calib_cmd_gain.json" 2>/dev/null
