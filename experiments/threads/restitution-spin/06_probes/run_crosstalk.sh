#!/bin/bash
#SBATCH --job-name=spin_xtalk
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/spin_xtalk_%j.out
#SBATCH --error=logs/spin_xtalk_%j.err

# The crosstalk answer, end to end: gate the decoder, then measure, then summarise.
#
# The GATE RUNS FIRST AND IS ALLOWED TO STOP THE PIPELINE. Nothing in the decoder's loss constrains
# the marker's phase, so it can smear the marker into a phase-averaged ring while its loss falls --
# and measured_spin would still return a number (a random slope through noise). That would read as
# "spin steering does not work" when the truth is "spin was never rendered". Those are opposite
# conclusions, so the gate is a hard precondition rather than a diagnostic printed alongside.
#
#   sbatch --dependency=afterok:<train_job> experiments/threads/restitution-spin/06_probes/run_crosstalk.sh
#   CKPT=step_4000 sbatch experiments/threads/restitution-spin/06_probes/run_crosstalk.sh

module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export PYTHONPATH=.
export MUJOCO_GL=egl        # the decode path renders nothing, but the analysis imports the scene module

LAT_BASE=${LAT_BASE:-.}
ANA=${ANA:-outputs/analysis/spin_ball3d}
CKPT_TAG=${CKPT:-last}
CKPT=$LAT_BASE/outputs/runs/spin_ball3d_decoder/checkpoints/${CKPT_TAG}.pt
TEST=$LAT_BASE/outputs/latents/spin_ball3d/test/vjepa2_large
CONFIG=configs/train/spin_ball3d_decoder.yaml

echo "[xtalk] ckpt=$CKPT"
test -f "$CKPT" || { echo "FAIL: checkpoint missing"; exit 1; }
test -s "$ANA/operators/operator_both.npz" || { echo "FAIL: operators missing (run _fit_ops.sh)"; exit 1; }

python -u experiments/threads/restitution-spin/06_probes/decoder_gate.py \
    --config "$CONFIG" --test_dir "$TEST" --checkpoint "$CKPT" \
    --output_dir "$ANA/decoder_gate_${CKPT_TAG}" --num_clips "${GATE_CLIPS:-96}"
GATE=$?
if [ $GATE -ne 0 ]; then
    echo "[xtalk] DECODER GATE FAILED (rc=$GATE) -- refusing to report steering numbers measured"
    echo "[xtalk] through an instrument that cannot see the quantity. See decoder_gate.json."
    exit $GATE
fi

python -u experiments/threads/restitution-spin/06_probes/crosstalk_eval.py \
    --config "$CONFIG" --test_dir "$TEST" \
    --operators_dir "$ANA/operators" --checkpoint "$CKPT" \
    --output_dir "$ANA/crosstalk_${CKPT_TAG}" \
    --num_scenes "${NUM_SCENES:-64}" --squares_per_scene "${SQ:-4}" ${SKIP_ORDER:+--skip_order}
RC=$?
test $RC -eq 0 || { echo "[xtalk] crosstalk_eval failed rc=$RC"; exit $RC; }

python -u experiments/threads/restitution-spin/06_probes/crosstalk_summary.py \
    --raw "$ANA/crosstalk_${CKPT_TAG}/crosstalk_raw.json" \
    --out "$ANA/crosstalk_${CKPT_TAG}/crosstalk_summary.json"

echo "[xtalk] done -> $ANA/crosstalk_${CKPT_TAG}"
