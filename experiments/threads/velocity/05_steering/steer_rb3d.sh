#!/bin/bash
#SBATCH --job-name=rb3d_steer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rb3d_steer_%j.out
#SBATCH --error=logs/rb3d_steer_%j.err
# Decode + track the 3D rolling-ball steer: full_delta / subspace_U[k] / random[k] /
# ridge_global / cmd_U8 (gain sweep) / ridge_rich on held-out test scenes. CMD_KU selects U8 (default)
# or the U16 operator. ALWAYS submit via the queue-aware picker:
#   sbatch -p <partition> experiments/threads/velocity/05_steering/steer_rb3d.sh [CKPT] [OUTTAG]
#
# EDIT_LAYERS restricts which encoder layer the edit is WRITTEN to; the decoder still reads all four.
# Empty (default) is the historical all-layer write. This is what makes the read-locus/write-locus
# sweep possible -- one job per layer:
#   for L in 6 12 18 23; do
#     EDIT_LAYERS=$L sbatch -p <partition> experiments/threads/velocity/05_steering/steer_rb3d.sh last writelayer_L$L
#   done
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
LAT_BASE=${LAT_BASE:-.}  # latents on scratch (see slurm_extract_rb3d.sh)
# CK and OUTTAG are POSITIONAL ($1, $2), not environment variables -- passing them as env vars is
# silently ignored (the assignments below overwrite them) and the job then re-runs the DEFAULT
# decoder into steer_last, overwriting it. That happened once; hence this note.
CK="${1:-last}"
OUTTAG="${2:-$CK}"
# RUN selects the decoder run directory, and IS an env var. Use it for the decoder-independence
# test, where the point is to score the same edits through a different decoder:
#   RUN=rolling_ball3d_decoder_fp_seed1 sbatch ... experiments/threads/velocity/05_steering/steer_rb3d.sh last decoderB
RUN=${RUN:-rolling_ball3d_decoder_fp}
CKPT=$LAT_BASE/outputs/runs/${RUN}/checkpoints/${CK}.pt
if [ ! -f "$CKPT" ]; then echo "[rb3d_steer] FAIL: no checkpoint at $CKPT"; exit 2; fi
echo "[rb3d_steer] RUN=$RUN CK=$CK OUTTAG=$OUTTAG CKPT=$CKPT"
python -u experiments/threads/velocity/05_steering/steer_velocity2d.py \
    --config configs/train/rolling_ball3d_decoder.yaml \
    --test_dir $LAT_BASE/outputs/latents/rolling_ball3d/test/vjepa2_large \
    --artifacts_dir $BASE/outputs/analysis/rolling_ball3d/subspace \
    --checkpoint "$CKPT" \
    --output_dir $BASE/outputs/analysis/rolling_ball3d/steer_${OUTTAG} \
    --ks 2,4,8,16 --num_scenes ${NUM_SCENES:-100} --cmd_scales "${CMD_SCALES:-1.0,1.5,2.0,2.5,3.0}" \
    --cmd_ku ${CMD_KU:-8} ${CMD_POS:+--cmd_pos} --dir_bins "" \
    --edit_layers "${EDIT_LAYERS:-}" \
    --device cuda
RC=$?
echo "[rb3d_steer] steer exit=$RC ckpt=$CKPT out=steer_${OUTTAG}"

# Leakage-free held-out gain calibration: pick cmd_U8 gain on a val split, report on a disjoint test
# split. Writes calib_cmd_gain.json next to the steer summary so the held-out number self-produces.
SUM=$BASE/outputs/analysis/rolling_ball3d/steer_${OUTTAG}/steer2d_summary.json
if [ -f "$SUM" ]; then
    python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary "$SUM" --val_frac 0.5 \
        --out $BASE/outputs/analysis/rolling_ball3d/steer_${OUTTAG}/calib_cmd_gain.json
fi
echo "[rb3d_steer] done (exit $RC)"
exit $RC
