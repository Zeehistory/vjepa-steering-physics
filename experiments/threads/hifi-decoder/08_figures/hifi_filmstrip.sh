#!/bin/bash
#SBATCH --job-name=hifi_filmstrip
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hifi_filmstrip_%j.out
#SBATCH --error=logs/hifi_filmstrip_%j.err

# Re-dump the Real-vs-Decoded-steered frame stacks with the NEW high-fidelity decoders.
# _filmstrip_all.sh hardcodes the OLD *_decoder_fp runs and _hifi_post.sh only scores fidelity,
# so nothing else in the chain actually refreshes the frames.
#
# raw vs EMA is NOT assumed: we read the fidelity JSONs _hifi_post.sh just wrote and take
# whichever variant has the lower mean LPIPS. EMA decay 0.9999 over 16k steps still leaves a
# chunk of the init in the shadow, so it may well lose -- that is the point of checking.
# Gains are the ones each quantity's steer_*.py picked on its disjoint validation half.

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=. MUJOCO_GL=egl

FS=outputs/filmstrip
H=outputs/hifi
AN=outputs/analysis
NS=${NUM_SCENES:-6}

# name | latent dataset dir | analysis dir | quantity | gain
SPECS=(
"v2d_mixed|moving_ball_scene_v2d_mixed|moving_ball_v2d_mixed|velocity|2.5"
"rb3d|rolling_ball3d|rolling_ball3d|velocity|2.0"
)

FAILED=""
for spec in "${SPECS[@]}"; do
    IFS='|' read -r NAME DSET ANDIR QUANT GAIN <<< "$spec"
    CKDIR="outputs/runs/hifi_${NAME}_decoder/checkpoints"
    CK="$CKDIR/last.pt"
    if [ ! -f "$CK" ]; then
        CK=$(ls -1 "$CKDIR"/step_*.pt 2>/dev/null | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    fi
    if [ -z "${CK:-}" ] || [ ! -f "$CK" ]; then
        echo "[hifi_filmstrip] no checkpoint for $NAME -- skipping"; FAILED="$FAILED $NAME"; continue
    fi

    # pick raw or ema by mean LPIPS from the post-eval; default to raw if a JSON is missing.
    EMA_FLAG=$(python - "$H/fidelity/hifi_${NAME}_raw.json" "$H/fidelity/hifi_${NAME}_ema.json" <<'PY'
import json, sys
def lp(p):
    try:
        return json.load(open(p))["summary"]["lpips"]["mean"]
    except Exception:
        return float("inf")
raw, ema = lp(sys.argv[1]), lp(sys.argv[2])
sys.stderr.write(f"  lpips raw={raw} ema={ema}\n")
print("--use_ema" if ema < raw else "")
PY
)
    echo ""
    echo "=================== $NAME (quantity=$QUANT gain=$GAIN ${EMA_FLAG:-raw} $CK) ==================="
    python experiments/pipeline/08_figures/dump_filmstrip.py \
        --config "configs/train/hifi_${NAME}_decoder.yaml" \
        --test_dir "$FS/latents/$DSET/test/vjepa2_large" \
        --artifacts_dir "$AN/$ANDIR/subspace" \
        --checkpoint "$CK" $EMA_FLAG \
        --output_dir "$FS/frames/hifi_$NAME" \
        --quantity "$QUANT" --gain "$GAIN" --num_scenes "$NS" --device cuda
    if [ $? -ne 0 ]; then FAILED="$FAILED $NAME"; fi
done

echo ""
if [ -n "$FAILED" ]; then echo "[hifi_filmstrip] FAILED:$FAILED"; exit 1; fi
echo "[hifi_filmstrip] done -> $FS/frames/hifi_*"
