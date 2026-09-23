#!/bin/bash
#SBATCH --job-name=hifi_post
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hifi_post_%j.out
#SBATCH --error=logs/hifi_post_%j.err

# After the high-fidelity decoders train: score them against the old ones (raw weights AND the EMA
# shadow, so we can pick), then re-dump the filmstrip frames with whichever decoder we shipped.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=. MUJOCO_GL=egl
FS=outputs/filmstrip
H=outputs/hifi
AN=outputs/analysis

# name | latent dataset dir | analysis dir | quantity | gain
SPECS=(
"v2d_mixed|moving_ball_scene_v2d_mixed|moving_ball_v2d_mixed|velocity|2.5"
"rb3d|rolling_ball3d|rolling_ball3d|velocity|2.0"
)

for spec in "${SPECS[@]}"; do
    IFS='|' read -r NAME DSET ANDIR QUANT GAIN <<< "$spec"
    CFG="configs/train/hifi_${NAME}_decoder.yaml"
    # last.pt only exists on a clean finish; after a wall-clock kill the newest step_N.pt is the run.
    CKDIR="outputs/runs/hifi_${NAME}_decoder/checkpoints"
    CK="$CKDIR/last.pt"
    if [ ! -f "$CK" ]; then
        CK=$(ls -1 "$CKDIR"/step_*.pt 2>/dev/null | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    fi
    if [ -z "${CK:-}" ] || [ ! -f "$CK" ]; then
        echo "[hifi_post] no checkpoint for $NAME -- skipping"; continue
    fi
    TEST="$FS/latents/$DSET/test/vjepa2_large"
    echo "=============== $NAME  ($CK) ==============="
    for MODE in raw ema; do
        EMA=""; [ "$MODE" = "ema" ] && EMA="--use_ema"
        python experiments/pipeline/07_eval/eval_decoder_fidelity.py --label "hifi_${NAME}_${MODE}" \
            --config "$CFG" --test_dir "$TEST" --checkpoint "$CK" $EMA \
            --out "$H/fidelity/hifi_${NAME}_${MODE}.json"
    done
done
echo "[hifi_post] fidelity done -> $H/fidelity"
