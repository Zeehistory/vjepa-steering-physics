#!/bin/bash
#SBATCH --job-name=hifi_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=logs/hifi_probe_%j.out
#SBATCH --error=logs/hifi_probe_%j.err

# Mid-flight fidelity read on the newest checkpoint, so we know whether the convup head is actually
# tracking toward the 30dB / LPIPS<0.05 target instead of waiting ~5h to find out. Read-only w.r.t.
# the live run: it only reads step_6000.pt and writes JSON to a probe/ dir.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=. MUJOCO_GL=egl
FS=outputs/filmstrip
H=outputs/hifi
mkdir -p "$H/probe"

SPECS=("v2d_mixed|moving_ball_scene_v2d_mixed" "rb3d|rolling_ball3d")
for spec in "${SPECS[@]}"; do
    IFS='|' read -r NAME DSET <<< "$spec"
    CKDIR="outputs/runs/hifi_${NAME}_decoder/checkpoints"
    CK=$(ls -1 "$CKDIR"/step_*.pt 2>/dev/null | sed 's/.*step_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "${CK:-}" ] && [ -f "$CK" ] || { echo "[probe] no ckpt for $NAME"; continue; }
    STEP=$(basename "$CK" .pt)
    echo "=============== $NAME @ $STEP ==============="
    for MODE in raw ema; do
        EMA=""; [ "$MODE" = "ema" ] && EMA="--use_ema"
        python experiments/pipeline/07_eval/eval_decoder_fidelity.py --label "probe_${NAME}_${STEP}_${MODE}" \
            --config "configs/train/hifi_${NAME}_decoder.yaml" \
            --test_dir "$FS/latents/$DSET/test/vjepa2_large" \
            --checkpoint "$CK" $EMA --out "$H/probe/${NAME}_${STEP}_${MODE}.json" 2>&1 | tail -4
    done
done
echo "[hifi_probe] done"
