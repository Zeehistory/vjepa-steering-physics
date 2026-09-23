#!/bin/bash
#SBATCH --job-name=act_decoder
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/act_decoder_%j.out
#SBATCH --error=logs/act_decoder_%j.err

# Train a latent->pixel decoder on the ACTIVATION (mlp.fc2) cache. The STEERING section decodes an
# edited representation into video, and the published decoders were trained on the block-output
# residual stream -- feeding them activations is the wrong input distribution, not a milder version
# of the same thing. So each family gets its own decoder trained on its activation cache.
#
# The config is the PUBLISHED run's own resolved_config.yaml, so architecture, loss weights and
# optimizer are identical to the latent decoder and the ONLY difference is which tap the inputs came
# from. That is what makes the two STEERING panels comparable.
#
#   FAMILY=velocity sbatch experiments/pipeline/03_train/act_decoder.sh
#

FAMILY=${FAMILY:-"velocity"}
SCR=${SCR:-"."}
BASE=${BASE:-"."}

case "$FAMILY" in
  velocity) CACHE="moving_ball_scene_v2d_act";     REF="moving_ball_scene_v2d_decoder_fp" ;;
  angvel)   CACHE="moving_ball_scene_angvel2d_act"; REF="moving_ball_scene_angvel2d_decoder_fp" ;;
  accel)    CACHE="moving_ball_scene_accel2d_act";  REF="moving_ball_scene_accel2d_decoder_fp" ;;
  *) echo "unknown FAMILY=$FAMILY (velocity|angvel|accel)"; exit 1 ;;
esac

LATENT_DIR="$SCR/outputs/latents/$CACHE/train/vjepa2_large"
OUTPUT_DIR="$BASE/outputs/runs/${REF}_act"
CONFIG="$BASE/outputs/runs/$REF/resolved_config.yaml"

module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

[ -f "$LATENT_DIR/extract_meta.json" ] || { echo "[act_decoder] incomplete cache: $LATENT_DIR"; exit 2; }
[ -f "$CONFIG" ] || { echo "[act_decoder] no reference config: $CONFIG"; exit 2; }
echo "[act_decoder] $FAMILY hook_site=$(python -c "import json;print(json.load(open('$LATENT_DIR/extract_meta.json'))['config'].get('hook_site','block'))")"

# The reference config is a RESOLVED config, so it carries that run's own train.resume -- accel's
# points at step_1500.pt of the LATENT run. Inheriting it either crashes (the file is gone) or, far
# worse, silently initializes the activation decoder from latent-trained weights and destroys the
# comparison. So resume is always set explicitly: our own newest checkpoint, or null.
RESUME="train.resume=null"
CKDIR="$OUTPUT_DIR/checkpoints"
if [ -d "$CKDIR" ]; then
    LATEST=$(ls -1t "$CKDIR"/last.pt "$CKDIR"/step_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then RESUME="train.resume=$LATEST"; echo "[act_decoder] resuming from $LATEST"; fi
fi
echo "[act_decoder] $RESUME"

# The 02:54 run died when the project group quota hit 100% mid-write, silently, on all six jobs at
# once. Each checkpoint is 3.3 GB every 500 steps, so an 8000-step run left to itself writes ~53 GB
# of which only the newest is ever used. Reap all but the newest two while training runs.
( while true; do
    sleep 300
    ls -1t "$CKDIR"/step_*.pt 2>/dev/null | tail -n +3 | xargs -r rm -f
  done ) &
REAPER=$!
trap 'kill $REAPER 2>/dev/null' EXIT

echo "[act_decoder] $FAMILY: $LATENT_DIR -> $OUTPUT_DIR (config $REF)"
accelerate launch \
    --num_processes $SLURM_GPUS_ON_NODE \
    --mixed_precision bf16 \
    experiments/pipeline/03_train/train_decoder.py \
    --config "$CONFIG" \
    --latent_dir "$LATENT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $RESUME
STATUS=$?
echo "[act_decoder] done (exit $STATUS): $FAMILY -> $OUTPUT_DIR"
exit $STATUS
