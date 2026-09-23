#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=outputs/vjepa_sweep/_logs/pilot_%x_%j.out
#SBATCH --error=outputs/vjepa_sweep/_logs/pilot_%x_%j.err
#
# DECODER-FREE model-size pilot: is the physics signal better in a bigger encoder?
#
# The full sweep is ~2 weeks almost entirely because of decoder training (95 of tier-1's 128
# GPU-h). Every number here is measurable WITHOUT a decoder, so this lands in hours:
#
#   probe R^2          how linearly the quantity is encoded            (experiments/threads/acceleration/06_probes/probe_accel.py)
#   latent gate cos    cos(operator's predicted dH, TRUE H_b - H_a)    (fit_command_operators_*)
#                      on held-out scenes, against a shuffled-command control
#
# The gate is the honest decoder-free proxy for steerability: it asks whether the command->edit
# operator points the right way in latent space. It does NOT establish that decoded PIXELS move
# (that needs the decoder, i.e. the full sweep) -- but if the gate does not scale with model size,
# the pixel result almost certainly will not either, and that is worth knowing in 2 hours.
#
# Usage:  ENC=vjepa2_large QUANT=accel2d_mixed sbatch experiments/threads/model-size-sweep/11_run/j_pilot.sh
set -euo pipefail

ENC=${ENC:?set ENC=vjepa2_large|vjepa2_huge|vjepa2_giant}
QUANT=${QUANT:-accel2d_mixed}
N_TRAIN=${N_TRAIN:-800}
N_TEST=${N_TEST:-400}

# Matched RELATIVE depth -- comparing at matched layer INDEX would be a depth comparison, not a
# size comparison (layer 23 is 96% of ViT-L but only 58% of ViT-g).
case "$ENC" in
  vjepa2_large) LAYERS=12,23 ; BATCH=8 ;;
  vjepa2_huge)  LAYERS=16,31 ; BATCH=4 ;;
  vjepa2_giant) LAYERS=20,38 ; BATCH=4 ;;
  *) echo "unknown ENC=$ENC" >&2; exit 1 ;;
esac

# QUANT -> data config, quantity fn, probe label keys, and which appearance factors that dataset
# INTENDS to hold fixed (the `_mixed` variants randomize appearance; the base ones pin background
# on purpose to isolate the physics).
case "$QUANT" in
  accel2d_mixed)    DATA=moving_ball_scene_accel2d_mixed    ; QARG=accel    ; FIXED=""           ; LK=obj0_acc_x,obj0_acc_y ;;
  accel2d)          DATA=moving_ball_scene_accel2d          ; QARG=accel    ; FIXED="background" ; LK=obj0_acc_x,obj0_acc_y ;;
  velocity2d_mixed) DATA=moving_ball_scene_velocity2d_mixed ; QARG=velocity ; FIXED=""           ; LK=obj0_vel_x,obj0_vel_y ;;
  velocity2d)       DATA=moving_ball_scene_velocity2d       ; QARG=velocity ; FIXED="background" ; LK=obj0_vel_x,obj0_vel_y ;;
  gravity)          DATA=moving_ball_scene_gravity          ; QARG=accel    ; FIXED="background" ; LK=obj0_acc_y ;;
  angvel2d)         DATA=moving_ball_scene_angvel2d         ; QARG=angvel   ; FIXED="background" ; LK=obj0_omega ;;
  angaccel2d)       DATA=moving_ball_scene_angaccel2d       ; QARG=angaccel ; FIXED="background" ; LK=obj0_alpha ;;
  *) echo "unknown QUANT=$QUANT" >&2; exit 1 ;;
esac

REPO=.
ROOT=outputs/vjepa_sweep/pilot/$QUANT/$ENC
LAT=$ROOT/latents
ART=$ROOT/artifacts
OUT=$ROOT/out
mkdir -p "$LAT" "$ART" "$OUT" outputs/vjepa_sweep/_logs

module purge
module load miniconda
eval "$(conda shell.bash hook)"
conda activate vjepa-physics-decoder
export PYTHONPATH=$REPO
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # never /home: 125 GiB quota
cd "$REPO"

echo "=== pilot ENC=$ENC QUANT=$QUANT LAYERS=$LAYERS n=$N_TRAIN/$N_TEST on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- 0. preflight: never spend GPU time on a degenerate dataset -----------------------------
if [ ! -f "$ART/preflight.json" ]; then
  python -u experiments/threads/model-size-sweep/01_data/preflight_dataset.py --data_cfg "$DATA" --label_keys "$LK" \
    --fixed_appearance "$FIXED" --n_train 128 --n_test 128 \
    --train_clips "$N_TRAIN" --test_clips "$N_TEST" --out "$ART/preflight.json"
fi

# ---- 1. extract (the only GPU step) ---------------------------------------------------------
# Resumable: a finished split leaves metadata.parquet, so a requeue skips straight past it.
# Train split stores no frames -- nothing downstream of here reads pixels.
if [ ! -f "$LAT/train/metadata.parquet" ]; then
  python -u experiments/pipeline/02_encode/extract_latents.py --encoder "$ENC" --dataset "$DATA" --layers "$LAYERS" \
    --output_dir "$LAT/train" --batch_size "$BATCH" \
    --latent_dtype float16 --no_store_frames \
    data.split=train "data.num_clips=$N_TRAIN" data.seed=0
fi
if [ ! -f "$LAT/test/metadata.parquet" ]; then
  python -u experiments/pipeline/02_encode/extract_latents.py --encoder "$ENC" --dataset "$DATA" --layers "$LAYERS" \
    --output_dir "$LAT/test" --batch_size "$BATCH" \
    --latent_dtype float16 --frames_dtype float16 \
    data.split=test "data.num_clips=$N_TEST" data.seed=2
fi
du -sh "$LAT"/train "$LAT"/test

# ---- 2. subspace -> global_basis_L*.npy (CPU) -----------------------------------------------
python -u experiments/threads/acceleration/04_operators/accel_subspace.py --train_dir "$LAT/train" --test_dir "$LAT/test" \
  --layers "$LAYERS" --quantity "$QARG" --output_dir "$ART" --save_k 8

# ---- 3. command operator + HELD-OUT LATENT GATE (CPU, no decoder) ---------------------------
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py --train_dir "$LAT/train" --test_dir "$LAT/test" \
  --layers "$LAYERS" --quantity "$QARG" --artifacts_dir "$ART" --ku 8

# ---- 4. probe R^2 (CPU) ----------------------------------------------------------------------
# --skip_probe_steer: the probe->steer section needs canon artifacts from a different pipeline
# (fit_command_operators_accel_canon.py). The pilot only wants probe R^2.
python -u experiments/threads/acceleration/06_probes/probe_accel.py --train_dir "$LAT/train" --test_dir "$LAT/test" \
  --layers "$LAYERS" --quantity "$QARG" --artifacts_dir "$ART" --output_dir "$OUT" --skip_probe_steer

echo "=== pilot DONE ENC=$ENC QUANT=$QUANT -> $ROOT ==="
