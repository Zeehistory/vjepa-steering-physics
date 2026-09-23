#!/bin/bash
#SBATCH --job-name=dec_sweep
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=logs/dec_sweep_%j.out
#SBATCH --error=logs/dec_sweep_%j.err
# 2026-09-15 DECODED model-size sweep: the full ViT-L steering ladder repeated at ViT-H / ViT-g, through a
# decoder trained per encoder (the arm the 08-20 L/H/g sweep cut). One cell per job, resume-safe at every
# stage (each stage is skipped when its output exists), everything on scratch except the final last.pt.
#
#   ENC=vjepa2_huge|vjepa2_giant   QUANT=v2d|accel2d|angvel
#
#   v2d     : extract train/test (single appearance, same seeds/ranges as ViT-L) -> train frame_position
#             decoder -> velocity subspace -> command operator (baseline + STANDARDIZED) -> 100-scene
#             decoded steer x2 -> gain calibration.
#   accel2d : same with the acceleration scripts.
#   angvel  : extract angvel_big train (1000, seed 7) / test (240, seed 11) + angaccel test (240, seed 11)
#             -> train orientation decoder -> Fourier-in-orientation steer for angvel AND zero-shot angaccel.
#
# Layers are matched by RELATIVE depth (L 6/12/18/23 -> H 8/16/24/31 -> g 10/20/30/38).
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
export MUJOCO_GL=egl
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
requeue() { echo "[dec_sweep] USR1/timeout -> requeue $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0; }
trap requeue USR1

ENC=${ENC:?vjepa2_huge|vjepa2_giant}
QUANT=${QUANT:?v2d|accel2d|angvel}
case "$ENC" in
  vjepa2_huge)  LAYERS=8,16,24,31;  CTAG=vith; BATCH=6 ;;
  vjepa2_giant) LAYERS=10,20,30,38; CTAG=vitg; BATCH=4 ;;
  vjepa2_large) LAYERS=6,12,18,23;  CTAG=vitl; BATCH=8 ;;
  *) echo "bad ENC"; exit 1 ;;
esac
SC=outputs/vjepa_sweep/decoded/$QUANT/$ENC
BASE=.
LAT=$SC/latents; RUN=$SC/decoder; ART=$SC/artifacts; OUT=$BASE/outputs/analysis/size_sweep_decoded/$QUANT/$ENC
mkdir -p "$LAT" "$RUN" "$ART" "$OUT"
MAX_STEPS=${MAX_STEPS:-8000}
echo "[dec_sweep] ENC=$ENC QUANT=$QUANT LAYERS=$LAYERS host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

extract() {  # split outdir num_clips seed config scenario extra...
  local split=$1 outd=$2 n=$3 seed=$4 cfg=$5 scen=$6; shift 6
  if [ -f "$outd/metadata.parquet" ]; then echo "[dec_sweep] extract $split: exists, skip"; return 0; fi
  python -u experiments/pipeline/02_encode/extract_latents.py --config "$cfg" --encoder "$ENC" --layers "$LAYERS" \
    --output_dir "$outd" --batch_size $BATCH --shard_size 64 --latent_dtype float16 --frames_dtype float16 \
    data.scenario=$scen data.clips_per_scene=8 data.num_clips=$n data.seed=$seed "$@"
}
train_dec() {  # cfg latent_dir
  local cfg=$1 lat=$2
  if [ -f "$RUN/checkpoints/last.pt" ] && [ -f "$RUN/DONE" ]; then echo "[dec_sweep] decoder: done, skip"; return 0; fi
  local RESUME=""; local CK=$RUN/checkpoints
  if [ -d "$CK" ]; then L=$(ls -1t "$CK"/last.pt "$CK"/step_*.pt 2>/dev/null | head -1); [ -n "$L" ] && RESUME="train.resume=$L" && echo "[dec_sweep] resume $L"; fi
  accelerate launch --num_processes 1 --mixed_precision bf16 experiments/pipeline/03_train/train_decoder.py \
    --config "$cfg" --latent_dir "$lat" --output_dir "$RUN" optim.max_steps=$MAX_STEPS train.log_every=50 ${TRAIN_EXTRA:-} $RESUME &
  wait $! ; local rc=$?
  [ $rc -eq 0 ] && touch "$RUN/DONE" && { ls "$CK"/step_*.pt 2>/dev/null | head -n -1 | xargs -r rm -f; }
  return $rc
}

case "$QUANT" in
v2d|accel2d|rb3d_accel)
  CFG=configs/train/moving_ball_scene_decoder_$CTAG.yaml
  if [ "$QUANT" = v2d ]; then SCEN=scene_velocity2d; RANGES=("data.speed_range=[0.012,0.026]" "data.radius_range=[0.11,0.11]"); Q=velocity
  elif [ "$QUANT" = rb3d_accel ]; then CFG=configs/train/rolling_ball3d_accel_decoder.yaml; SCEN=scene_rollingball3d_accel
       RANGES=("data.speed_range=[0.04,0.08]" "data.accel_range=[0.03,0.07]"); Q=accel   # 3-D MuJoCo acceleration (ViT-L only)
  else SCEN=scene_accel2d; RANGES=("data.speed_range=[0.008,0.016]" "data.accel_range=[0.0015,0.0035]" "data.radius_range=[0.08,0.12]"); Q=accel; fi
  extract train $LAT/train 4000 0 $CFG $SCEN "${RANGES[@]}" || exit 1
  extract test  $LAT/test   800 2 $CFG $SCEN "${RANGES[@]}" || exit 1
  train_dec $CFG $LAT/train || exit 1
  if [ ! -f "$ART/global_basis_L${LAYERS%%,*}.npy" ]; then
    if [ "$Q" = velocity ]; then
      python -u experiments/threads/velocity/04_operators/velocity_subspace.py --train_dir $LAT/train --test_dir $LAT/test \
        --layers $LAYERS --output_dir $ART --ridge 1.0 --save_k 16 --max_global_pairs 800 || exit 1
    else
      python -u experiments/threads/acceleration/04_operators/accel_subspace.py --train_dir $LAT/train --test_dir $LAT/test \
        --layers $LAYERS --quantity accel --output_dir $ART --ridge 1.0 --save_k 16 --max_global_pairs 800 || exit 1
    fi
  fi
  KU=$([ "$Q" = velocity ] && echo 16 || echo 8)
  for ARM in baseline standardized; do
    TAG=""; FITX=(); STDX=(); [ "$ARM" = standardized ] && { TAG="_std"; FITX=(--standardize); STDX=(--cmd_std); }
    SO=$OUT/steer_refit$TAG
    [ -f "$SO/steer2d_summary.json" ] && { echo "[dec_sweep] $ARM steer exists, skip"; continue; }
    if [ "$Q" = velocity ]; then
      python -u experiments/pipeline/04_operators/fit_command_operators.py --train_dir $LAT/train --test_dir $LAT/test \
        --layers $LAYERS --artifacts_dir $ART --ridge 1.0 --ku $KU --skip_rich "${FITX[@]}" || exit 1
      python -u experiments/threads/velocity/05_steering/steer_velocity2d.py --config $CFG --test_dir $LAT/test \
        --artifacts_dir $ART --checkpoint $RUN/checkpoints/last.pt --output_dir $SO --ks 2,4,8,16 --num_scenes 100 \
        --cmd_scales "1.0,1.5,2.0,2.5,3.0" --cmd_ku $KU --dir_bins "" "${STDX[@]}" --device cuda || exit 1
    else
      python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel.py --train_dir $LAT/train --test_dir $LAT/test \
        --layers $LAYERS --artifacts_dir $ART --ridge 1.0 --ku $KU --quantity accel "${FITX[@]}" || exit 1
      python -u experiments/threads/acceleration/05_steering/steer_accel2d.py --config $CFG --test_dir $LAT/test \
        --artifacts_dir $ART --checkpoint $RUN/checkpoints/last.pt --output_dir $SO --ks 2,4,8,16 --num_scenes 100 \
        --cmd_scales "0.5,1.0,1.5,2.0,2.5,3.0,4.0" --cmd_ku $KU "${STDX[@]}" --device cuda || exit 1
    fi
    python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py --summary $SO/steer2d_summary.json --val_frac 0.5 --out $SO/calib_cmd_gain.json
  done
  python -u experiments/pipeline/07_eval/compare_magnitude_control.py --base $OUT/steer_refit/steer2d_summary.json \
    --std $OUT/steer_refit_std/steer2d_summary.json --label "$QUANT $ENC"
  ;;
angvel)
  CFG=configs/train/moving_ball_scene_angvel_big_decoder_orient_$CTAG.yaml
  R=("data.radius_range=[0.32,0.42]")
  extract train $LAT/train 1000 7  $CFG scene_angvel2d   "data.omega_range=[0.06,0.20]" "${R[@]}" || exit 1
  extract test  $LAT/test   240 11 $CFG scene_angvel2d   "data.omega_range=[0.06,0.20]" "${R[@]}" || exit 1
  extract angaccel_test $LAT/angaccel_test 240 11 $CFG scene_angaccel2d "data.omega0_range=[-0.06,0.06]" "data.alpha_range=[0.005,0.014]" "${R[@]}" || exit 1
  extract angaccel_mixed_test $LAT/angaccel_mixed_test 240 11 $CFG scene_angaccel2d_mixed "data.omega0_range=[-0.06,0.06]" "data.alpha_range=[0.005,0.014]" "${R[@]}" || exit 1
  extract angvel_mixed_test $LAT/angvel_mixed_test 240 23 $CFG scene_angvel2d_mixed "data.omega_range=[0.06,0.20]" "${R[@]}" || exit 1
  train_dec $CFG $LAT/train || exit 1
  for spec in "angvel|test|angvel_heldout" "angvel|angvel_mixed_test|angvel_mixed" "angaccel|angaccel_test|angaccel_zeroshot" "angaccel|angaccel_mixed_test|angaccel_zeroshot_mixed"; do
    IFS='|' read -r QQ TD TG <<< "$spec"
    [ -f "$OUT/fourier_$TG.json" ] && { echo "[dec_sweep] $TG exists, skip"; continue; }
    python -u experiments/pipeline/05_steering/steer_fourier.py --config $CFG --train_dir $LAT/train --test_dir $LAT/$TD \
      --quantity $QQ --checkpoint $RUN/checkpoints/last.pt --n_train_scenes 125 --n_test_scenes 30 \
      --order 4 --harmonics all --basis orientation --ridge 10 --gains 0.5,1,1.5,2,3,4 --out $OUT/fourier_$TG.json || exit 1
  done
  ;;
esac
# keep the decoder on project (scratch purges in ~30 d); 3.3G each
mkdir -p $BASE/outputs/runs/size_sweep_decoded/$QUANT/$ENC && cp -n $RUN/checkpoints/last.pt $BASE/outputs/runs/size_sweep_decoded/$QUANT/$ENC/ 2>/dev/null
echo "[dec_sweep] DONE ENC=$ENC QUANT=$QUANT -> $OUT"
