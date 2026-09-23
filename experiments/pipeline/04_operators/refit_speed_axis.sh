#!/bin/bash
#SBATCH --job-name=spd_axis
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=logs/spd_axis_%j.out
#SBATCH --error=logs/spd_axis_%j.err
# 2026-09-15 velocity MAGNITUDE fix: the PCA subspace U8/U16 carries heading but drops the speed change
# (even the oracle dH projected onto U8 keeps only 0.32 of |v_b| and 0.55 of |v_a|). Fit supervised speed
# axes (speed_axis.py), refit the standardized command operator on [U; speed axes], steer 100 scenes with a
# plain gain sweep AND a split gain on the speed-axis part, calibrate. One SPEC per job:
#   SPEC=v2d | vmix | rb3d
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
export MUJOCO_GL=egl
SC=.
BASE=.
SPEC=${SPEC:?v2d|vmix|rb3d}
case "$SPEC" in
  v2d)  DSET=moving_ball_scene_v2d;       ANDIR=moving_ball_v2d;       CFG=moving_ball_scene_decoder; RUN=moving_ball_scene_v2d_decoder_fp;       KU=8;  KS=2,4,8,11
        TE=$BASE/outputs/latents/$DSET/test/vjepa2_large ;;
  vmix) DSET=moving_ball_scene_v2d_mixed; ANDIR=moving_ball_v2d_mixed; CFG=moving_ball_scene_decoder; RUN=moving_ball_scene_v2d_mixed_decoder_fp; KU=16; KS=2,4,8,16,19
        TE=$SC/outputs/latents/$DSET/test/vjepa2_large ;;
  rb3d) DSET=rolling_ball3d;              ANDIR=rolling_ball3d;        CFG=rolling_ball3d_decoder;    RUN=rolling_ball3d_decoder_fp_seed1;        KU=16; KS=2,4,8,16,19
        TE=$SC/outputs/latents/$DSET/test/vjepa2_large ;;
  *) echo "bad SPEC=$SPEC"; exit 1 ;;
esac
TR=$SC/outputs/latents/$DSET/train/vjepa2_large
ART=$BASE/outputs/analysis/$ANDIR/subspace
OUT=$BASE/outputs/analysis/$ANDIR/steer_refit_std_spd
echo "[spd] SPEC=$SPEC KU=$KU train=$TR test=$TE"

[ -f "$ART/global_basis_spd_L23.npy" ] && echo "[spd] speed axes exist, skip" || \
python -u experiments/threads/velocity/04_operators/speed_axis.py \
    --train_dir "$TR" --layers 6,12,18,23 --artifacts_dir "$ART" --ku "$KU" || exit 1
python -u experiments/pipeline/04_operators/fit_command_operators.py \
    --train_dir "$TR" --test_dir "$TE" --layers 6,12,18,23 --artifacts_dir "$ART" \
    --ridge 1.0 --ku 0 --basis_tag spd --skip_rich --standardize || exit 1
python -u experiments/threads/velocity/05_steering/steer_velocity2d.py \
    --config configs/train/$CFG.yaml --test_dir "$TE" --artifacts_dir "$ART" \
    --checkpoint "$BASE/outputs/runs/$RUN/checkpoints/last.pt" --output_dir "$OUT" \
    --ks "$KS" --num_scenes 100 --cmd_scales "1.0,1.5,2.0,2.5,3.0" --spd_scales "1.5,2.0,3.0" \
    --cmd_ku 0 --basis_tag spd --cmd_std --dir_bins "" --device cuda || exit 1
python -u experiments/pipeline/04_operators/calibrate_cmd_gain.py \
    --summary "$OUT/steer2d_summary.json" --val_frac 0.5 --out "$OUT/calib_cmd_gain.json"
echo "[spd] DONE $SPEC -> $OUT"
