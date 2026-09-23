#!/bin/bash
#SBATCH --job-name=vec_frames
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/vec_frames_%j.out
#SBATCH --error=logs/vec_frames_%j.err

# Still sheet for VECTOR-valued steering: three commanded outgoing directions, one controller.
#
# A GPU job, unlike vector_steer.py which is CPU-only: this one RENDERS (render=True), so it needs a
# real EGL context. MUJOCO_GL=disable would break it, and osmesa is not installed in this env.
#
#   sbatch experiments/threads/paddle-robotics/08_figures/demo_vector_frames.sh
#   EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/08_figures/demo_vector_frames.sh   # the negative control
#
# NOTE the `--lat_targets=...` equals form below. The default list starts with a MINUS SIGN, and in
# the space-separated form argparse reads "-0.45,0.0,0.45" as an option name rather than a value and
# exits with "expected one argument". The equals form is unambiguous.

cd .
module purge; module load miniconda; conda activate vjepa-physics-decoder
mkdir -p logs

EMBODIMENT=${EMBODIMENT:-franka_dynamic}
OUT=${OUT:-outputs/paddle_strike/vector_frames}
export PYTHONPATH=.
export MUJOCO_GL=egl

python -u experiments/threads/paddle-robotics/08_figures/demo_vector_frames.py \
    --output_dir "$OUT" \
    --embodiment "$EMBODIMENT" \
    --image_size "${IMAGE_SIZE:-768}" \
    --lat_targets="${LAT_TARGETS:--0.45,0.0,0.45}" \
    ${WIDE:+--wide}
RC=$?
test -s "$OUT/vector_frames_${EMBODIMENT}.png" || { echo "FAIL: sheet missing"; exit 1; }
echo "[vec_frames] done (rc=$RC) -> $OUT"
exit $RC
