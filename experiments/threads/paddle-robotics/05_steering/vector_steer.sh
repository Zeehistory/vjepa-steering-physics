#!/bin/bash
#SBATCH --job-name=vec_steer
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/vec_steer_%j.out
#SBATCH --error=logs/vec_steer_%j.err

# 2-D (vx, vy) outcome steering: does the scalar strike law generalise to a vector target?
#
# CPU-only and no GPU requested on purpose -- every rollout here runs with render=False, so there is
# no EGL context and nothing to put on a GPU. Asking for one would just queue longer.
#
# --partition=day ALONE with --cpus-per-task=4. Do not "helpfully" widen this to a partition list:
# submits, and it took four tries to find.
#
# Usage:
#   sbatch experiments/threads/paddle-robotics/05_steering/vector_steer.sh
#   EMBODIMENT=paddle sbatch experiments/threads/paddle-robotics/05_steering/vector_steer.sh

cd .

EMBODIMENT=${EMBODIMENT:-franka_dynamic}
# SEED selects the held-out test draw. The headline 82.5% came from a SINGLE seed, which says nothing
# about how much of it was that draw. Setting SEED_TAG=1 routes each seed to its own directory so a
# robustness sweep accumulates alongside the original run instead of overwriting it.
SEED=${SEED:-0}
if [ -n "${SEED_TAG:-}" ]; then
    OUT=${OUT:-outputs/paddle_strike/vector_steer_seeds/${EMBODIMENT}/seed${SEED}}
else
    OUT=${OUT:-outputs/paddle_strike/vector_steer/${EMBODIMENT}}
fi

PY=python
set -eo pipefail

export PYTHONPATH=.
# NOT osmesa: that is what killed 21099383/4 at 27 s. The `day` nodes have no OSMesa library, so
# PyOpenGL loaded a null GL and `import mujoco` died before a single rollout. NOT egl either -- egl
# needs a GPU and this job wants none. `disable` short-circuits MuJoCo's entire GL import chain (see
# mujoco/rendering/classic/gl_context.py), which is exactly right when every rollout is render=False,
# paddle_strike.py sets its default via os.environ.setdefault, so this export wins.
export MUJOCO_GL=disable

"$PY" -u experiments/threads/paddle-robotics/05_steering/vector_steer.py \
    --output_dir "$OUT" \
    --embodiment "$EMBODIMENT" \
    --n_vp "${N_VP:-9}" --n_y "${N_Y:-9}" \
    --y_max "${Y_MAX:-0.030}" \
    --n_test "${N_TEST:-40}" \
    --seed "$SEED"

test -s "$OUT/vector_steer.json" || { echo "FAIL: $OUT/vector_steer.json missing"; exit 1; }
echo "wrote $OUT"
