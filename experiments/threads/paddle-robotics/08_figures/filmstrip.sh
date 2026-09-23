#!/bin/bash
#SBATCH --job-name=robo_filmstrip
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/robo_filmstrip_%j.out
#SBATCH --error=logs/robo_filmstrip_%j.err
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=. MUJOCO_GL=egl
OUT=outputs/filmstrip/frames/robotics
FAILED=""
for EMB in paddle franka franka_dynamic; do
  echo "=========== $EMB ==========="
  python experiments/threads/paddle-robotics/08_figures/dump_robotics_filmstrip.py \
      --output_dir "$OUT" --embodiment "$EMB" --ratio 0.5 --n_instants 6 \
      --image_size 512 ${WIDE:+--wide} || FAILED="$FAILED $EMB"
done
[ -n "$FAILED" ] && { echo "[robo] FAILED:$FAILED"; exit 1; }
echo "[robo] done -> $OUT"
