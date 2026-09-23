#!/bin/bash
#SBATCH --job-name=pstrike_valsys
# devel IS included here, and that is the opposite of slurm_close_action_loop.sh on purpose. Slurm
# the whole list for any job that exceeds it -- which is why the action-loop job (8 cpu, 96 G) must
# leave devel out. This job is 4 cpu / 16 G, i.e. exactly at the cpu cap and well under the memory
# one, so it is admissible, and devel is normally the shortest queue on the cluster.
#SBATCH --requeue
# 2, not 4, and the reason is devel's cap being per-USER not per-job: an interactive devel allocation
# here -- the work is one MuJoCo rollout at a time and the linear algebra is on tables of a few
# hundred rows, so this job is serial in practice.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pstrike_valsys_%j.out
#SBATCH --error=logs/pstrike_valsys_%j.err

# E9-E11: the three gaps between a certified simulator and a validated system.
#
#   sbatch experiments/threads/paddle-robotics/11_run/validated_system.sh
#   ONLY=e10 sbatch experiments/threads/paddle-robotics/11_run/validated_system.sh    # re-run one arm, keep the rest
#
# NO GPU and modest memory: every rollout is simulate(render=False), so this is pure CPU MuJoCo and
# holds nothing bigger than a sweep table. That is the opposite of the action-loop job, whose 96 GB is
# for pooled latents -- nothing here touches a latent.

BASE=${BASE:-"outputs/paddle_strike"}
OUT="${OUT:-${BASE}/validated_system}"

module purge
module load miniconda
conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "[valsys] out=$OUT only=${ONLY:-all}"
PYTHONPATH=. python experiments/threads/paddle-robotics/11_run/validated_system.py \
    --output_dir "$OUT" ${ONLY:+--only "$ONLY"}
STATUS=$?
echo "[valsys] done (exit $STATUS) -> $OUT"
exit $STATUS
