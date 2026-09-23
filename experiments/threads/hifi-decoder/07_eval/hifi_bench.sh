#!/bin/bash
#SBATCH --job-name=hifi_bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=logs/hifi_bench_%j.out
#SBATCH --error=logs/hifi_bench_%j.err

# Throughput probe. The live run does batch_size 1 x grad_accum 8 and leaves the GPU at 7.6/46 GB
# and ~56% util, so it is starving, not memory-bound. Each arm below keeps effective batch = 8, so
# they are all the SAME optimization -- only wall-clock and peak memory differ. Writes to a scratch
# output_dir so it cannot touch the real run's checkpoints.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
NAME=v2d_mixed
CFG="configs/train/hifi_${NAME}_decoder.yaml"
CK="outputs/runs/hifi_${NAME}_decoder/checkpoints/step_2000.pt"
BENCH=outputs/hifi/bench

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# bs | accum | workers
for arm in "1|8|0" "4|2|0" "8|1|0" "4|2|2"; do
    IFS='|' read -r BS AC NW <<< "$arm"
    echo "=========== arm bs=$BS accum=$AC workers=$NW ==========="
    /usr/bin/time -v python experiments/pipeline/03_train/train_decoder.py --config "$CFG" \
        "train.resume=$CK" "train.batch_size=$BS" "optim.grad_accum=$AC" \
        "train.num_workers=$NW" "optim.max_steps=2120" "train.log_every=20" \
        "train.ckpt_every=100000" "output_dir=$BENCH/bs${BS}_ac${AC}_nw${NW}" \
        2>&1 | grep -E "step [0-9]+/|Maximum resident|CUDA out of memory|Error|error" | tail -12
    echo "--- exit: $? ---"
done
echo "[hifi_bench] done"
