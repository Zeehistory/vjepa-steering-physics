#!/bin/bash
#SBATCH --job-name=hifi_bw
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:10:00
#SBATCH --output=logs/hifi_bw_%j.out
#SBATCH --error=logs/hifi_bw_%j.err

# The L40S sits at 40% util in bursts -> dataloader-bound, so batch size is irrelevant and
# num_workers is the actual lever. It is pinned at 0 because LatentDataset caches shards PER WORKER;
# with max_cached_shards now bounded, w workers cost ~w x (shards x 2.5GB fp16) instead of unbounded.
# Arms trade cache depth against worker count to stay well under the 180G request.
# Scratch output_dir: cannot touch the live run. Must be L40S to compare against the live 12.55.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=.
CFG="configs/train/hifi_v2d_mixed_decoder.yaml"
CK="outputs/runs/hifi_v2d_mixed_decoder/checkpoints/step_4000.pt"
B=outputs/hifi/bench
nvidia-smi --query-gpu=name --format=csv,noheader

# workers | max_cached_shards
for arm in "2|8" "4|4" "6|3"; do
    IFS='|' read -r NW MC <<< "$arm"
    echo "=========== workers=$NW cache=$MC ==========="
    /usr/bin/time -v python experiments/pipeline/03_train/train_decoder.py --config "$CFG" \
        "train.resume=$CK" "train.num_workers=$NW" "train.max_cached_shards=$MC" \
        "optim.max_steps=4090" "train.log_every=10" "train.ckpt_every=100000" \
        "output_dir=$B/nw${NW}_mc${MC}" \
        2>&1 | grep -E "step [0-9]+/[0-9]+ loss|Maximum resident|out of memory|Killed" | tail -8
    echo "--- rc=$? ---"
done
echo "[hifi_bw] done"
