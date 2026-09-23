#!/bin/bash
#SBATCH --job-name=v2d_cmdop
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2d_cmdop_%j.out
#SBATCH --error=logs/v2d_cmdop_%j.err
# Fit COMMAND-ONLY subspace-synthesis operators (W_U: command -> U8 coords; B_rich: rich command -> dH).
# Latent-only, CPU, big RAM for the train shard cache. Reuses global_basis_L*.npy from velocity_subspace.
# ALWAYS submit via the queue-aware picker:
#   PART=$(experiments/pipeline/00_common/pick_partition.sh 256000 bigmem week day); sbatch -p <partition> --mem=256G experiments/pipeline/04_operators/fit_command.sh
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
python -u experiments/pipeline/04_operators/fit_command_operators.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_v2d/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_v2d/test/vjepa2_large \
    --layers ${LAYERS:-6,12,18,23} --ridge 1.0 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_v2d/subspace
echo "[v2d_cmdop] exit=$?"
