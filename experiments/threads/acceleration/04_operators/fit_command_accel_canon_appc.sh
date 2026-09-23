#!/bin/bash
#SBATCH --job-name=amix_cnapp
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/accelmix_canonappc_%j.out
#SBATCH --error=logs/accelmix_canonappc_%j.err
# COMBINED canon+appc operator (finale): [command || appc(H_a)] -> U_canon coords. Single streaming pass
# (reuses appc_mean/appc_basis + global_basis_canon already on disk), so fast. Writes
# cmd_Wu_canon_appc_L*.npy. Consumed by steer_accel2d.py --features canon_appc. day/256G (LatentDataset
# caching OOM'd smaller runs at 64G; day nodes are 1TB w/ 400-600G free -> 256G still schedules instantly).
module purge; module load miniconda; conda activate vjepa-physics-decoder
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
BASE=.
KU=${KU:-16}
python -u experiments/threads/acceleration/04_operators/fit_command_operators_accel_canon_appc.py \
    --train_dir $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/train/vjepa2_large \
    --test_dir  $BASE/outputs/latents/moving_ball_scene_accel2d_mixed/test/vjepa2_large \
    --layers 6,12,18,23 \
    --artifacts_dir $BASE/outputs/analysis/moving_ball_accel2d_mixed/subspace \
    --ridge 1.0 --ku $KU
echo "[amix_cnapp] exit=$? (KU=$KU)"
