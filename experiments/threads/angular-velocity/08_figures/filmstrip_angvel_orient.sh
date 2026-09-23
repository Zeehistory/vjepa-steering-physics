#!/bin/bash
#SBATCH --job-name=fs_angvel_or
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fs_angvel_or_%j.out
#SBATCH --error=logs/fs_angvel_or_%j.err
# The `_fp` angvel decoder renders a static smear -- the unsteered control row proves the failure is
# the DECODER, not the operator. `_decoder_orient` is the one trained with the orientation loss and
# is what the rotation analyses actually render with, so re-dump against it.
set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
module purge; module load miniconda; conda activate vjepa-physics-decoder
export PYTHONPATH=. MUJOCO_GL=egl
FS=outputs/filmstrip
python experiments/pipeline/08_figures/dump_filmstrip.py \
  --config configs/train/moving_ball_scene_angvel_decoder_orient.yaml \
  --test_dir "$FS/latents/moving_ball_scene_angvel2d/test/vjepa2_large" \
  --artifacts_dir outputs/analysis/moving_ball_angvel2d/subspace \
  --checkpoint outputs/runs/moving_ball_scene_angvel2d_decoder_orient/checkpoints/last.pt \
  --output_dir "$FS/frames/angvel2d_orient" \
  --quantity angvel --gain 2.0 --num_scenes 6 --device cuda
