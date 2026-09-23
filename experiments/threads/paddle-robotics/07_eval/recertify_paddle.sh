#!/bin/bash
#SBATCH --job-name=pstrike_recert
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pstrike_recert_%j.out
#SBATCH --error=logs/pstrike_recert_%j.err

# Re-run the 240-held-out-command controllability certificate for BOTH embodiments.
#
# Purpose here is regression, not discovery: the contact-visibility work changed only rendering
# (image_size, crops, figures) plus two extra diagnostic arrays in simulate()'s return dict, so the
# certificate MUST come back identical. If it does not, a "render-only" change was not render-only.
#
# No GPU and no partition with a GPU: the certificate runs with render=False, so it needs neither
# EGL nor a display. That is also why it can sit on `day` rather than queueing behind GPU work.

cd .
PY=python
set -eo pipefail
export PYTHONPATH=.

OUT=${OUT:-outputs/paddle_strike/recert}

for EMB in paddle franka; do
    echo "=================== $EMB ==================="
    "$PY" -u experiments/pipeline/01_data/validate_paddle_strike.py \
        --output_dir "$OUT/$EMB" --embodiment "$EMB" --n_commands 240
done
echo "RECERT DONE -> $OUT"
