#!/usr/bin/env bash
# End-to-end pipeline for both 3DVA components.
#
#   1. stitch        block-aligned stitching, ACTN4 docking, map occupancy
#   2. central_axis  Markov-chain central-axis refinement
#   3. deformation   curvature deformation scores vs the tracing baseline
#   4. figure        ACTN4 occupancy vs curvature deformation score
#
# Input locations are configured in src/config.py (override with the
# ACTN4_DATA_ROOT / ACTN4_CURVATURE_CSV / ACTN4_MODEL environment variables).
set -euo pipefail
cd "$(dirname "$0")"

COMPONENTS=("component_1" "component_2")

for component in "${COMPONENTS[@]}"; do
    mkdir -p "results/${component}/logs"
    echo "=== ${component}: stitching + ACTN4 docking + occupancy ==="
    python3 -m src.stitch "${component}" | tee "results/${component}/logs/1_stitch.log"
    echo "=== ${component}: central-axis refinement (Markov chain) ==="
    python3 -m src.central_axis "${component}" | tee "results/${component}/logs/2_central_axis.log"
    echo "=== ${component}: curvature deformation scores ==="
    python3 -m src.deformation "${component}" | tee "results/${component}/logs/3_deformation.log"
done

echo "=== figure ==="
python3 -m src.plot_deformation | tee "results/4_plot_deformation.log"
echo "done"
