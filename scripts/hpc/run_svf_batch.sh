#!/usr/bin/env bash
#SBATCH --job-name=ivf-svf
#SBATCH --output=logs/svf_%A_%a.out
#SBATCH --error=logs/svf_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#
# ==========================================================================
# SVF v2 batch runner for HPC (SLURM)
#
# Usage:
#   Single area:
#     sbatch scripts/hpc/run_svf_batch.sh rocinha
#     sbatch scripts/hpc/run_svf_batch.sh vidigal_tls streets
#
#   All default areas (sequential loop):
#     sbatch scripts/hpc/run_svf_batch.sh
#
# Arguments:
#   $1  area name   (optional; if omitted, loops over DEFAULT_AREAS)
#   $2  mode        (optional; default: "all")
# ==========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- edit these to match your cluster
# ---------------------------------------------------------------------------

# Conda / virtualenv activation (uncomment and adjust for your cluster)
# MIT ORCD example:
# module load anaconda3/2024a
# source activate ivf
#
# Generic conda:
# source /path/to/conda/etc/profile.d/conda.sh && conda activate ivf

# Default areas to process when no argument is given
DEFAULT_AREAS=("rocinha" "vidigal_tls")

# SVF computation parameters
BACKEND="raycasting"
STREET_SPACING="1.5"
SKY_PATCHES="145"
MODE="${2:-all}"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

# Navigate to project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Create log directory for SLURM output
mkdir -p logs

# ---------------------------------------------------------------------------
# Run function
# ---------------------------------------------------------------------------

run_area() {
    local area="$1"
    local mode="$2"

    echo "============================================================"
    echo "  Area : ${area}"
    echo "  Mode : ${mode}"
    echo "  Start: $(date -Iseconds)"
    echo "============================================================"

    python scripts/run_svf_v2.py \
        --area "${area}" \
        --mode "${mode}" \
        --backend "${BACKEND}" \
        --street-spacing "${STREET_SPACING}" \
        --sky-patches "${SKY_PATCHES}"

    echo "  Done : $(date -Iseconds)"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "IVF -- SVF v2 batch job"
echo "Host    : $(hostname)"
echo "CPUs    : ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory  : ${SLURM_MEM_PER_NODE:-unknown}"
echo "Started : $(date -Iseconds)"
echo ""

if [[ $# -ge 1 ]]; then
    # Single area mode
    run_area "$1" "${MODE}"
else
    # Multi-area mode: loop over defaults
    for area in "${DEFAULT_AREAS[@]}"; do
        run_area "${area}" "${MODE}"
    done
fi

echo "All areas complete: $(date -Iseconds)"
