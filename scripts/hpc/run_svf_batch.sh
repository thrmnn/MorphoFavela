#!/usr/bin/env bash
#SBATCH --job-name=ivf-svf
#SBATCH --output=logs/svf_%j.out
#SBATCH --error=logs/svf_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal
#
# ==========================================================================
# SVF v2 batch runner for HPC (SLURM)
#
# Usage:
#   sbatch scripts/hpc/run_svf_batch.sh                       # all areas, streets
#   sbatch scripts/hpc/run_svf_batch.sh rocinha                # single area
#   sbatch scripts/hpc/run_svf_batch.sh rocinha all            # single area, all modes
#   sbatch scripts/hpc/run_svf_batch.sh vidigal_tls facades    # single area+mode
# ==========================================================================

# ---------------------------------------------------------------------------
# Setup — must happen BEFORE set -e so module sourcing doesn't kill the script
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module load miniforge/25.11.0-0
eval "$(conda shell.bash hook)"
conda activate ivf

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_AREAS=("rocinha" "vidigal_tls" "vidigal")

BACKEND="raycasting"
STREET_SPACING="1.5"
SKY_PATCHES="145"
N_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
MODE="${2:-streets}"
cd "${PROJECT_ROOT}"
mkdir -p logs

# ---------------------------------------------------------------------------
# SVF run function
# ---------------------------------------------------------------------------

run_svf() {
    local area="$1"
    local mode="$2"

    echo "============================================================"
    echo "  SVF: ${area} / ${mode}"
    echo "  Jobs: ${N_JOBS} cores"
    echo "  Start: $(date -Iseconds)"
    echo "============================================================"

    python scripts/run_svf_v2.py \
        --area "${area}" \
        --mode "${mode}" \
        --backend "${BACKEND}" \
        --street-spacing "${STREET_SPACING}" \
        --sky-patches "${SKY_PATCHES}" \
        --n-jobs "${N_JOBS}" \
        --checkpoint

    echo "  Done : $(date -Iseconds)"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "MorphoFavela -- SVF v2 batch job"
echo "Host    : $(hostname)"
echo "CPUs    : ${N_JOBS}"
echo "Memory  : ${SLURM_MEM_PER_NODE:-unknown}"
echo "Started : $(date -Iseconds)"
echo ""

if [[ $# -ge 1 ]]; then
    run_svf "$1" "${MODE}"
else
    for area in "${DEFAULT_AREAS[@]}"; do
        run_svf "${area}" "${MODE}"
    done
fi

echo "All SVF complete: $(date -Iseconds)"
