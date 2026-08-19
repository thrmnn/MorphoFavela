#!/usr/bin/env bash
#SBATCH --job-name=mf-pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal
#
# ==========================================================================
# Full MorphoFavela analysis pipeline for HPC (SLURM)
#
# Runs ALL analyses for one or more areas:
#   1. SVF v2 (streets, grid, facades)
#   2. Urban morphology (zone metrics)
#   3. Spatial analysis (Moran's I, LISA)
#   4. Typology classification (cross-area clustering)
#
# Usage:
#   sbatch scripts/hpc/run_full_pipeline.sh                    # all areas
#   sbatch scripts/hpc/run_full_pipeline.sh rocinha             # single area
#   sbatch scripts/hpc/run_full_pipeline.sh "rocinha vidigal"   # specific areas
# ==========================================================================

# ---------------------------------------------------------------------------
# Setup — must happen BEFORE set -e so module sourcing doesn't kill the script
# ---------------------------------------------------------------------------

# Use SLURM_SUBMIT_DIR (where sbatch was invoked) or fall back to ~/SCL/SCR/MorphoFavela
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/MorphoFavela}"
cd "${PROJECT_ROOT}"
mkdir -p logs

# Source module system and activate environment
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module load miniforge/25.11.0-0
eval "$(conda shell.bash hook)"
conda activate morphofavela

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_AREAS=("vidigal" "rocinha" "riodaspedras" "complexo_do_alemao" "maré")

# SVF parameters
BACKEND="raycasting"
STREET_SPACING="1.5"
SKY_PATCHES="145"
N_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

# Morphology parameters
CELL_SIZE="50"
FLOOR_HEIGHT="3.0"

# Parse areas from argument or use defaults
if [[ $# -ge 1 ]]; then
    read -ra AREAS <<< "$1"
else
    AREAS=("${DEFAULT_AREAS[@]}")
fi

echo "================================================================"
echo "  MorphoFavela Full Analysis Pipeline"
echo "  Host    : $(hostname)"
echo "  CPUs    : ${N_JOBS}"
echo "  Memory  : ${SLURM_MEM_PER_NODE:-unknown}"
echo "  Areas   : ${AREAS[*]}"
echo "  Started : $(date -Iseconds)"
echo "================================================================"
echo ""

FAILED=()

# ---------------------------------------------------------------------------
# Step 1: SVF v2 (per area)
# ---------------------------------------------------------------------------

for area in "${AREAS[@]}"; do
    echo "============================================================"
    echo "[1/4] SVF v2: ${area} (streets + grid + facades)"
    echo "      Start: $(date -Iseconds)"
    echo "============================================================"

    python scripts/run_svf_v2.py \
        --area "${area}" \
        --mode all \
        --backend "${BACKEND}" \
        --street-spacing "${STREET_SPACING}" \
        --sky-patches "${SKY_PATCHES}" \
        --n-jobs "${N_JOBS}" \
        --checkpoint \
    && echo "  SVF ${area}: OK" \
    || { echo "  SVF ${area}: FAILED"; FAILED+=("svf-${area}"); }

    echo ""
done

# ---------------------------------------------------------------------------
# Step 2: Urban morphology (per area)
# ---------------------------------------------------------------------------

for area in "${AREAS[@]}"; do
    echo "============================================================"
    echo "[2/4] Urban morphology: ${area}"
    echo "      Start: $(date -Iseconds)"
    echo "============================================================"

    python scripts/compute_urban_morphology.py \
        --area "${area}" \
        --cell-size "${CELL_SIZE}" \
        --floor-height "${FLOOR_HEIGHT}" \
    && echo "  Morphology ${area}: OK" \
    || { echo "  Morphology ${area}: FAILED"; FAILED+=("morphology-${area}"); }

    echo ""
done

# ---------------------------------------------------------------------------
# Step 3: Typology classification (cross-area)
# ---------------------------------------------------------------------------

echo "============================================================"
echo "[3/4] Typology classification: ${AREAS[*]}"
echo "      Start: $(date -Iseconds)"
echo "============================================================"

python scripts/classify_typology.py \
    --areas "${AREAS[@]}" \
&& echo "  Typology: OK" \
|| { echo "  Typology: FAILED"; FAILED+=("typology"); }

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "================================================================"
echo "  Pipeline complete: $(date -Iseconds)"
echo "  Areas processed: ${AREAS[*]}"
echo ""

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  Status: ALL SUCCEEDED"
else
    echo "  Status: ${#FAILED[@]} FAILURE(S): ${FAILED[*]}"
fi

echo ""
echo "  Outputs:"
for area in "${AREAS[@]}"; do
    echo "    outputs/${area}/svf_v2/"
    echo "    outputs/${area}/urban_morphology/"
done
echo "    outputs/comparative/typology/"
echo "================================================================"

# Exit with failure if anything failed
[[ ${#FAILED[@]} -eq 0 ]]
