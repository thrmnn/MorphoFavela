#!/usr/bin/env bash
#SBATCH --job-name=svf-streets
#SBATCH --output=logs/streets_%j.out
#SBATCH --error=logs/streets_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=mit_normal
#
# ==========================================================================
# Fast SVF streets-only pipeline
#
# Usage:
#   sbatch scripts/hpc/run_streets_fast.sh rocinha
#   sbatch scripts/hpc/run_streets_fast.sh maré
#   sbatch --array=0-1 scripts/hpc/run_streets_fast.sh   # both in parallel
# ==========================================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/MorphoFavela}"
cd "${PROJECT_ROOT}"
mkdir -p logs

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module load miniforge/25.11.0-0
eval "$(conda shell.bash hook)"
conda activate morphofavela

set -euo pipefail

# Array mode: map task ID to area
AREA_LIST=("complexo_do_alemao" "vidigal")

if [[ $# -ge 1 ]]; then
    AREA="$1"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    AREA="${AREA_LIST[$SLURM_ARRAY_TASK_ID]}"
else
    echo "Usage: sbatch scripts/hpc/run_streets_fast.sh <area>"
    echo "   or: sbatch --array=0-1 scripts/hpc/run_streets_fast.sh"
    exit 1
fi

N_JOBS="${SLURM_CPUS_PER_TASK:-32}"
FAILED=()

echo "================================================================"
echo "  SVF Streets + Morphology: ${AREA}"
echo "  Host    : $(hostname)"
echo "  CPUs    : ${N_JOBS}"
echo "  Started : $(date -Iseconds)"
echo "================================================================"

# 1. SVF streets only (32 parallel workers)
echo "[1/3] SVF streets: ${AREA}"
python scripts/run_svf_v2.py \
    --area "${AREA}" \
    --mode streets \
    --backend raycasting \
    --street-spacing 1.5 \
    --sky-patches 145 \
    --n-jobs "${N_JOBS}" \
    --checkpoint \
&& echo "  SVF streets: OK" \
|| { echo "  SVF streets: FAILED"; FAILED+=("svf-streets"); }

# 2. Urban morphology
echo "[2/3] Urban morphology: ${AREA}"
python scripts/compute_urban_morphology.py \
    --area "${AREA}" --cell-size 50 --floor-height 3.0 \
&& echo "  Morphology: OK" \
|| { echo "  Morphology: FAILED"; FAILED+=("morphology"); }

# 3. Morphology metrics
echo "[3/3] Morphology metrics: ${AREA}"
python scripts/calculate_morphology_metrics.py --area "${AREA}" \
&& echo "  Metrics: OK" \
|| { echo "  Metrics: FAILED"; FAILED+=("metrics"); }

echo ""
echo "================================================================"
echo "  Done: $(date -Iseconds)"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  Status: ALL SUCCEEDED"
else
    echo "  Status: ${#FAILED[@]} FAILURE(S): ${FAILED[*]}"
fi
echo "================================================================"

[[ ${#FAILED[@]} -eq 0 ]]
