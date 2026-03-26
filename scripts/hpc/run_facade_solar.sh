#!/usr/bin/env bash
#SBATCH --job-name=facade-solar
#SBATCH --output=logs/facade_solar_%j.out
#SBATCH --error=logs/facade_solar_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=mit_normal
#
# ==========================================================================
# Facade solar irradiance + WHO 2-hour sunlight threshold
#
# Computes facade-level solar access for each CFD campaign area on the
# winter solstice (June 21, worst case for Southern Hemisphere).
#
# Usage:
#   sbatch scripts/hpc/run_facade_solar.sh vidigal_tls
#   sbatch scripts/hpc/run_facade_solar.sh vidigal
#   sbatch --array=0-4 scripts/hpc/run_facade_solar.sh   # all 5 areas
# ==========================================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/IVF}"
cd "${PROJECT_ROOT}"
mkdir -p logs

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module load miniforge/25.11.0-0
eval "$(conda shell.bash hook)"
conda activate ivf

set -euo pipefail

# Array mode: map task ID to area
AREA_LIST=("vidigal_tls" "vidigal" "rocinha" "riodaspedras" "complexo_do_alemao")

if [[ $# -ge 1 ]]; then
    AREA="$1"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    AREA="${AREA_LIST[$SLURM_ARRAY_TASK_ID]}"
else
    echo "Usage: sbatch scripts/hpc/run_facade_solar.sh <area>"
    echo "   or: sbatch --array=0-4 scripts/hpc/run_facade_solar.sh"
    exit 1
fi

N_JOBS="${SLURM_CPUS_PER_TASK:-32}"

echo "================================================================"
echo "  Facade Solar: ${AREA}"
echo "  Host    : $(hostname)"
echo "  CPUs    : ${N_JOBS}"
echo "  Started : $(date -Iseconds)"
echo "================================================================"

# Run facade solar pipeline (winter solstice, skip SVF for speed)
python scripts/run_facade_solar.py \
    --area "${AREA}" \
    --date 2026-06-21 \
    --n-jobs "${N_JOBS}" \
    --facade-spacing 1.0 \
    --skip-svf \
    --interval 30 \
&& echo "  Facade solar: OK" \
|| echo "  Facade solar: FAILED"

echo ""
echo "================================================================"
echo "  Done: $(date -Iseconds)"
echo "================================================================"
