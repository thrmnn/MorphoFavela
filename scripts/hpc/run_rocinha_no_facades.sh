#!/usr/bin/env bash
#SBATCH --job-name=rocinha-svf
#SBATCH --output=logs/rocinha_%j.out
#SBATCH --error=logs/rocinha_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mit_normal

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$HOME/MorphoFavela}"
cd "${PROJECT_ROOT}"
mkdir -p logs

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module load miniforge/25.11.0-0
eval "$(conda shell.bash hook)"
conda activate morphofavela

set -euo pipefail

AREA="rocinha"
N_JOBS="${SLURM_CPUS_PER_TASK:-8}"
FAILED=()

echo "================================================================"
echo "  Rocinha Analysis (grid + streets, no facades)"
echo "  Host    : $(hostname)"
echo "  CPUs    : ${N_JOBS}"
echo "  Started : $(date -Iseconds)"
echo "================================================================"

# 1. SVF grid
echo "[1/4] SVF grid: ${AREA}"
python scripts/run_svf_v2.py \
    --area "${AREA}" --mode grid \
    --backend raycasting --sky-patches 145 \
    --n-jobs "${N_JOBS}" --checkpoint \
&& echo "  SVF grid: OK" \
|| { echo "  SVF grid: FAILED"; FAILED+=("svf-grid"); }

# 2. SVF streets
echo "[2/4] SVF streets: ${AREA}"
python scripts/run_svf_v2.py \
    --area "${AREA}" --mode streets \
    --backend raycasting --street-spacing 1.5 --sky-patches 145 \
    --n-jobs "${N_JOBS}" --checkpoint \
&& echo "  SVF streets: OK" \
|| { echo "  SVF streets: FAILED"; FAILED+=("svf-streets"); }

# 3. Urban morphology
echo "[3/4] Urban morphology: ${AREA}"
python scripts/compute_urban_morphology.py \
    --area "${AREA}" --cell-size 50 --floor-height 3.0 \
&& echo "  Morphology: OK" \
|| { echo "  Morphology: FAILED"; FAILED+=("morphology"); }

# 4. Morphology metrics
echo "[4/4] Morphology metrics: ${AREA}"
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
