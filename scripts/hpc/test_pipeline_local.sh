#!/usr/bin/env bash
# ============================================================================
# Local pipeline smoke test — run before HPC submission
#
# Tests each pipeline step on vidigal_tls (smallest area, 436 buildings)
# with reduced parameters to complete in ~5 min.
#
# Usage:
#   bash scripts/hpc/test_pipeline_local.sh           # full smoke test
#   bash scripts/hpc/test_pipeline_local.sh svf        # SVF only
#   bash scripts/hpc/test_pipeline_local.sh morphology  # morphology only
#   bash scripts/hpc/test_pipeline_local.sh typology    # typology only
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TEST_AREA="vidigal_tls"
TEST_MODE="${1:-all}"
FAILED=()

echo "============================================================"
echo "  Local Pipeline Smoke Test"
echo "  Area: ${TEST_AREA}"
echo "  Mode: ${TEST_MODE}"
echo "  Start: $(date -Iseconds)"
echo "============================================================"
echo ""

# ---- Step 0: Unit tests ----
if [[ "${TEST_MODE}" == "all" || "${TEST_MODE}" == "tests" ]]; then
    echo "[0/4] Running unit tests..."
    python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
    echo ""
fi

# ---- Step 1: SVF (reduced parameters) ----
if [[ "${TEST_MODE}" == "all" || "${TEST_MODE}" == "svf" ]]; then
    echo "[1/4] SVF smoke test: ${TEST_AREA} (streets only, spacing=5m, 2 jobs)"
    python scripts/run_svf_v2.py \
        --area "${TEST_AREA}" \
        --mode streets \
        --backend raycasting \
        --street-spacing 5.0 \
        --sky-patches 145 \
        --n-jobs 2 \
        --checkpoint \
    && echo "  SVF: OK" \
    || { echo "  SVF: FAILED"; FAILED+=("svf"); }
    echo ""
fi

# ---- Step 2: Morphology ----
if [[ "${TEST_MODE}" == "all" || "${TEST_MODE}" == "morphology" ]]; then
    echo "[2/4] Morphology smoke test: ${TEST_AREA}"
    python scripts/compute_urban_morphology.py \
        --area "${TEST_AREA}" \
        --cell-size 100 \
        --floor-height 3.0 \
    && echo "  Morphology: OK" \
    || { echo "  Morphology: FAILED"; FAILED+=("morphology"); }
    echo ""
fi

# ---- Step 3: Typology ----
if [[ "${TEST_MODE}" == "all" || "${TEST_MODE}" == "typology" ]]; then
    echo "[3/4] Typology smoke test: ${TEST_AREA}"
    python scripts/classify_typology.py \
        --areas "${TEST_AREA}" \
    && echo "  Typology: OK" \
    || { echo "  Typology: FAILED"; FAILED+=("typology"); }
    echo ""
fi

# ---- Summary ----
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  SMOKE TEST PASSED — safe to submit to HPC"
else
    echo "  SMOKE TEST FAILED: ${FAILED[*]}"
    echo "  DO NOT submit to HPC until these are fixed"
fi
echo "  Finished: $(date -Iseconds)"
echo "============================================================"

[[ ${#FAILED[@]} -eq 0 ]]
