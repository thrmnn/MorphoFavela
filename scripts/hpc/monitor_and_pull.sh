#!/usr/bin/env bash
# ============================================================================
# Monitor HPC jobs, pull results when done, and generate visualizations.
#
# Usage:
#   bash scripts/hpc/monitor_and_pull.sh                # monitor all MorphoFavela jobs
#   bash scripts/hpc/monitor_and_pull.sh --once          # check once, don't loop
#   bash scripts/hpc/monitor_and_pull.sh --poll 120      # check every 120s (default: 300)
#
# What it does:
#   1. Polls SLURM queue for your MorphoFavela pipeline jobs
#   2. When a job completes, pulls outputs via rsync
#   3. Runs local visualization generation for completed areas
#   4. Logs everything to logs/monitor.log
#
# Prerequisites:
#   - SSH mux session active: ssh orcd (from another terminal)
#   - Or key auth configured in ~/.ssh/config
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

POLL_INTERVAL=300
ONCE=false
LOG_FILE="logs/monitor.log"
PULLED_MARKER_DIR="logs/.pulled"

mkdir -p logs "${PULLED_MARKER_DIR}"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --once) ONCE=true; shift ;;
        --poll) POLL_INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

ssh_orcd() {
    ssh -o ConnectTimeout=10 -o BatchMode=yes orcd "$@" 2>/dev/null
}

check_ssh() {
    if ! ssh_orcd "echo ok" >/dev/null 2>&1; then
        log "ERROR: Cannot reach ORCD. Start mux session: ssh orcd"
        return 1
    fi
    return 0
}

# Extract area name from job name (e.g., "rocinha_pipeline" -> "rocinha")
job_to_area() {
    echo "$1" | sed 's/_pipeline$//'
}

# ---------------------------------------------------------------------------
# Pull results for a completed area
# ---------------------------------------------------------------------------
pull_area() {
    local area="$1"
    log "Pulling outputs for ${area}..."

    rsync -avz \
        --include='*/' \
        --include='*.gpkg' --include='*.csv' --include='*.json' \
        --include='*.png' --include='*.tif' --include='*.html' \
        --exclude='*.stl' --exclude='*.vtk' --exclude='*.npz' \
        --exclude='*' \
        -e ssh \
        "orcd:~/MorphoFavela/outputs/${area}/" \
        "outputs/${area}/" \
        >> "${LOG_FILE}" 2>&1

    # Also pull logs
    rsync -avz \
        --include="pipeline_*.out" --include="pipeline_*.err" \
        --exclude='*' \
        -e ssh \
        "orcd:~/MorphoFavela/logs/" \
        "logs/" \
        >> "${LOG_FILE}" 2>&1

    log "Pull complete for ${area}"
}

# ---------------------------------------------------------------------------
# Generate visualizations for a completed area
# ---------------------------------------------------------------------------
generate_visualizations() {
    local area="$1"
    log "Generating visualizations for ${area}..."

    # SVF visualizations (if SVF outputs exist)
    if [[ -f "outputs/${area}/svf_v2/svf_streets.gpkg" ]]; then
        python -c "
import geopandas as gpd
from src.svf_v2.visualize import (
    plot_svf_dashboard,
    plot_svf_distribution,
    plot_svf_interactive,
)
import os

area = '${area}'
out = f'outputs/{area}/svf_v2'

# Load data
pts = gpd.read_file(f'{out}/svf_streets.gpkg')
segs = gpd.read_file(f'{out}/svf_streets_segments.gpkg') if os.path.exists(f'{out}/svf_streets_segments.gpkg') else None

# Dashboard
try:
    plot_svf_dashboard(pts, segs, f'{out}/svf_dashboard.png')
    print(f'  Dashboard saved: {out}/svf_dashboard.png')
except Exception as e:
    print(f'  Dashboard skipped: {e}')

# Distribution
try:
    plot_svf_distribution(pts['svf'].values, f'{out}/svf_distribution_detailed.png')
    print(f'  Distribution saved: {out}/svf_distribution_detailed.png')
except Exception as e:
    print(f'  Distribution skipped: {e}')

# Interactive map
try:
    plot_svf_interactive(pts, f'{out}/svf_interactive.html')
    print(f'  Interactive saved: {out}/svf_interactive.html')
except Exception as e:
    print(f'  Interactive skipped: {e}')

print(f'  SVF viz complete for {area}')
" >> "${LOG_FILE}" 2>&1 && log "  SVF visualizations done" || log "  SVF visualizations failed (non-fatal)"
    fi

    # Morphology visualizations (if morphology outputs exist)
    if [[ -f "outputs/${area}/urban_morphology/zone_metrics.gpkg" ]]; then
        python -c "
import geopandas as gpd
from src.visualization.morphology import (
    plot_zone_metrics_panel,
    plot_lisa_clusters_panel,
)
import os

area = '${area}'
out = f'outputs/{area}/urban_morphology'

# Zone metrics panel
try:
    gdf = gpd.read_file(f'{out}/zone_metrics.gpkg')
    plot_zone_metrics_panel(gdf, f'{out}/maps/zone_metrics_panel.png')
    print(f'  Zone panel saved')
except Exception as e:
    print(f'  Zone panel skipped: {e}')

# LISA clusters
try:
    lisa = gpd.read_file(f'{out}/lisa_clusters.gpkg')
    plot_lisa_clusters_panel(lisa, f'{out}/maps/')
    print(f'  LISA panels saved')
except Exception as e:
    print(f'  LISA panels skipped: {e}')

print(f'  Morphology viz complete for {area}')
" >> "${LOG_FILE}" 2>&1 && log "  Morphology visualizations done" || log "  Morphology visualizations failed (non-fatal)"
    fi

    log "Visualizations complete for ${area}"
}

# ---------------------------------------------------------------------------
# Generate cross-area comparison when multiple areas are done
# ---------------------------------------------------------------------------
generate_comparison() {
    local -a completed_areas=("$@")

    if [[ ${#completed_areas[@]} -lt 2 ]]; then
        log "Need at least 2 completed areas for comparison, skipping"
        return
    fi

    log "Generating cross-area SVF comparison for: ${completed_areas[*]}"

    python -c "
import geopandas as gpd
from src.svf_v2.visualize import plot_svf_comparison
import os

areas = '${completed_areas[*]}'.split()
data = {}
for area in areas:
    f = f'outputs/{area}/svf_v2/svf_streets.gpkg'
    if os.path.exists(f):
        gdf = gpd.read_file(f)
        data[area] = gdf['svf'].values
        print(f'  Loaded {area}: {len(gdf)} points, mean SVF={gdf[\"svf\"].mean():.3f}')

if len(data) >= 2:
    os.makedirs('outputs/comparative', exist_ok=True)
    plot_svf_comparison(data, 'outputs/comparative/svf_comparison.png')
    print('  Comparison saved: outputs/comparative/svf_comparison.png')
else:
    print('  Not enough areas with SVF data for comparison')
" >> "${LOG_FILE}" 2>&1 && log "  Cross-area comparison done" || log "  Cross-area comparison failed (non-fatal)"
}

# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------
log "=========================================="
log "HPC Job Monitor started"
log "Poll interval: ${POLL_INTERVAL}s"
log "=========================================="

while true; do
    if ! check_ssh; then
        if $ONCE; then exit 1; fi
        log "Retrying in ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
        continue
    fi

    # Get current job status
    QUEUE_OUTPUT=$(ssh_orcd "squeue -u thermann --name='rocinha_pipeline,riodaspedras_pipeline,vidigal_pipeline,vidigal_tls_pipeline' --format='%j %T' --noheader" 2>/dev/null || echo "")

    # Get recently completed jobs (last 24h)
    COMPLETED=$(ssh_orcd "sacct -u thermann --starttime=now-1day --format='JobName%30,State' --noheader --parsable2" 2>/dev/null | grep 'COMPLETED' | cut -d'|' -f1 | sort -u || echo "")

    # Track which areas are running, pending, completed
    RUNNING_AREAS=()
    PENDING_AREAS=()
    NEWLY_COMPLETED=()
    ALL_COMPLETED=()

    for area in rocinha riodaspedras vidigal vidigal_tls; do
        job_name="${area}_pipeline"
        marker="${PULLED_MARKER_DIR}/${area}.done"

        if echo "${QUEUE_OUTPUT}" | grep -q "${job_name}.*RUNNING"; then
            RUNNING_AREAS+=("${area}")
        elif echo "${QUEUE_OUTPUT}" | grep -q "${job_name}.*PENDING"; then
            PENDING_AREAS+=("${area}")
        elif echo "${COMPLETED}" | grep -q "${job_name}" && [[ ! -f "${marker}" ]]; then
            NEWLY_COMPLETED+=("${area}")
        fi

        # Track all completed (including previously pulled)
        if echo "${COMPLETED}" | grep -q "${job_name}" || [[ -f "${marker}" ]]; then
            ALL_COMPLETED+=("${area}")
        fi
    done

    # Status report
    log "Status: ${#RUNNING_AREAS[@]} running [${RUNNING_AREAS[*]:-none}] | ${#PENDING_AREAS[@]} pending [${PENDING_AREAS[*]:-none}] | ${#NEWLY_COMPLETED[@]} newly done [${NEWLY_COMPLETED[*]:-none}]"

    # Process newly completed areas
    for area in "${NEWLY_COMPLETED[@]}"; do
        log "=== ${area} COMPLETED — processing ==="
        pull_area "${area}"
        generate_visualizations "${area}"
        touch "${PULLED_MARKER_DIR}/${area}.done"
        log "=== ${area} fully processed ==="
    done

    # Cross-area comparison if we have ≥2 completed
    if [[ ${#NEWLY_COMPLETED[@]} -gt 0 && ${#ALL_COMPLETED[@]} -ge 2 ]]; then
        generate_comparison "${ALL_COMPLETED[@]}"
    fi

    # All done?
    if [[ ${#RUNNING_AREAS[@]} -eq 0 && ${#PENDING_AREAS[@]} -eq 0 ]]; then
        if [[ ${#ALL_COMPLETED[@]} -gt 0 ]]; then
            log "=========================================="
            log "ALL HPC JOBS COMPLETE"
            log "Completed areas: ${ALL_COMPLETED[*]}"
            log "=========================================="

            # Final cross-area comparison
            if [[ ${#ALL_COMPLETED[@]} -ge 2 ]]; then
                generate_comparison "${ALL_COMPLETED[@]}"
            fi
        else
            log "No MorphoFavela pipeline jobs found in queue or recent history"
        fi
        break
    fi

    if $ONCE; then
        log "Single check complete (--once mode)"
        break
    fi

    sleep "${POLL_INTERVAL}"
done

log "Monitor finished"
