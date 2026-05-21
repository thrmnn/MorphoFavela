#!/usr/bin/env bash
# ============================================================================
# Sync MorphoFavela repo + data to HPC cluster, and pull results back.
#
# Usage:
#   # Push code + data to cluster:
#   ./scripts/hpc/sync_to_cluster.sh push user@cluster:/path/to/MorphoFavela
#
#   # Pull results from cluster:
#   ./scripts/hpc/sync_to_cluster.sh pull user@cluster:/path/to/MorphoFavela
#
#   # Dry-run (see what would transfer):
#   ./scripts/hpc/sync_to_cluster.sh push user@cluster:/path/to/MorphoFavela --dry-run
# ============================================================================

set -euo pipefail

ACTION="${1:-}"
REMOTE="${2:-}"
EXTRA_ARGS="${@:3}"

if [[ -z "$ACTION" || -z "$REMOTE" ]]; then
    echo "Usage: $0 {push|pull} user@host:/remote/path [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 push theo@hpc.uni.edu:~/MorphoFavela"
    echo "  $0 pull theo@hpc.uni.edu:~/MorphoFavela"
    echo "  $0 push theo@hpc.uni.edu:~/MorphoFavela --dry-run"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Areas with raw data needed for SVF computation
AREAS=(vidigal_tls vidigal rocinha riodaspedras complexo_do_alemao maré)

if [[ "$ACTION" == "push" ]]; then
    echo "=== Pushing code + data to $REMOTE ==="

    # 1. Sync code (git-tracked files only, plus essential configs)
    echo "[1/3] Syncing source code..."
    rsync -avz --progress $EXTRA_ARGS \
        --include='/' \
        --include='/src/***' \
        --include='/scripts/***' \
        --include='/tests/***' \
        --include='/pyproject.toml' \
        --include='/requirements.txt' \
        --include='/.gitignore' \
        --include='/CLAUDE.md' \
        --exclude='__pycache__/' \
        --exclude='.pytest_cache/' \
        --exclude='.ruff_cache/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='outputs/' \
        --exclude='data/' \
        "$REPO_ROOT/" "$REMOTE/"

    # 2. Sync raw data for priority areas only (skip large city-wide files)
    echo "[2/3] Syncing raw geodata for: ${AREAS[*]}..."
    for area in "${AREAS[@]}"; do
        local_dir="$REPO_ROOT/data/$area/raw/"
        if [[ -d "$local_dir" ]]; then
            echo "  -> $area ($(du -sh "$local_dir" | cut -f1))"
            rsync -avz --progress $EXTRA_ARGS \
                --exclude='*.stl' \
                --exclude='copacabana/' \
                "$local_dir" "$REMOTE/data/$area/raw/"
        else
            echo "  -> $area (SKIPPED: not found locally)"
        fi
    done

    # 3. Create output directories on remote
    echo "[3/3] Creating output directories..."
    REMOTE_HOST="${REMOTE%%:*}"
    REMOTE_PATH="${REMOTE#*:}"
    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH/outputs/{vidigal_tls,vidigal,rocinha,riodaspedras,complexo_do_alemao,maré}/svf_v2 $REMOTE_PATH/outputs/comparative/typology $REMOTE_PATH/logs"

    echo ""
    echo "=== Push complete ==="
    echo "Next steps on cluster:"
    echo "  cd $REMOTE_PATH"
    echo "  pip install -r requirements.txt"
    echo "  sbatch scripts/hpc/run_svf_batch.sh              # all default areas"
    echo "  sbatch scripts/hpc/run_svf_batch.sh rocinha all   # single area"

elif [[ "$ACTION" == "pull" ]]; then
    echo "=== Pulling results from $REMOTE ==="

    # Pull outputs directory (SVF results, morphology, typology)
    echo "[1/2] Syncing outputs..."
    rsync -avz --progress $EXTRA_ARGS \
        --include='*/' \
        --include='*.gpkg' \
        --include='*.csv' \
        --include='*.json' \
        --include='*.png' \
        --include='*.tif' \
        --exclude='*.stl' \
        --exclude='*.vtk' \
        "$REMOTE/outputs/" "$REPO_ROOT/outputs/"

    # Pull logs
    echo "[2/2] Syncing logs..."
    rsync -avz --progress $EXTRA_ARGS \
        "$REMOTE/logs/" "$REPO_ROOT/logs/" 2>/dev/null || true

    echo ""
    echo "=== Pull complete ==="
    echo "Results in: $REPO_ROOT/outputs/"

else
    echo "Unknown action: $ACTION. Use 'push' or 'pull'."
    exit 1
fi
