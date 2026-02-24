#!/bin/bash
# Test GPU-accelerated SVF computation for riodaspedras
# Compares with previous CPU results

set -e

# Source conda environment
source /home/theo/miniconda3/etc/profile.d/conda.sh
conda activate IVF

# Data paths
STL="data/riodaspedras/raw/full_scan.stl"
ROADS="data/riodaspedras/raw/roads_riodaspedras.shp"
FOOTPRINTS="data/riodaspedras/raw/riodaspedras_buildings.shp"
AREA="riodaspedras"

# Output directories
CPU_OUTPUT="outputs/riodaspedras/svf_streets"
GPU_OUTPUT="outputs/riodaspedras/svf_streets_gpu"
COMPARISON_OUTPUT="outputs/riodaspedras/svf_streets_comparison"

echo "=========================================="
echo "GPU SVF COMPUTATION TEST - RIODASPEDRAS"
echo "=========================================="
echo ""

# Check if GPU is available
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('✗ GPU not available - will use CPU')
" || echo "PyTorch not installed - will use CPU fallback"

echo ""
echo "Step 1: Running GPU-accelerated SVF computation..."
python scripts/compute_svf_streets_gpu.py \
    --stl "$STL" \
    --roads "$ROADS" \
    --footprints "$FOOTPRINTS" \
    --area "$AREA" \
    --spacing 3.0 \
    --height 1.5 \
    --sky-patches 145 \
    --use-gpu \
    --gpu-batch-size 100 \
    --gpu-samples-per-ray 50

echo ""
echo "Step 2: Comparing with CPU results..."
if [ -f "$CPU_OUTPUT/street_svf_points.gpkg" ]; then
    python scripts/compare_svf_cpu_gpu.py \
        --cpu-results "$CPU_OUTPUT/street_svf_points.gpkg" \
        --gpu-results "$GPU_OUTPUT/street_svf_points.gpkg" \
        --output-dir "$COMPARISON_OUTPUT"
    echo ""
    echo "✓ Comparison complete! Results saved to: $COMPARISON_OUTPUT"
else
    echo "⚠ CPU results not found at: $CPU_OUTPUT/street_svf_points.gpkg"
    echo "  Skipping comparison. Run CPU version first if you want to compare."
fi

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo "GPU results: $GPU_OUTPUT"
if [ -d "$COMPARISON_OUTPUT" ]; then
    echo "Comparison results: $COMPARISON_OUTPUT"
fi
