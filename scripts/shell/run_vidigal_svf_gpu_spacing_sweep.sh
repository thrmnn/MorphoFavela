#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/theo/MorphoFavela"
BASE_OUTPUT="/home/theo/MorphoFavela/outputs/vidigal_tls"

STL="/home/theo/MorphoFavela/data/vidigal_tls/raw/full_scan.stl"
ROADS="/home/theo/MorphoFavela/data/vidigal_tls/raw/roads_vidigal.shp"
FOOTPRINTS="/home/theo/MorphoFavela/data/vidigal_tls/raw/vidigal_buildings.shp"
DTM="/home/theo/MorphoFavela/data/vidigal_tls/raw/vidigal_dtm_cropped.tif"

SPACINGS=(0.5 1 1.5 3)
SKY_PATCHES=300

source /home/theo/miniconda3/etc/profile.d/conda.sh
conda activate morphofavela

cd "${PROJECT_ROOT}"

echo "Starting Vidigal_TLS SVF GPU spacing sweep..."
echo "Spacings: ${SPACINGS[*]}"
echo "Sky patches: ${SKY_PATCHES}"

for spacing in "${SPACINGS[@]}"; do
  OUT_DIR="${BASE_OUTPUT}/svf_spacing${spacing}_sp${SKY_PATCHES}"
  mkdir -p "${OUT_DIR}"

  echo ""
  echo "============================================================"
  echo "Running spacing=${spacing} -> ${OUT_DIR}"
  echo "============================================================"

  python "${PROJECT_ROOT}/scripts/compute_svf_streets_gpu.py" \
    --stl "${STL}" \
    --roads "${ROADS}" \
    --dtm "${DTM}" \
    --footprints "${FOOTPRINTS}" \
    --area vidigal_tls \
    --spacing "${spacing}" \
    --height 1.5 \
    --sky-patches "${SKY_PATCHES}" \
    --use-gpu \
    --gpu-batch-size 200 \
    --gpu-ray-chunk-size 1024 \
    --gpu-tri-chunk-size 4096 \
    --output-dir "${OUT_DIR}"

  # Third visualization: explicit minmax + isolines map as a separate file.
  python "${PROJECT_ROOT}/scripts/plot_street_svf_with_isolines.py" \
    --segments "${OUT_DIR}/street_svf_segments.gpkg" \
    --dtm "${DTM}" \
    --footprints "${FOOTPRINTS}" \
    --reference-roads "${ROADS}" \
    --levels 18 \
    --scale-mode minmax \
    --clip-percentile 95 \
    --output "${OUT_DIR}/street_svf_map_minmax_isolines.png"

  echo "Completed spacing=${spacing}"
  echo "  - ${OUT_DIR}/street_svf_map.png"
  echo "  - ${OUT_DIR}/street_svf_map_minmax.png"
  echo "  - ${OUT_DIR}/street_svf_map_minmax_isolines.png"
done

echo ""
echo "All runs completed."
echo "Results root: ${BASE_OUTPUT}"
