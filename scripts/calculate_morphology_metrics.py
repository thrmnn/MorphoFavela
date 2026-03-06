#!/usr/bin/env python3
"""
Calculate extended morphology metrics for building footprints.

Usage:
    python scripts/calculate_morphology_metrics.py --area riodaspedras
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA, OUTPUTS_DIR, MAX_FILTER_HEIGHT, MAX_FILTER_AREA,
    MAX_FILTER_VOLUME, MAX_FILTER_HW_RATIO, MAX_HEIGHT_AREA_RATIO, HEIGHT_AREA_PERCENTILE,
    get_area_data_dir, get_area_analysis_dir, is_formal_area
)
from src.metrics import calculate_basic_metrics, validate_footprints, normalize_height_columns
from src.morphology_metrics import calculate_morphology_metrics
from src.visualize import create_metric_map, create_metric_distributions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_summary_stats(gdf: gpd.GeoDataFrame, metrics: list[str]) -> pd.DataFrame:
    available_cols = [col for col in metrics if col in gdf.columns]
    stats = gdf[available_cols].describe()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate extended morphology metrics")
    parser.add_argument("--area", type=str, default=None, help="Area name (vidigal_tls, riodaspedras)")
    parser.add_argument("--input", type=str, default=None, help="Input footprints file path")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--streets", type=str, default=None, help="Optional streets file path for facade ratio")
    parser.add_argument("--street-buffer", type=float, default=1.0, help="Street buffer distance (m)")
    parser.add_argument("--voronoi-buffer", type=float, default=50.0, help="Voronoi envelope buffer distance (m)")
    parser.add_argument("--weight-mode", type=str, default="adjacent", help="Weighting mode for d_w (adjacent)")
    args = parser.parse_args()

    logger.info("Starting extended morphology analysis")

    # 1. Load data
    def _find_buildings_file(area_dir: Path) -> Path | None:
        candidates = []
        for pattern in ["*.shp", "*.gpkg", "*.geojson"]:
            candidates.extend(list(area_dir.glob(pattern)))
        building_candidates = [c for c in candidates if "building" in c.name.lower()]
        if building_candidates:
            return building_candidates[0]
        return candidates[0] if candidates else None

    if args.input:
        input_path = Path(args.input)
    elif args.area:
        area_data_dir = get_area_data_dir(args.area)
        input_path = _find_buildings_file(area_data_dir)
    else:
        input_path = _find_buildings_file(RAW_DATA)

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            "No geospatial file found. Use --input PATH or --area AREA to specify data location."
        )

    logger.info(f"Loading footprints from {input_path}")
    buildings = gpd.read_file(input_path)
    logger.info(f"Loaded {len(buildings)} buildings")

    # 2. Validate
    logger.info("Validating data...")
    is_valid, issues = validate_footprints(buildings)
    if issues:
        logger.warning("Validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        critical_issues = [i for i in issues if any(keyword in i.lower()
                          for keyword in ['missing', 'crs', 'geometry', 'null values'])]
        if critical_issues:
            logger.error("Critical validation errors found. Please fix data issues before proceeding.")
            return

    logger.info("✓ Data validation passed")

    # 3. Preprocess
    logger.info("Preprocessing data...")
    buildings = normalize_height_columns(buildings)

    apply_filtering = True
    if args.area:
        if is_formal_area(args.area):
            apply_filtering = False
            logger.info(f"Formal area detected ({args.area}): Skipping building filters")
        else:
            logger.info(f"Informal area detected ({args.area}): Applying building filters")

    # 3.1 Height filter (informal only)
    if apply_filtering and MAX_FILTER_HEIGHT is not None:
        initial_count = len(buildings)
        buildings['_temp_height'] = buildings['top_height'] - buildings['base_height']
        buildings = buildings[buildings['_temp_height'] <= MAX_FILTER_HEIGHT].copy()
        buildings = buildings.drop(columns=['_temp_height'])
        removed = initial_count - len(buildings)
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with height > {MAX_FILTER_HEIGHT}m")

    # 4. Basic metrics (needed for some morphology metrics)
    logger.info("Calculating basic metrics...")
    buildings = calculate_basic_metrics(buildings)

    # 4.1 Area filter
    if apply_filtering and MAX_FILTER_AREA is not None and 'area' in buildings.columns:
        initial_count = len(buildings)
        buildings = buildings[buildings['area'] <= MAX_FILTER_AREA].copy()
        removed = initial_count - len(buildings)
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with area > {MAX_FILTER_AREA}m²")

    # 4.2 Volume filter
    if apply_filtering and MAX_FILTER_VOLUME is not None and 'volume' in buildings.columns:
        initial_count = len(buildings)
        buildings = buildings[buildings['volume'] <= MAX_FILTER_VOLUME].copy()
        removed = initial_count - len(buildings)
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with volume > {MAX_FILTER_VOLUME}m³")

    # 4.3 H/W ratio filter
    if apply_filtering and MAX_FILTER_HW_RATIO is not None and 'hw_ratio' in buildings.columns:
        initial_count = len(buildings)
        hw_valid = buildings['hw_ratio'].notna()
        buildings = buildings[(~hw_valid) | (buildings['hw_ratio'] <= MAX_FILTER_HW_RATIO)].copy()
        removed = initial_count - len(buildings)
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with h/w ratio > {MAX_FILTER_HW_RATIO}")

    # 4.4 Height/area outlier filter
    if apply_filtering and 'height' in buildings.columns and 'area' in buildings.columns:
        initial_count = len(buildings)
        buildings['height_area_ratio'] = buildings['height'] / buildings['area']
        valid_mask = buildings['height_area_ratio'].notna() & (buildings['area'] > 0)
        if HEIGHT_AREA_PERCENTILE is not None and valid_mask.any():
            threshold = buildings[valid_mask]['height_area_ratio'].quantile(HEIGHT_AREA_PERCENTILE / 100.0)
            buildings = buildings[buildings['height_area_ratio'] <= threshold].copy()
            logger.info(f"Filtered height/area outliers using {HEIGHT_AREA_PERCENTILE}th percentile (threshold: {threshold:.4f})")
        elif MAX_HEIGHT_AREA_RATIO is not None:
            buildings = buildings[buildings['height_area_ratio'] <= MAX_HEIGHT_AREA_RATIO].copy()
            logger.info(f"Filtered height/area outliers (max ratio: {MAX_HEIGHT_AREA_RATIO})")
        if 'height_area_ratio' in buildings.columns:
            buildings = buildings.drop(columns=['height_area_ratio'])
        removed = initial_count - len(buildings)
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with extreme height/area ratios")

    # 5. Load streets (optional)
    streets = None
    if args.streets:
        streets_path = Path(args.streets)
        if streets_path.exists():
            streets = gpd.read_file(streets_path)
            logger.info(f"Loaded streets from {streets_path} ({len(streets)} features)")
        else:
            logger.warning(f"Streets file not found: {streets_path}")

    # 6. Morphology metrics
    logger.info("Calculating morphology metrics...")
    buildings = calculate_morphology_metrics(
        buildings,
        streets=streets,
        street_buffer=args.street_buffer,
        voronoi_buffer=args.voronoi_buffer,
        weight_mode=args.weight_mode,
    )
    logger.info(f"✓ Calculated morphology metrics for {len(buildings)} buildings")

    # 7. Output directory
    if args.output:
        output_base = Path(args.output)
    elif args.area:
        output_base = get_area_analysis_dir(args.area, "morphology_metrics")
    else:
        output_base = OUTPUTS_DIR / "morphology_metrics"
    output_base.mkdir(parents=True, exist_ok=True)

    # 8. Save GeoPackage
    output_gpkg = output_base / "buildings_with_morphology_metrics.gpkg"
    buildings.to_file(output_gpkg, driver="GPKG")
    logger.info(f"Saved metrics to {output_gpkg}")

    # 9. Summary statistics
    metric_cols = [
        "area", "perimeter", "longest_axis_length",
        "shape_index", "compactness_weighted_axis", "convexity",
        "shared_walls", "perimeter_wall", "num_corners",
        "equivalent_rectangular_index", "rectangularity", "squareness",
        "square_compactness", "elongation", "cwt",
        "mean_distance_between_buildings", "inter_building_distance",
        "building_adjacency", "covered_area_ratio",
        "tessellation_area", "cell_alignment", "tessellation_neighbors",
        "fractal_dimension", "average_weighted_distance", "facade_ratio",
    ]
    stats = generate_summary_stats(buildings, metric_cols)
    stats.to_csv(output_base / "morphology_summary_stats.csv")

    # 10. Visualizations
    maps_dir = output_base / "maps"
    maps_dir.mkdir(exist_ok=True)
    for metric in metric_cols:
        if metric in buildings.columns:
            create_metric_map(
                buildings,
                metric,
                maps_dir / f"{metric}_map.png",
                cmap="viridis",
                title=metric.replace("_", " ").title()
            )

    create_metric_distributions(
        buildings,
        metric_cols,
        output_base / "morphology_distributions.png"
    )

    logger.info("✓ Morphology analysis complete")


if __name__ == "__main__":
    main()
