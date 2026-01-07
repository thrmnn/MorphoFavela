#!/usr/bin/env python3
"""
Calculate basic morphometric metrics for building footprints.

Usage:
    python scripts/calculate_metrics.py [--area AREA] [--input INPUT_PATH] [--output OUTPUT_DIR]
"""

import logging
import sys
import argparse
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
from src.visualize import (
    create_thematic_maps,
    create_multi_panel_summary,
    create_statistical_distributions,
    create_scatter_plots
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_summary_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for all metrics.
    
    Args:
        gdf: GeoDataFrame with metric columns
        
    Returns:
        DataFrame with summary statistics
    """
    metric_cols = ['height', 'area', 'volume', 'perimeter', 'hw_ratio', 'inter_building_distance']
    available_cols = [col for col in metric_cols if col in gdf.columns]
    
    stats = gdf[available_cols].describe()
    return stats


def main():
    """Run basic metrics calculation pipeline."""
    
    parser = argparse.ArgumentParser(description='Calculate basic morphometric metrics')
    parser.add_argument('--area', type=str, default=None, help='Area name (vidigal, copacabana)')
    parser.add_argument('--input', type=str, default=None, help='Input footprints file path')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    logger.info("Starting morphometric analysis")
    
    # 1. Load data
    if args.input:
        input_path = Path(args.input)
    elif args.area:
        # Use area-based data directory
        area_data_dir = get_area_data_dir(args.area)
        input_path = None
        for pattern in ["*.shp", "*.gpkg", "*.geojson"]:
            matches = list(area_data_dir.glob(pattern))
            if matches:
                input_path = matches[0]
                break
    else:
        # Fallback to legacy RAW_DATA
        input_path = None
        for pattern in ["footprints.gpkg", "footprints.geojson", "footprints.shp", "*.gpkg", "*.geojson", "*.shp"]:
            matches = list(RAW_DATA.glob(pattern))
            if matches:
                input_path = matches[0]
                break
    
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            f"No geospatial file found. "
            f"Use --input PATH or --area AREA to specify data location."
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
        # Only fail on critical issues (missing columns, CRS, geometry)
        critical_issues = [i for i in issues if any(keyword in i.lower() 
                          for keyword in ['missing', 'crs', 'geometry', 'null values'])]
        if critical_issues:
            logger.error("Critical validation errors found. Please fix data issues before proceeding.")
            return
    
    logger.info("✓ Data validation passed")
    
    # 2.5. Preprocess: Normalize columns and filter by height
    logger.info("Preprocessing data...")
    buildings = normalize_height_columns(buildings)
    
    # Determine if filtering should be applied
    # Formal areas (e.g., Copacabana) skip filtering
    # Informal areas (e.g., Vidigal) apply filtering
    apply_filtering = True
    if args.area:
        if is_formal_area(args.area):
            apply_filtering = False
            logger.info(f"Formal area detected ({args.area}): Skipping building filters")
        else:
            logger.info(f"Informal area detected ({args.area}): Applying building filters")
    
    # Filter buildings by maximum height (only for informal areas)
    if apply_filtering and MAX_FILTER_HEIGHT is not None:
        initial_count = len(buildings)
        # Calculate height for filtering
        buildings['_temp_height'] = buildings['top_height'] - buildings['base_height']
        buildings = buildings[buildings['_temp_height'] <= MAX_FILTER_HEIGHT].copy()
        buildings = buildings.drop(columns=['_temp_height'])
        filtered_count = len(buildings)
        removed = initial_count - filtered_count
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with height > {MAX_FILTER_HEIGHT}m")
            logger.info(f"Processing {filtered_count} buildings after height filter")
        else:
            logger.info(f"All buildings within height limit ({MAX_FILTER_HEIGHT}m)")
    
    # 3. Calculate metrics
    logger.info("Calculating basic metrics...")
    buildings = calculate_basic_metrics(buildings)
    logger.info(f"✓ Calculated metrics for {len(buildings)} buildings")
    
    # 3.5. Filter buildings by maximum area (only for informal areas)
    if apply_filtering and MAX_FILTER_AREA is not None and 'area' in buildings.columns:
        initial_count = len(buildings)
        buildings = buildings[buildings['area'] <= MAX_FILTER_AREA].copy()
        filtered_count = len(buildings)
        removed = initial_count - filtered_count
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with area > {MAX_FILTER_AREA}m²")
            logger.info(f"Processing {filtered_count} buildings after area filter")
    
    # 3.6. Filter buildings by maximum volume (only for informal areas)
    if apply_filtering and MAX_FILTER_VOLUME is not None and 'volume' in buildings.columns:
        initial_count = len(buildings)
        buildings = buildings[buildings['volume'] <= MAX_FILTER_VOLUME].copy()
        filtered_count = len(buildings)
        removed = initial_count - filtered_count
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with volume > {MAX_FILTER_VOLUME}m³")
            logger.info(f"Processing {filtered_count} buildings after volume filter")
    
    # 3.7. Filter buildings by maximum h/w ratio (only for informal areas)
    if apply_filtering and MAX_FILTER_HW_RATIO is not None and 'hw_ratio' in buildings.columns:
        initial_count = len(buildings)
        hw_valid = buildings['hw_ratio'].notna()
        buildings = buildings[(~hw_valid) | (buildings['hw_ratio'] <= MAX_FILTER_HW_RATIO)].copy()
        filtered_count = len(buildings)
        removed = initial_count - filtered_count
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with h/w ratio > {MAX_FILTER_HW_RATIO}")
            logger.info(f"Processing {filtered_count} buildings after h/w ratio filter")
    
    # 3.8. Filter outliers based on height vs area relationship (only for informal areas)
    if apply_filtering and 'height' in buildings.columns and 'area' in buildings.columns:
        initial_count = len(buildings)
        # Calculate height/area ratio
        buildings['height_area_ratio'] = buildings['height'] / buildings['area']
        valid_mask = buildings['height_area_ratio'].notna() & (buildings['area'] > 0)
        
        if HEIGHT_AREA_PERCENTILE is not None and valid_mask.any():
            # Use percentile-based filtering
            threshold = buildings[valid_mask]['height_area_ratio'].quantile(HEIGHT_AREA_PERCENTILE / 100.0)
            buildings = buildings[buildings['height_area_ratio'] <= threshold].copy()
            logger.info(f"Filtered height/area outliers using {HEIGHT_AREA_PERCENTILE}th percentile (threshold: {threshold:.4f})")
        elif MAX_HEIGHT_AREA_RATIO is not None:
            # Use fixed threshold
            buildings = buildings[buildings['height_area_ratio'] <= MAX_HEIGHT_AREA_RATIO].copy()
            logger.info(f"Filtered height/area outliers (max ratio: {MAX_HEIGHT_AREA_RATIO})")
        
        # Remove temporary column
        if 'height_area_ratio' in buildings.columns:
            buildings = buildings.drop(columns=['height_area_ratio'])
        
        filtered_count = len(buildings)
        removed = initial_count - filtered_count
        if removed > 0:
            logger.info(f"Filtered out {removed} buildings with extreme height/area ratios")
            logger.info(f"Processing {filtered_count} buildings after height/area filter")
    
    # 4. Generate statistics
    logger.info("Generating summary statistics...")
    stats = generate_summary_stats(buildings)
    
    # Determine output directory
    if args.output:
        output_base = Path(args.output)
    elif args.area:
        output_base = get_area_analysis_dir(args.area, "metrics")
    else:
        output_base = OUTPUTS_DIR
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 5. Create visualizations
    logger.info("Creating visualizations...")
    maps_dir = output_base / "maps"
    maps_dir.mkdir(exist_ok=True)
    
    # Original height/volume maps
    logger.info("  - Creating height/volume maps...")
    create_thematic_maps(
        buildings,
        maps_dir / "height_volume_maps.png"
    )
    
    # Multi-panel summary
    logger.info("  - Creating multi-panel summary...")
    create_multi_panel_summary(
        buildings,
        maps_dir / "multi_panel_summary.png"
    )
    
    # Statistical distributions
    logger.info("  - Creating statistical distributions...")
    create_statistical_distributions(
        buildings,
        maps_dir / "statistical_distributions.png"
    )
    
    # Scatter plots
    logger.info("  - Creating scatter plots...")
    create_scatter_plots(
        buildings,
        maps_dir / "scatter_plots.png"
    )
    
    # 6. Save outputs
    logger.info("Saving outputs...")
    
    # Save enhanced dataset
    output_path = output_base / "buildings_with_metrics.gpkg"
    buildings.to_file(output_path, driver="GPKG")
    logger.info(f"✓ Saved metrics to {output_path}")
    
    # Save statistics
    stats_path = output_base / "summary_stats.csv"
    stats.to_csv(stats_path)
    logger.info(f"✓ Saved statistics to {stats_path}")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Metrics: {output_path}")
    logger.info(f"  Stats:   {stats_path}")
    logger.info(f"  Maps:    {maps_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

