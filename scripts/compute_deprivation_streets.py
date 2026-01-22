#!/usr/bin/env python3
"""
Compute street-level deprivation index from SVF and solar access data.

This script computes a composite deprivation index at street level by combining
SVF deficit and solar access deficit. The deprivation index ranges from 0-1,
where higher values indicate more environmental deprivation.

Usage:
    python scripts/compute_deprivation_streets.py --area vidigal
    python scripts/compute_deprivation_streets.py --area copacabana
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_area_analysis_dir, get_area_data_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_deprivation_index(svf_values: np.ndarray, solar_values: np.ndarray) -> tuple:
    """
    Compute deprivation index from SVF and solar access values.
    
    Args:
        svf_values: Array of SVF values (0-1)
        solar_values: Array of solar access values (hours)
        
    Returns:
        Tuple of (deprivation_index, svf_deficit, solar_deficit)
    """
    # Calculate deficits
    svf_deficit = 1.0 - svf_values  # Lower SVF = higher deficit
    solar_deficit = np.maximum(0, (3.0 - solar_values) / 3.0)  # Below 3h = deficit
    
    # Combined deprivation index (0-1, higher = more deprived)
    # Weight: 50% SVF deficit, 50% solar deficit
    deprivation_index = 0.5 * svf_deficit + 0.5 * solar_deficit
    
    return deprivation_index, svf_deficit, solar_deficit


def create_street_deprivation_map(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame = None,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None
):
    """
    Create a map showing streets colored by deprivation index values.
    
    Args:
        segments_gdf: GeoDataFrame with segment-level deprivation statistics
        points_gdf: Optional GeoDataFrame with point-level deprivation (for detailed view)
        building_footprints: Optional building footprints for context
        output_path: Path to save the map
    """
    logger.info("Creating street deprivation map...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot building footprints as background (if available)
    if building_footprints is not None:
        building_footprints.plot(
            ax=ax, facecolor='lightgrey', edgecolor='black', 
            linewidth=0.5, alpha=0.5, label='Buildings'
        )
    
    # Use reversed colormap (higher = more deprived = darker/redder)
    import matplotlib
    dep_cmap = matplotlib.colormaps.get_cmap('RdYlGn_r').copy()
    
    # Plot street segments colored by mean deprivation
    segments_gdf.plot(
        ax=ax, column='deprivation_mean', cmap=dep_cmap,
        vmin=0, vmax=1, linewidth=3, legend=True,
        legend_kwds={'label': 'Deprivation Index (0-1)', 'shrink': 0.8},
        label='Street segments'
    )
    
    # Optionally plot points (for detailed view)
    if points_gdf is not None and len(points_gdf) < 1000:  # Only if not too many points
        points_gdf.plot(
            ax=ax, column='deprivation_index', cmap=dep_cmap,
            vmin=0, vmax=1, markersize=2, alpha=0.6, zorder=10
        )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Street-Level Deprivation Index\nColored by Mean Deprivation per Segment', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if building_footprints is not None:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved street deprivation map to {output_path}")


def create_deprivation_distribution_plots(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    output_dir: Path
):
    """
    Create statistical distribution plots for deprivation index.
    
    Args:
        segments_gdf: GeoDataFrame with segment-level statistics
        points_gdf: GeoDataFrame with point-level deprivation
        output_dir: Output directory
    """
    logger.info("Creating deprivation distribution plots...")
    
    # Distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dep_values = points_gdf['deprivation_index'].dropna().values
    ax.hist(dep_values, bins=50, edgecolor='black', alpha=0.7, color='darkred')
    ax.set_xlabel('Deprivation Index', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Street-Level Deprivation Index', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_dep = np.mean(dep_values)
    median_dep = np.median(dep_values)
    ax.axvline(mean_dep, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dep:.3f}')
    ax.axvline(median_dep, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dep:.3f}')
    
    # Add threshold lines
    ax.axvline(0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='High deprivation (>0.5)')
    ax.axvline(0.75, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Extreme deprivation (>0.75)')
    
    ax.legend()
    
    plt.tight_layout()
    hist_path = output_dir / "street_deprivation_distribution.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved distribution plot to {hist_path}")


def main():
    """Compute and save street-level deprivation index."""
    parser = argparse.ArgumentParser(description='Compute street-level deprivation index')
    parser.add_argument('--area', type=str, required=True, 
                       help='Area name (vidigal, copacabana)')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("STREET-LEVEL DEPRIVATION INDEX COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"Area: {args.area}")
    
    # Get output directory
    output_dir = get_area_analysis_dir(args.area, 'deprivation_streets')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load SVF street-level data
    svf_dir = get_area_analysis_dir(args.area, 'svf_streets')
    svf_points_file = svf_dir / 'street_svf_points.gpkg'
    
    if not svf_points_file.exists():
        logger.error(f"SVF street-level data not found: {svf_points_file}")
        logger.error("Please run compute_svf_streets.py first")
        sys.exit(1)
    
    logger.info(f"Loading SVF data from {svf_points_file}")
    svf_points = gpd.read_file(svf_points_file)
    logger.info(f"  Loaded {len(svf_points)} SVF points")
    
    # Load solar street-level data
    solar_dir = get_area_analysis_dir(args.area, 'solar_streets')
    solar_points_file = solar_dir / 'street_solar_points.gpkg'
    
    if not solar_points_file.exists():
        logger.error(f"Solar street-level data not found: {solar_points_file}")
        logger.error("Please run compute_solar_access_streets.py first")
        sys.exit(1)
    
    logger.info(f"Loading solar data from {solar_points_file}")
    solar_points = gpd.read_file(solar_points_file)
    logger.info(f"  Loaded {len(solar_points)} solar points")
    
    # Merge SVF and solar data
    logger.info("Merging SVF and solar data...")
    if 'segment_idx' in svf_points.columns and 'segment_idx' in solar_points.columns:
        # Merge on segment_idx and distance_along
        merged = svf_points.merge(
            solar_points[['segment_idx', 'distance_along', 'solar_hours']],
            on=['segment_idx', 'distance_along'],
            how='inner'
        )
        logger.info(f"  Merged {len(merged)} points")
    else:
        logger.warning("  segment_idx not found, attempting spatial join...")
        # Fallback to spatial join
        merged = gpd.sjoin(
            svf_points, 
            solar_points[['geometry', 'solar_hours']],
            how='inner',
            predicate='dwithin',
            distance=0.1
        )
        logger.info(f"  Merged {len(merged)} points via spatial join")
    
    if len(merged) == 0:
        logger.error("No matching points found between SVF and solar data")
        sys.exit(1)
    
    # Check required columns
    if 'svf' not in merged.columns or 'solar_hours' not in merged.columns:
        logger.error("Required columns (svf, solar_hours) not found in merged data")
        logger.error(f"Available columns: {list(merged.columns)}")
        sys.exit(1)
    
    # Compute deprivation index
    logger.info("Computing deprivation index...")
    svf_vals = merged['svf'].values
    solar_vals = merged['solar_hours'].values
    
    deprivation_index, svf_deficit, solar_deficit = compute_deprivation_index(svf_vals, solar_vals)
    
    # Create output GeoDataFrame
    deprivation_points = merged[['geometry']].copy()
    if 'segment_idx' in merged.columns:
        deprivation_points['segment_idx'] = merged['segment_idx']
    if 'distance_along' in merged.columns:
        deprivation_points['distance_along'] = merged['distance_along']
    
    deprivation_points['deprivation_index'] = deprivation_index
    deprivation_points['svf_deficit'] = svf_deficit
    deprivation_points['solar_deficit'] = solar_deficit
    deprivation_points['svf'] = svf_vals  # Keep original values for reference
    deprivation_points['solar_hours'] = solar_vals  # Keep original values for reference
    
    # Aggregate to segment level
    logger.info("Aggregating to segment level...")
    if 'segment_idx' in deprivation_points.columns:
        segments = deprivation_points.groupby('segment_idx').agg({
            'deprivation_index': ['mean', 'std', 'min', 'max', 'median'],
            'svf_deficit': 'mean',
            'solar_deficit': 'mean',
            'svf': 'mean',
            'solar_hours': 'mean'
        }).reset_index()
        
        # Flatten column names
        segments.columns = ['segment_idx', 'deprivation_mean', 'deprivation_std', 
                           'deprivation_min', 'deprivation_max', 'deprivation_median',
                           'svf_deficit_mean', 'solar_deficit_mean', 
                           'svf_mean', 'solar_hours_mean']
        
        # Get geometry from original segments (try SVF segments first)
        svf_segments_file = svf_dir / 'street_svf_segments.gpkg'
        if svf_segments_file.exists():
            svf_segments = gpd.read_file(svf_segments_file)
            if 'segment_idx' in svf_segments.columns:
                segments = svf_segments[['segment_idx', 'geometry']].merge(
                    segments, on='segment_idx', how='inner'
                )
                segments = gpd.GeoDataFrame(segments, crs=svf_segments.crs)
                logger.info(f"  Created {len(segments)} segment-level records")
    else:
        segments = None
        logger.warning("  Cannot aggregate to segments (segment_idx not available)")
    
    # Save results
    logger.info("Saving results...")
    
    # Point-level output
    points_output = output_dir / "street_deprivation_points.gpkg"
    deprivation_points.to_file(points_output, driver='GPKG')
    logger.info(f"  Saved point-level results to {points_output}")
    
    # Segment-level output (if available)
    if segments is not None:
        segments_output = output_dir / "street_deprivation_segments.gpkg"
        segments.to_file(segments_output, driver='GPKG')
        logger.info(f"  Saved segment-level results to {segments_output}")
    
    # CSV summary statistics
    stats = {
        'n_points': len(deprivation_points),
        'n_segments': len(segments) if segments is not None else 0,
        'deprivation_mean': float(deprivation_index.mean()),
        'deprivation_std': float(deprivation_index.std()),
        'deprivation_min': float(deprivation_index.min()),
        'deprivation_max': float(deprivation_index.max()),
        'deprivation_median': float(np.median(deprivation_index)),
        'high_deprivation_pct': float((deprivation_index > 0.5).mean() * 100),
        'extreme_deprivation_pct': float((deprivation_index > 0.75).mean() * 100),
        'svf_deficit_mean': float(svf_deficit.mean()),
        'solar_deficit_mean': float(solar_deficit.mean())
    }
    
    stats_df = pd.DataFrame([stats])
    stats_path = output_dir / 'street_deprivation_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"  Saved statistics to {stats_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Load building footprints if available
    building_footprints = None
    area_data_dir = get_area_data_dir(args.area)
    
    # Try to find building footprints - look for building-specific files first
    building_file = None
    for pattern in ["*buildings*.gpkg", "*buildings*.shp", "*buildings*.geojson", 
                    "*footprints*.gpkg", "*footprints*.shp", "*footprints*.geojson"]:
        matches = list(area_data_dir.glob(pattern))
        if matches:
            building_file = matches[0]
            break
    
    # If not found in raw data, try processed metrics
    if building_file is None:
        metrics_dir = get_area_analysis_dir(args.area, 'metrics')
        metrics_file = metrics_dir / 'buildings_with_metrics.gpkg'
        if metrics_file.exists():
            building_file = metrics_file
    
    if building_file is not None and segments is not None:
        try:
            from src.svf_utils import load_building_footprints
            bounds = segments.total_bounds
            building_footprints = load_building_footprints(
                building_file,
                terrain_bounds=bounds,
                buffer_distance=0.0
            )
            # Ensure same CRS
            if building_footprints.crs != segments.crs:
                building_footprints = building_footprints.to_crs(segments.crs)
            logger.info(f"  Loaded building footprints for visualization")
        except Exception as e:
            logger.warning(f"  Could not load building footprints: {e}")
    
    # Create street deprivation map
    if segments is not None and isinstance(segments, gpd.GeoDataFrame):
        map_path = output_dir / "street_deprivation_map.png"
        create_street_deprivation_map(segments, deprivation_points, building_footprints, map_path)
    else:
        logger.warning("  Cannot create map visualization (segments not available as GeoDataFrame)")
    
    # Create distribution plots
    if segments is not None and isinstance(segments, gpd.GeoDataFrame):
        create_deprivation_distribution_plots(segments, deprivation_points, output_dir)
    else:
        logger.warning("  Cannot create distribution plots (segments not available as GeoDataFrame)")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("STREET-LEVEL DEPRIVATION INDEX SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total points: {stats['n_points']}")
    if segments is not None:
        logger.info(f"Total segments: {stats['n_segments']}")
    logger.info(f"\nDeprivation Index Statistics:")
    logger.info(f"  Mean: {stats['deprivation_mean']:.3f}")
    logger.info(f"  Median: {stats['deprivation_median']:.3f}")
    logger.info(f"  Min: {stats['deprivation_min']:.3f}")
    logger.info(f"  Max: {stats['deprivation_max']:.3f}")
    logger.info(f"  Std: {stats['deprivation_std']:.3f}")
    logger.info(f"\nDeprivation Thresholds:")
    logger.info(f"  High deprivation (>0.5): {stats['high_deprivation_pct']:.1f}%")
    logger.info(f"  Extreme deprivation (>0.75): {stats['extreme_deprivation_pct']:.1f}%")
    logger.info(f"\nComponent Deficits:")
    logger.info(f"  Mean SVF deficit: {stats['svf_deficit_mean']:.3f}")
    logger.info(f"  Mean solar deficit: {stats['solar_deficit_mean']:.3f}")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

