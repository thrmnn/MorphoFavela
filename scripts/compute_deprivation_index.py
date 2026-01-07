#!/usr/bin/env python3
"""
Morphological Environmental Deprivation Index

This script identifies environmental "hotspots" in an informal settlement by combining
solar access deficit, ventilation deficit, and occupancy pressure into a composite,
non-causal deprivation index.

The goal is to highlight zones of compounded environmental deprivation, NOT to predict
health outcomes. All metrics represent environmental performance proxies, and hotspots
indicate compounded deprivation. No causality is inferred.

Usage:
    python scripts/compute_deprivation_index.py \
        --units outputs/density/density_proxy.gpkg \
        --solar outputs/solar/solar_access.npy \
        --svf outputs/svf/svf.npy \
        --porosity outputs/porosity/porosity.npy \
        --density outputs/density/density_proxy.gpkg
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import logging
from shapely.geometry import Point, box
from tqdm import tqdm
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raster_data(raster_path: Path) -> tuple:
    """
    Load raster data (NumPy array) and infer spatial extent.
    
    Args:
        raster_path: Path to .npy file
        
    Returns:
        Tuple of (raster_array, extent) where extent is (minx, miny, maxx, maxy)
        Note: Extent is inferred from array shape and will need to be matched
        to actual coordinates from analysis units
    """
    logger.info(f"Loading raster from {raster_path}...")
    raster = np.load(raster_path)
    logger.info(f"  Loaded raster shape: {raster.shape}")
    return raster, None  # Extent will be inferred from analysis units


def aggregate_metrics(
    analysis_units: gpd.GeoDataFrame,
    solar_map: np.ndarray,
    svf_map: np.ndarray,
    porosity_map: np.ndarray,
    density_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate environmental metrics to analysis units.
    
    For each analysis unit, extracts:
    - Mean ground-level solar hours
    - Mean SVF
    - Mean sectional porosity
    - Occupancy density proxy (from density_gdf)
    
    Args:
        analysis_units: GeoDataFrame with analysis unit polygons
        solar_map: 2D NumPy array of solar access hours
        svf_map: 2D NumPy array of Sky View Factor values
        porosity_map: 2D NumPy array of sectional porosity values
        density_gdf: GeoDataFrame with density_proxy column
        
    Returns:
        GeoDataFrame with aggregated metrics per unit
    """
    logger.info("Aggregating environmental metrics to analysis units...")
    
    result = analysis_units.copy()
    
    # Get bounds of analysis units to create sampling grid
    bounds = analysis_units.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    
    # Infer grid resolution from raster dimensions
    # Assume rasters cover the same extent as analysis units
    n_y, n_x = solar_map.shape
    
    # Create coordinate arrays
    x_coords = np.linspace(minx, maxx, n_x)
    y_coords = np.linspace(miny, maxy, n_y)
    
    # Initialize aggregated metrics
    result['mean_solar_hours'] = 0.0
    result['mean_svf'] = 0.0
    result['mean_porosity'] = 0.0
    
    # Aggregate raster data per unit
    logger.info("  Aggregating raster metrics...")
    for idx, unit_row in tqdm(result.iterrows(), total=len(result), desc="  Aggregating"):
        unit_geom = unit_row.geometry
        
        # Find raster cells within this unit
        solar_values = []
        svf_values = []
        porosity_values = []
        
        # Sample raster at grid points
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                point = Point(x, y)
                if unit_geom.contains(point) or unit_geom.touches(point):
                    # Get raster values (note: y is first dimension in numpy arrays)
                    if not np.isnan(solar_map[i, j]):
                        solar_values.append(solar_map[i, j])
                    if not np.isnan(svf_map[i, j]):
                        svf_values.append(svf_map[i, j])
                    if not np.isnan(porosity_map[i, j]):
                        porosity_values.append(porosity_map[i, j])
        
        # Compute means
        if solar_values:
            result.loc[idx, 'mean_solar_hours'] = np.mean(solar_values)
        if svf_values:
            result.loc[idx, 'mean_svf'] = np.mean(svf_values)
        if porosity_values:
            result.loc[idx, 'mean_porosity'] = np.mean(porosity_values)
    
    # Join density proxy from density_gdf
    logger.info("  Joining occupancy density data...")
    if 'density_proxy' in density_gdf.columns:
        # Spatial join to match units
        if density_gdf.crs != result.crs:
            density_gdf = density_gdf.to_crs(result.crs)
        
        # Use spatial join to match units
        # Since both have the same geometry (analysis units), we can do a simpler join
        # Match by geometry or use index if they're the same units
        if len(result) == len(density_gdf):
            # Assume same units, match by index
            result['density_proxy'] = density_gdf['density_proxy'].values
        else:
            # Spatial join
            joined = gpd.sjoin(result, density_gdf, how='left', predicate='intersects')
            
            # Aggregate density_proxy per unit (take mean if multiple matches)
            # The left index is the original result index
            result_index = result.index
            density_by_unit = joined.groupby(result_index)['density_proxy'].mean()
            result['density_proxy'] = 0.0
            result.loc[density_by_unit.index, 'density_proxy'] = density_by_unit.values
    else:
        logger.warning("  'density_proxy' column not found in density_gdf")
        result['density_proxy'] = 0.0
    
    logger.info(f"  Aggregated metrics for {len(result)} analysis units")
    
    return result


def compute_deficits(
    analysis_units: gpd.GeoDataFrame,
    solar_reference: float = None
) -> gpd.GeoDataFrame:
    """
    Compute deficit scores normalized to [0, 1].
    
    1. Solar access deficit: 1 - (solar_hours / solar_reference)
    2. Ventilation deficit: 1 - ((svf + porosity) / 2)
    3. Occupancy pressure: percentile rank of density_proxy
    
    Args:
        analysis_units: GeoDataFrame with aggregated metrics
        solar_reference: Reference solar hours (if None, use median of data)
        
    Returns:
        GeoDataFrame with added deficit columns
    """
    logger.info("Computing deficit scores...")
    
    result = analysis_units.copy()
    
    # 1. Solar access deficit
    if solar_reference is None:
        # Use median as reference
        valid_solar = result['mean_solar_hours'][result['mean_solar_hours'] > 0]
        if len(valid_solar) > 0:
            solar_reference = valid_solar.median()
        else:
            solar_reference = 3.0  # Default reference (3 hours)
    
    logger.info(f"  Solar reference: {solar_reference:.2f} hours")
    
    # Compute solar deficit
    result['solar_deficit'] = 1.0 - (result['mean_solar_hours'] / solar_reference)
    result['solar_deficit'] = result['solar_deficit'].clip(0.0, 1.0)
    
    # 2. Ventilation deficit
    # Combine SVF and porosity: ventilation_score = (svf + porosity) / 2
    result['ventilation_score'] = (result['mean_svf'] + result['mean_porosity']) / 2.0
    result['ventilation_deficit'] = 1.0 - result['ventilation_score']
    result['ventilation_deficit'] = result['ventilation_deficit'].clip(0.0, 1.0)
    
    # 3. Occupancy pressure (percentile rank of density_proxy)
    valid_density = result['density_proxy'][result['density_proxy'] > 0]
    if len(valid_density) > 0:
        # Compute percentile rank (0-1 scale) using pandas
        result['occupancy_score'] = result['density_proxy'].rank(method='average', pct=True)
    else:
        result['occupancy_score'] = 0.0
    
    logger.info(f"  Mean solar deficit: {result['solar_deficit'].mean():.3f}")
    logger.info(f"  Mean ventilation deficit: {result['ventilation_deficit'].mean():.3f}")
    logger.info(f"  Mean occupancy score: {result['occupancy_score'].mean():.3f}")
    
    return result


def compute_hotspot_index(
    analysis_units: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Compute composite hotspot index using equal weighting.
    
    hotspot_index = (solar_deficit + ventilation_deficit + occupancy_score) / 3
    
    Equal weighting is used for transparency and simplicity. This does not imply
    that all deficits have equal impact, but rather provides a clear, interpretable
    composite measure.
    
    Args:
        analysis_units: GeoDataFrame with deficit columns
        
    Returns:
        GeoDataFrame with added 'hotspot_index' column
    """
    logger.info("Computing composite hotspot index...")
    
    result = analysis_units.copy()
    
    # Compute composite index with equal weighting
    result['hotspot_index'] = (
        result['solar_deficit'] + 
        result['ventilation_deficit'] + 
        result['occupancy_score']
    ) / 3.0
    
    # Clamp to [0, 1]
    result['hotspot_index'] = result['hotspot_index'].clip(0.0, 1.0)
    
    logger.info(f"  Mean hotspot index: {result['hotspot_index'].mean():.3f}")
    logger.info(f"  Min hotspot index: {result['hotspot_index'].min():.3f}")
    logger.info(f"  Max hotspot index: {result['hotspot_index'].max():.3f}")
    
    return result


def classify_hotspots(
    analysis_units: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Classify analysis units based on hotspot index distribution.
    
    Classification:
    - Top 10% → Extreme hotspot
    - 10-20% → High deprivation
    - Remaining → Baseline
    
    Args:
        analysis_units: GeoDataFrame with 'hotspot_index' column
        
    Returns:
        GeoDataFrame with added 'hotspot_class' column
    """
    logger.info("Classifying hotspots...")
    
    result = analysis_units.copy()
    
    # Compute percentiles
    hotspot_values = result['hotspot_index'].values
    p90 = np.percentile(hotspot_values, 90)
    p80 = np.percentile(hotspot_values, 80)
    
    logger.info(f"  90th percentile (extreme threshold): {p90:.3f}")
    logger.info(f"  80th percentile (high threshold): {p80:.3f}")
    
    # Classify
    result['hotspot_class'] = 'Baseline'
    result.loc[result['hotspot_index'] >= p90, 'hotspot_class'] = 'Extreme hotspot'
    result.loc[(result['hotspot_index'] >= p80) & (result['hotspot_index'] < p90), 'hotspot_class'] = 'High deprivation'
    
    # Count units per class
    class_counts = result['hotspot_class'].value_counts()
    logger.info(f"  Classification counts:")
    for class_name, count in class_counts.items():
        logger.info(f"    {class_name}: {count} units ({count/len(result)*100:.1f}%)")
    
    return result


def compute_deficit_overlap(
    analysis_units: gpd.GeoDataFrame,
    solar_threshold: float = 0.5,
    ventilation_threshold: float = 0.5,
    occupancy_threshold: float = 0.8
) -> gpd.GeoDataFrame:
    """
    Count how many deficits exceed critical thresholds for each unit.
    
    Args:
        analysis_units: GeoDataFrame with deficit columns
        solar_threshold: Threshold for low solar (default: 0.5 = 50% deficit)
        ventilation_threshold: Threshold for low ventilation (default: 0.5 = 50% deficit)
        occupancy_threshold: Threshold for high occupancy (default: 0.8 = 80th percentile)
        
    Returns:
        GeoDataFrame with added 'deficit_overlap_count' column (0-3)
    """
    logger.info("Computing deficit overlap...")
    
    result = analysis_units.copy()
    
    # Count deficits exceeding thresholds
    low_solar = result['solar_deficit'] >= solar_threshold
    low_ventilation = result['ventilation_deficit'] >= ventilation_threshold
    high_occupancy = result['occupancy_score'] >= occupancy_threshold
    
    result['deficit_overlap_count'] = (
        low_solar.astype(int) + 
        low_ventilation.astype(int) + 
        high_occupancy.astype(int)
    )
    
    logger.info(f"  Units with 0 deficits: {(result['deficit_overlap_count'] == 0).sum()}")
    logger.info(f"  Units with 1 deficit: {(result['deficit_overlap_count'] == 1).sum()}")
    logger.info(f"  Units with 2 deficits: {(result['deficit_overlap_count'] == 2).sum()}")
    logger.info(f"  Units with 3 deficits: {(result['deficit_overlap_count'] == 3).sum()}")
    
    return result


def plot_hotspot_map(
    analysis_units: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """
    Create choropleth map of hotspot classification.
    
    Color scheme:
    - Dark red: Extreme hotspot
    - Orange: High deprivation
    - Grey: Baseline
    
    Args:
        analysis_units: GeoDataFrame with 'hotspot_class' column
        output_path: Path to save the visualization
    """
    logger.info("Creating hotspot map...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define color mapping
    color_map = {
        'Extreme hotspot': '#8B0000',  # Dark red
        'High deprivation': '#FF8C00',  # Orange
        'Baseline': '#D3D3D3'  # Light grey
    }
    
    # Plot each class separately to control colors
    handles = []
    for class_name, color in color_map.items():
        class_units = analysis_units[analysis_units['hotspot_class'] == class_name]
        if len(class_units) > 0:
            patch = class_units.plot(
                ax=ax,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                label=class_name
            )
            # Get the first patch for legend
            if hasattr(patch, 'collections') and len(patch.collections) > 0:
                handles.append(plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', label=class_name))
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Environmental Deprivation Hotspot Map\n(Composite Index: Solar + Ventilation + Occupancy)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    if handles:
        ax.legend(handles=handles, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved hotspot map to {output_path}")


def plot_deficit_overlap_map(
    analysis_units: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """
    Create map showing deficit overlap count (0-3).
    
    Args:
        analysis_units: GeoDataFrame with 'deficit_overlap_count' column
        output_path: Path to save the visualization
    """
    logger.info("Creating deficit overlap map...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot with color scale for overlap count
    analysis_units.plot(
        column='deficit_overlap_count',
        ax=ax,
        cmap='Reds',  # Red scale (more deficits = darker)
        legend=True,
        edgecolor='black',
        linewidth=0.5,
        vmin=0,
        vmax=3,
        legend_kwds={
            'label': 'Number of Deficits Exceeding Thresholds',
            'shrink': 0.8,
            'ticks': [0, 1, 2, 3]
        }
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Deficit Overlap Map\n(Count of Exceeded Thresholds: Solar, Ventilation, Occupancy)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved deficit overlap map to {output_path}")


def export_ranking_table(
    analysis_units: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """
    Export ranking table with percentiles and composite index.
    
    Args:
        analysis_units: GeoDataFrame with all metrics
        output_path: Path to save CSV
    """
    logger.info("Exporting ranking table...")
    
    # Compute percentiles for each metric
    result = analysis_units.copy()
    
    # Percentile ranks (0-100 scale) using pandas
    result['solar_percentile'] = result['solar_deficit'].rank(method='average', pct=True) * 100
    result['ventilation_percentile'] = result['ventilation_deficit'].rank(method='average', pct=True) * 100
    result['occupancy_percentile'] = result['occupancy_score'].rank(method='average', pct=True) * 100
    
    # Rank by hotspot index (1 = worst, descending order)
    result['rank'] = result['hotspot_index'].rank(method='min', ascending=False)
    
    # Select columns for export
    export_columns = [
        'rank',
        'hotspot_index',
        'hotspot_class',
        'solar_percentile',
        'ventilation_percentile',
        'occupancy_percentile',
        'solar_deficit',
        'ventilation_deficit',
        'occupancy_score',
        'deficit_overlap_count',
        'mean_solar_hours',
        'mean_svf',
        'mean_porosity',
        'density_proxy'
    ]
    
    # Create DataFrame with selected columns
    export_df = result[export_columns].copy()
    export_df = export_df.sort_values('rank')
    
    # Save to CSV
    export_df.to_csv(output_path, index=False)
    logger.info(f"  Saved ranking table to {output_path}")
    logger.info(f"  Exported {len(export_df)} units")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute Morphological Environmental Deprivation Index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script identifies environmental hotspots by combining multiple deficits.
All metrics are relative and distribution-based. No causality is inferred.
        """
    )
    parser.add_argument('--units', type=str, required=True, help='Path to analysis units shapefile/GPKG')
    parser.add_argument('--solar', type=str, required=True, help='Path to solar access raster (.npy)')
    parser.add_argument('--svf', type=str, required=True, help='Path to SVF raster (.npy)')
    parser.add_argument('--porosity', type=str, required=True, help='Path to porosity raster (.npy)')
    parser.add_argument('--density', type=str, required=True, help='Path to density proxy GeoDataFrame (.gpkg or .shp)')
    parser.add_argument('--solar-reference', type=float, default=None, help='Reference solar hours (default: median of data)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/deprivation)')
    
    args = parser.parse_args()
    
    # Setup paths
    units_path = Path(args.units)
    solar_path = Path(args.solar)
    svf_path = Path(args.svf)
    porosity_path = Path(args.porosity)
    density_path = Path(args.density)
    
    # Validate inputs
    for path, name in [(units_path, 'units'), (solar_path, 'solar'), 
                       (svf_path, 'svf'), (porosity_path, 'porosity'), 
                       (density_path, 'density')]:
        if not path.exists():
            logger.error(f"{name} file not found: {path}")
            sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "deprivation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MORPHOLOGICAL ENVIRONMENTAL DEPRIVATION INDEX")
    print("=" * 60)
    print(f"Analysis units: {units_path}")
    print(f"Solar access: {solar_path}")
    print(f"SVF: {svf_path}")
    print(f"Porosity: {porosity_path}")
    print(f"Density proxy: {density_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load data
    logger.info("Loading data...")
    analysis_units = gpd.read_file(str(units_path))
    logger.info(f"  Loaded {len(analysis_units)} analysis units")
    
    solar_map, _ = load_raster_data(solar_path)
    svf_map, _ = load_raster_data(svf_path)
    porosity_map, _ = load_raster_data(porosity_path)
    density_gdf = gpd.read_file(str(density_path))
    logger.info(f"  Loaded density data with {len(density_gdf)} units")
    
    # Aggregate metrics
    units_with_metrics = aggregate_metrics(
        analysis_units, solar_map, svf_map, porosity_map, density_gdf
    )
    
    # Compute deficits
    units_with_deficits = compute_deficits(
        units_with_metrics, solar_reference=args.solar_reference
    )
    
    # Compute hotspot index
    units_with_index = compute_hotspot_index(units_with_deficits)
    
    # Classify hotspots
    units_classified = classify_hotspots(units_with_index)
    
    # Compute deficit overlap
    result = compute_deficit_overlap(units_classified)
    
    # Save results
    output_gpkg = output_dir / "deprivation_index.gpkg"
    result.to_file(output_gpkg, driver="GPKG")
    logger.info(f"  Saved results to {output_gpkg}")
    
    # Create visualizations
    hotspot_map_path = output_dir / "hotspot_map.png"
    plot_hotspot_map(result, hotspot_map_path)
    
    overlap_map_path = output_dir / "deficit_overlap_map.png"
    plot_deficit_overlap_map(result, overlap_map_path)
    
    # Export ranking table
    ranking_path = output_dir / "ranking_table.csv"
    export_ranking_table(result, ranking_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

