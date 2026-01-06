#!/usr/bin/env python3
"""
Occupancy Density Proxy Computation

This script computes an occupancy density proxy defined as the ratio of built volume
to open space area, aggregated at the block or cluster level.

Density proxy: D = V_built / A_open

where:
    V_built = total building volume
    A_open = total open (non-built) ground area

This metric is used as a relative indicator of crowding and environmental pressure,
NOT as a direct population estimate. It is used only for relative ranking.

Usage:
    python scripts/compute_occupancy_density.py \
        --stl data/raw/full_scan.stl \
        --footprints data/raw/vidigal_buildings.shp \
        --units data/raw/analysis_units.shp
"""

import numpy as np
import pyvista as pv
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import logging
from shapely.geometry import Polygon, Point, box
from tqdm import tqdm
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from src.svf_utils import (
    load_mesh,
    load_building_footprints,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_building_volumes(
    full_mesh: pv.PolyData,
    footprints: gpd.GeoDataFrame,
    terrain_bounds: tuple,
    z_threshold: float = None
) -> gpd.GeoDataFrame:
    """
    Compute built volume for each building by extracting meshes from STL.
    
    Args:
        full_mesh: Complete STL mesh containing terrain and buildings
        footprints: GeoDataFrame with building footprint polygons
        terrain_bounds: Bounding box of terrain (minx, miny, maxx, maxy, minz, maxz)
        z_threshold: Optional Z threshold to separate terrain from buildings
        
    Returns:
        GeoDataFrame with added 'volume' column
    """
    logger.info("Computing building volumes...")
    
    # Estimate terrain level (minimum Z of mesh)
    if z_threshold is None:
        z_threshold = full_mesh.bounds[4]  # Minimum Z
    
    logger.info(f"  Using terrain level estimate: {z_threshold:.2f}m")
    
    footprints = footprints.copy()
    footprints['volume'] = 0.0
    
    # Extract building meshes and compute volumes
    volumes = []
    
    # Get all mesh points once
    mesh_points = full_mesh.points
    
    for idx, row in tqdm(footprints.iterrows(), total=len(footprints), desc="  Computing volumes"):
        geom = row.geometry
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        
        # Create 3D bounding box for building extraction
        min_z = z_threshold - 1.0  # Slightly below terrain
        max_z = z_threshold + 100.0  # Max possible building height
        
        # Filter points within 3D bounding box and 2D footprint
        points_in_footprint = []
        for p in mesh_points:
            x, y, z = p
            # Check if point is within 2D footprint and Z range
            if (bounds[0] <= x <= bounds[2] and 
                bounds[1] <= y <= bounds[3] and
                min_z <= z <= max_z and
                geom.contains(Point(x, y))):
                points_in_footprint.append(p)
        
        if len(points_in_footprint) < 4:
            volumes.append(0.0)
            continue
        
        # Create mesh from points
        building_mesh = pv.PolyData(np.array(points_in_footprint))
        
        # Compute volume using convex hull (fast approximation)
        try:
            hull = building_mesh.convex_hull()
            if hull.n_cells > 0:
                volume = hull.volume
            else:
                # Fallback: bounding box volume
                bounds_3d = building_mesh.bounds
                volume = (bounds_3d[1] - bounds_3d[0]) * \
                         (bounds_3d[3] - bounds_3d[2]) * \
                         (bounds_3d[5] - bounds_3d[4])
        except:
            # Fallback: bounding box volume
            bounds_3d = building_mesh.bounds
            volume = (bounds_3d[1] - bounds_3d[0]) * \
                     (bounds_3d[3] - bounds_3d[2]) * \
                     (bounds_3d[5] - bounds_3d[4])
        
        volumes.append(volume)
    
    footprints['volume'] = volumes
    
    total_volume = sum(volumes)
    logger.info(f"  Computed volumes for {len(footprints)} buildings")
    logger.info(f"  Total built volume: {total_volume:.2f} m³")
    logger.info(f"  Mean volume per building: {np.mean(volumes):.2f} m³")
    
    return footprints


def generate_analysis_units(
    footprints: gpd.GeoDataFrame,
    grid_size: float = 50.0
) -> gpd.GeoDataFrame:
    """
    Generate analysis units as a regular grid covering the footprint area.
    
    Args:
        footprints: GeoDataFrame with building footprints
        grid_size: Size of grid cells in meters (default: 50m)
        
    Returns:
        GeoDataFrame with grid cell polygons as analysis units
    """
    logger.info(f"Generating analysis units as {grid_size}m × {grid_size}m grid...")
    
    bounds = footprints.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    
    # Generate grid cells
    grid_cells = []
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    
    for y in y_coords:
        for x in x_coords:
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)
    
    # Create GeoDataFrame
    units = gpd.GeoDataFrame(geometry=grid_cells, crs=footprints.crs)
    units['unit_id'] = range(len(units))
    
    logger.info(f"  Generated {len(units)} analysis units")
    
    return units


def aggregate_by_unit(
    footprints: gpd.GeoDataFrame,
    analysis_units: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate built volume per analysis unit using spatial join.
    
    Args:
        footprints: GeoDataFrame with building footprints and 'volume' column
        analysis_units: GeoDataFrame with analysis unit polygons (blocks/clusters)
        
    Returns:
        GeoDataFrame with aggregated volumes per unit
    """
    logger.info("Aggregating volumes by analysis unit...")
    
    # Ensure CRS match
    if footprints.crs != analysis_units.crs:
        logger.info(f"  Reprojecting footprints from {footprints.crs} to {analysis_units.crs}")
        footprints = footprints.to_crs(analysis_units.crs)
    
    # Spatial join: assign each building to an analysis unit
    logger.info("  Performing spatial join...")
    joined = gpd.sjoin(footprints, analysis_units, how='inner', predicate='within')
    
    # Aggregate volumes per unit
    logger.info("  Aggregating volumes...")
    aggregated = joined.groupby(joined.index_right).agg({
        'volume': 'sum'
    }).rename(columns={'volume': 'built_volume'})
    
    # Merge back to analysis units
    result = analysis_units.copy()
    result['built_volume'] = 0.0
    result.loc[aggregated.index, 'built_volume'] = aggregated['built_volume']
    
    logger.info(f"  Aggregated volumes for {len(result)} analysis units")
    logger.info(f"  Units with buildings: {(result['built_volume'] > 0).sum()}")
    logger.info(f"  Total aggregated volume: {result['built_volume'].sum():.2f} m³")
    
    return result


def compute_open_space(
    analysis_units: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Compute open space area for each analysis unit.
    
    Open space = total unit area - built footprint area
    
    Args:
        analysis_units: GeoDataFrame with analysis unit polygons
        footprints: GeoDataFrame with building footprints
        
    Returns:
        GeoDataFrame with added 'open_space_area' column
    """
    logger.info("Computing open space area...")
    
    # Ensure CRS match
    if footprints.crs != analysis_units.crs:
        footprints = footprints.to_crs(analysis_units.crs)
    
    result = analysis_units.copy()
    result['total_area'] = result.geometry.area
    result['built_area'] = 0.0
    
    # For each analysis unit, compute built footprint area
    for idx, unit_row in tqdm(result.iterrows(), total=len(result), desc="  Computing open space"):
        unit_geom = unit_row.geometry
        
        # Find buildings within this unit
        buildings_in_unit = footprints[footprints.geometry.within(unit_geom)]
        
        if len(buildings_in_unit) == 0:
            built_area = 0.0
        else:
            # Compute union of building footprints within unit
            # Clip to unit boundary to avoid overcounting
            built_geoms = []
            for _, building_row in buildings_in_unit.iterrows():
                building_geom = building_row.geometry
                # Clip building to unit boundary
                clipped = building_geom.intersection(unit_geom)
                if not clipped.is_empty:
                    built_geoms.append(clipped)
            
            if built_geoms:
                # Union all building footprints
                from shapely.ops import unary_union
                built_union = unary_union(built_geoms)
                built_area = built_union.area
            else:
                built_area = 0.0
        
        result.loc[idx, 'built_area'] = built_area
    
    # Compute open space area
    result['open_space_area'] = result['total_area'] - result['built_area']
    
    # Handle negative or near-zero values
    result['open_space_area'] = result['open_space_area'].clip(lower=0.0)
    
    logger.info(f"  Computed open space for {len(result)} analysis units")
    logger.info(f"  Mean open space: {result['open_space_area'].mean():.2f} m²")
    logger.info(f"  Units with zero open space: {(result['open_space_area'] == 0).sum()}")
    
    return result


def compute_density_proxy(
    analysis_units: gpd.GeoDataFrame,
    min_open_space: float = 1.0
) -> gpd.GeoDataFrame:
    """
    Compute density proxy: built_volume / open_space_area
    
    Args:
        analysis_units: GeoDataFrame with 'built_volume' and 'open_space_area' columns
        min_open_space: Minimum open space threshold to avoid division by zero
        
    Returns:
        GeoDataFrame with added 'density_proxy' column
    """
    logger.info("Computing density proxy...")
    
    result = analysis_units.copy()
    
    # Compute density proxy
    # Handle zero or near-zero open space safely
    open_space_safe = result['open_space_area'].clip(lower=min_open_space)
    result['density_proxy'] = result['built_volume'] / open_space_safe
    
    # Mark units with insufficient open space
    result['has_sufficient_open_space'] = result['open_space_area'] >= min_open_space
    
    # Set density to NaN or high value for units with zero open space
    zero_open_mask = result['open_space_area'] < min_open_space
    if zero_open_mask.any():
        logger.warning(f"  {zero_open_mask.sum()} units have insufficient open space (<{min_open_space}m²)")
        # Set to a high value to indicate extreme density
        result.loc[zero_open_mask, 'density_proxy'] = np.inf
    
    logger.info(f"  Computed density proxy for {len(result)} units")
    logger.info(f"  Mean density proxy: {result[result['density_proxy'] != np.inf]['density_proxy'].mean():.2f} m³/m²")
    logger.info(f"  Max density proxy: {result[result['density_proxy'] != np.inf]['density_proxy'].max():.2f} m³/m²")
    
    return result


def plot_density_map(
    analysis_units: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """
    Create a choropleth map of density proxy using quantile-based color scale.
    
    Args:
        analysis_units: GeoDataFrame with 'density_proxy' column
        output_path: Path to save the visualization
    """
    logger.info("Creating density map visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filter out infinite values for color scale
    finite_mask = np.isfinite(analysis_units['density_proxy'])
    finite_units = analysis_units[finite_mask]
    
    if len(finite_units) == 0:
        logger.warning("  No finite density values to plot")
        analysis_units.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)
    else:
        # Use quantile-based color scale (5 quantiles)
        n_quantiles = 5
        quantiles = np.linspace(0, 100, n_quantiles + 1)
        vmin = np.percentile(finite_units['density_proxy'], quantiles[0])
        vmax = np.percentile(finite_units['density_proxy'], quantiles[-1])
        
        # Plot finite values
        finite_units.plot(
            column='density_proxy',
            ax=ax,
            cmap='YlOrRd',  # Yellow-Orange-Red (higher density = redder)
            legend=True,
            edgecolor='black',
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                'label': 'Density Proxy (m³/m²)',
                'shrink': 0.8
            }
        )
        
        # Plot infinite values (zero open space) in a distinct color
        infinite_units = analysis_units[~finite_mask]
        if len(infinite_units) > 0:
            infinite_units.plot(
                ax=ax,
                color='darkred',
                edgecolor='black',
                linewidth=1.0,
                label='Insufficient open space'
            )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Occupancy Density Proxy Map\n(Built Volume / Open Space Ratio)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved density map to {output_path}")


def print_summary_statistics(analysis_units: gpd.GeoDataFrame) -> None:
    """
    Print summary statistics for density proxy.
    
    Args:
        analysis_units: GeoDataFrame with 'density_proxy' column
    """
    finite_mask = np.isfinite(analysis_units['density_proxy'])
    finite_values = analysis_units[finite_mask]['density_proxy']
    
    print("\n" + "=" * 60)
    print("OCCUPANCY DENSITY PROXY SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total analysis units: {len(analysis_units)}")
    print(f"Units with finite density: {len(finite_values)}")
    print(f"Units with insufficient open space: {(~finite_mask).sum()}")
    
    if len(finite_values) > 0:
        print(f"\nDensity proxy (m³/m²):")
        print(f"  Mean: {finite_values.mean():.2f}")
        print(f"  Median: {finite_values.median():.2f}")
        print(f"  Std: {finite_values.std():.2f}")
        print(f"  Min: {finite_values.min():.2f}")
        print(f"  Max: {finite_values.max():.2f}")
        print(f"  80th percentile: {np.percentile(finite_values, 80):.2f}")
        print(f"  90th percentile: {np.percentile(finite_values, 90):.2f}")
    
    print(f"\nVolume statistics:")
    print(f"  Total built volume: {analysis_units['built_volume'].sum():.2f} m³")
    print(f"  Mean volume per unit: {analysis_units['built_volume'].mean():.2f} m³")
    
    print(f"\nOpen space statistics:")
    print(f"  Total open space: {analysis_units['open_space_area'].sum():.2f} m²")
    print(f"  Mean open space per unit: {analysis_units['open_space_area'].mean():.2f} m²")
    
    print("=" * 60 + "\n")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute occupancy density proxy (built volume / open space ratio)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This metric is a proxy for occupancy pressure and environmental crowding.
It does NOT represent population. It is used only for relative ranking.
        """
    )
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file (terrain + buildings)')
    parser.add_argument('--footprints', type=str, required=True, help='Path to building footprints shapefile')
    parser.add_argument('--units', type=str, default=None, help='Optional path to analysis units shapefile (blocks/clusters). If not provided, a regular grid will be generated.')
    parser.add_argument('--grid-size', type=float, default=50.0, help='Grid cell size in meters for auto-generated units (default: 50.0)')
    parser.add_argument('--z-threshold', type=float, default=None, help='Z threshold to separate terrain from buildings')
    parser.add_argument('--min-open-space', type=float, default=1.0, help='Minimum open space threshold (m²) to avoid division by zero')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/density)')
    
    args = parser.parse_args()
    
    # Setup paths
    stl_path = Path(args.stl)
    footprints_path = Path(args.footprints)
    
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        sys.exit(1)
    if not footprints_path.exists():
        logger.error(f"Footprints file not found: {footprints_path}")
        sys.exit(1)
    
    # Check if units file is provided
    if args.units:
        units_path = Path(args.units)
        if not units_path.exists():
            logger.error(f"Analysis units file not found: {units_path}")
            sys.exit(1)
        use_custom_units = True
    else:
        use_custom_units = False
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "density"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("OCCUPANCY DENSITY PROXY COMPUTATION")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    print(f"Building footprints: {footprints_path}")
    if use_custom_units:
        print(f"Analysis units: {units_path}")
    else:
        print(f"Analysis units: Auto-generated grid ({args.grid_size}m × {args.grid_size}m)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load data
    logger.info("Loading data...")
    full_mesh = load_mesh(stl_path)
    
    footprints = load_building_footprints(
        footprints_path,
        terrain_bounds=full_mesh.bounds,
        buffer_distance=0.0  # No buffer for volume computation
    )
    
    # Load or generate analysis units
    if use_custom_units:
        analysis_units = gpd.read_file(str(units_path))
        logger.info(f"  Loaded {len(analysis_units)} analysis units from file")
        logger.info(f"  Units CRS: {analysis_units.crs}")
    else:
        analysis_units = generate_analysis_units(footprints, grid_size=args.grid_size)
        logger.info(f"  Generated {len(analysis_units)} analysis units as regular grid")
    
    # Compute building volumes
    footprints_with_volume = compute_building_volumes(
        full_mesh,
        footprints,
        full_mesh.bounds,
        z_threshold=args.z_threshold
    )
    
    # Aggregate volumes by analysis unit
    units_with_volume = aggregate_by_unit(footprints_with_volume, analysis_units)
    
    # Compute open space area
    units_with_open_space = compute_open_space(units_with_volume, footprints_with_volume)
    
    # Compute density proxy
    result = compute_density_proxy(units_with_open_space, min_open_space=args.min_open_space)
    
    # Print summary statistics
    print_summary_statistics(result)
    
    # Save results
    output_gpkg = output_dir / "density_proxy.gpkg"
    result.to_file(output_gpkg, driver="GPKG")
    logger.info(f"  Saved results to {output_gpkg}")
    
    # Save CSV summary
    output_csv = output_dir / "density_proxy.csv"
    csv_columns = ['built_volume', 'open_space_area', 'density_proxy', 'total_area', 'built_area']
    result[csv_columns].to_csv(output_csv, index=False)
    logger.info(f"  Saved CSV summary to {output_csv}")
    
    # Create visualization
    density_map_path = output_dir / "density_map.png"
    plot_density_map(result, density_map_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

