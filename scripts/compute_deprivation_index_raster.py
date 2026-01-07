#!/usr/bin/env python3
"""
Morphological Environmental Deprivation Index - Raster Version

This script computes a continuous 2D raster of environmental deprivation by combining
solar access deficit, ventilation deficit, and occupancy pressure at pixel level.

The raster approach provides higher spatial resolution and shows continuous gradients,
while also providing unit-level aggregation for policy interpretation.

Usage:
    python scripts/compute_deprivation_index_raster.py \
        --solar outputs/solar/solar_access.npy \
        --svf outputs/svf/svf.npy \
        --porosity outputs/porosity/porosity.npy \
        --stl data/raw/full_scan.stl \
        --footprints data/raw/vidigal_buildings.shp \
        --units outputs/density/density_proxy.gpkg
"""

import numpy as np
import pyvista as pv
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

# Import shared utilities
from src.svf_utils import (
    load_mesh,
    load_building_footprints,
    compute_ground_mask,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raster_with_coords(raster_path: Path, bounds: tuple) -> tuple:
    """
    Load raster and create coordinate arrays.
    
    Args:
        raster_path: Path to .npy file
        bounds: Bounding box (minx, miny, maxx, maxy)
        
    Returns:
        Tuple of (raster_array, x_coords, y_coords)
    """
    logger.info(f"Loading raster from {raster_path}...")
    raster = np.load(raster_path)
    logger.info(f"  Loaded raster shape: {raster.shape}")
    
    minx, miny, maxx, maxy = bounds
    n_y, n_x = raster.shape
    
    # Create coordinate arrays
    x_coords = np.linspace(minx, maxx, n_x)
    y_coords = np.linspace(miny, maxy, n_y)
    
    return raster, x_coords, y_coords


def compute_pixel_level_occupancy(
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    footprints: gpd.GeoDataFrame,
    full_mesh: pv.PolyData,
    z_threshold: float = None,
    grid_spacing: float = None
) -> np.ndarray:
    """
    Compute occupancy pressure (built volume / open space) at each grid cell.
    
    For each grid cell:
    1. Compute built volume within cell (from building footprints)
    2. Compute open space area (cell area - built footprint area)
    3. Occupancy pressure = built_volume / open_space_area
    
    Args:
        grid_x_coords: 1D array of X coordinates
        grid_y_coords: 1D array of Y coordinates
        footprints: GeoDataFrame with building footprints
        full_mesh: STL mesh for volume extraction
        z_threshold: Z threshold for terrain separation
        grid_spacing: Grid cell spacing (inferred from coords if None)
        
    Returns:
        2D NumPy array of occupancy pressure values
    """
    logger.info("Computing pixel-level occupancy pressure...")
    
    if z_threshold is None:
        z_threshold = full_mesh.bounds[4]  # Minimum Z
    
    if grid_spacing is None:
        # Infer from coordinate arrays
        if len(grid_x_coords) > 1:
            grid_spacing = grid_x_coords[1] - grid_x_coords[0]
        else:
            grid_spacing = 2.0  # Default
    
    cell_area = grid_spacing * grid_spacing
    n_y, n_x = len(grid_y_coords), len(grid_x_coords)
    
    occupancy_pressure = np.full((n_y, n_x), np.nan)
    
    # Get mesh points for volume computation
    mesh_points = full_mesh.points
    
    # Compute building volumes (reuse from occupancy density script logic)
    logger.info("  Computing building volumes...")
    footprints_with_volume = footprints.copy()
    footprints_with_volume['volume'] = 0.0
    
    volumes = []
    for idx, row in tqdm(footprints_with_volume.iterrows(), total=len(footprints_with_volume), desc="  Computing volumes"):
        geom = row.geometry
        bounds_2d = geom.bounds
        
        min_z = z_threshold - 1.0
        max_z = z_threshold + 100.0
        
        points_in_footprint = []
        for p in mesh_points:
            x, y, z = p
            if (bounds_2d[0] <= x <= bounds_2d[2] and 
                bounds_2d[1] <= y <= bounds_2d[3] and
                min_z <= z <= max_z and
                geom.contains(Point(x, y))):
                points_in_footprint.append(p)
        
        if len(points_in_footprint) < 4:
            volumes.append(0.0)
            continue
        
        building_mesh = pv.PolyData(np.array(points_in_footprint))
        try:
            hull = building_mesh.convex_hull()
            if hull.n_cells > 0:
                volume = hull.volume
            else:
                bounds_3d = building_mesh.bounds
                volume = (bounds_3d[1] - bounds_3d[0]) * (bounds_3d[3] - bounds_3d[2]) * (bounds_3d[5] - bounds_3d[4])
        except:
            bounds_3d = building_mesh.bounds
            volume = (bounds_3d[1] - bounds_3d[0]) * (bounds_3d[3] - bounds_3d[2]) * (bounds_3d[5] - bounds_3d[4])
        
        volumes.append(volume)
    
    footprints_with_volume['volume'] = volumes
    
    logger.info("  Computing occupancy pressure per grid cell...")
    for i, y in enumerate(tqdm(grid_y_coords, desc="  Processing rows")):
        for j, x in enumerate(grid_x_coords):
            # Create grid cell
            cell = box(x, y, x + grid_spacing, y + grid_spacing)
            
            # Find buildings intersecting this cell
            buildings_in_cell = footprints_with_volume[footprints_with_volume.geometry.intersects(cell)]
            
            if len(buildings_in_cell) == 0:
                # No buildings - fully open
                occupancy_pressure[i, j] = 0.0
                continue
            
            # Compute built volume in cell
            built_volume = 0.0
            built_area = 0.0
            
            for _, building_row in buildings_in_cell.iterrows():
                building_geom = building_row.geometry
                intersection = cell.intersection(building_geom)
                
                if not intersection.is_empty:
                    # Proportion of building in cell
                    building_area = building_geom.area
                    if building_area > 0:
                        intersection_area = intersection.area
                        volume_proportion = intersection_area / building_area
                        built_volume += building_row['volume'] * volume_proportion
                        built_area += intersection_area
            
            # Compute open space
            open_space = cell_area - built_area
            
            # Compute occupancy pressure
            if open_space > 1.0:  # Minimum threshold to avoid division by zero
                occupancy_pressure[i, j] = built_volume / open_space
            else:
                # Very little open space - set to high value
                occupancy_pressure[i, j] = np.inf if built_volume > 0 else 0.0
    
    logger.info(f"  Computed occupancy pressure for {n_y}×{n_x} grid")
    logger.info(f"  Mean occupancy pressure: {np.nanmean(occupancy_pressure[~np.isinf(occupancy_pressure)]):.2f} m³/m²")
    
    return occupancy_pressure


def compute_raster_deficits(
    solar_map: np.ndarray,
    svf_map: np.ndarray,
    porosity_map: np.ndarray,
    occupancy_map: np.ndarray,
    solar_reference: float = None
) -> tuple:
    """
    Compute deficit scores at pixel level.
    
    Args:
        solar_map: 2D array of solar access hours
        svf_map: 2D array of SVF values
        porosity_map: 2D array of porosity values
        occupancy_map: 2D array of occupancy pressure
        solar_reference: Reference solar hours (if None, use median)
        
    Returns:
        Tuple of (solar_deficit, ventilation_deficit, occupancy_score) arrays
    """
    logger.info("Computing pixel-level deficits...")
    
    # 1. Solar access deficit
    if solar_reference is None:
        valid_solar = solar_map[~np.isnan(solar_map) & (solar_map > 0)]
        if len(valid_solar) > 0:
            solar_reference = np.median(valid_solar)
        else:
            solar_reference = 3.0
    
    logger.info(f"  Solar reference: {solar_reference:.2f} hours")
    
    solar_deficit = 1.0 - (solar_map / solar_reference)
    solar_deficit = np.clip(solar_deficit, 0.0, 1.0)
    solar_deficit[np.isnan(solar_map)] = np.nan
    
    # 2. Ventilation deficit
    ventilation_score = (svf_map + porosity_map) / 2.0
    ventilation_deficit = 1.0 - ventilation_score
    ventilation_deficit = np.clip(ventilation_deficit, 0.0, 1.0)
    ventilation_deficit[np.isnan(svf_map) | np.isnan(porosity_map)] = np.nan
    
    # 3. Occupancy pressure (percentile rank)
    valid_occupancy = occupancy_map[~np.isnan(occupancy_map) & ~np.isinf(occupancy_map) & (occupancy_map >= 0)]
    if len(valid_occupancy) > 0:
        # Compute percentile rank for each pixel
        occupancy_flat = occupancy_map.flatten()
        valid_mask = ~np.isnan(occupancy_flat) & ~np.isinf(occupancy_flat) & (occupancy_flat >= 0)
        
        if valid_mask.sum() > 0:
            occupancy_score = np.full_like(occupancy_flat, np.nan)
            occupancy_score[valid_mask] = pd.Series(occupancy_flat[valid_mask]).rank(method='average', pct=True).values
            occupancy_score = occupancy_score.reshape(occupancy_map.shape)
        else:
            occupancy_score = np.full_like(occupancy_map, 0.0)
    else:
        occupancy_score = np.full_like(occupancy_map, 0.0)
    
    # Handle infinite values in occupancy
    occupancy_score[np.isinf(occupancy_map)] = 1.0  # Maximum pressure
    
    logger.info(f"  Mean solar deficit: {np.nanmean(solar_deficit):.3f}")
    logger.info(f"  Mean ventilation deficit: {np.nanmean(ventilation_deficit):.3f}")
    logger.info(f"  Mean occupancy score: {np.nanmean(occupancy_score):.3f}")
    
    return solar_deficit, ventilation_deficit, occupancy_score


def compute_hotspot_index_raster(
    solar_deficit: np.ndarray,
    ventilation_deficit: np.ndarray,
    occupancy_score: np.ndarray
) -> np.ndarray:
    """
    Compute composite hotspot index at pixel level.
    
    Args:
        solar_deficit: 2D array of solar deficits
        ventilation_deficit: 2D array of ventilation deficits
        occupancy_score: 2D array of occupancy scores
        
    Returns:
        2D array of hotspot indices
    """
    logger.info("Computing composite hotspot index...")
    
    hotspot_index = (solar_deficit + ventilation_deficit + occupancy_score) / 3.0
    hotspot_index = np.clip(hotspot_index, 0.0, 1.0)
    
    # Set NaN where any input is NaN
    mask = np.isnan(solar_deficit) | np.isnan(ventilation_deficit) | np.isnan(occupancy_score)
    hotspot_index[mask] = np.nan
    
    logger.info(f"  Mean hotspot index: {np.nanmean(hotspot_index):.3f}")
    logger.info(f"  Min hotspot index: {np.nanmin(hotspot_index):.3f}")
    logger.info(f"  Max hotspot index: {np.nanmax(hotspot_index):.3f}")
    
    return hotspot_index


def apply_building_mask(
    hotspot_index: np.ndarray,
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    footprints: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Apply building mask to set building interiors to NaN.
    
    Args:
        hotspot_index: 2D array of hotspot indices
        grid_x_coords: 1D array of X coordinates
        grid_y_coords: 1D array of Y coordinates
        footprints: Building footprints GeoDataFrame
        
    Returns:
        Masked hotspot index array
    """
    logger.info("Applying building mask...")
    
    masked_index = hotspot_index.copy()
    
    # Create ground mask (same logic as SVF/solar)
    # Buffer footprints first
    buffered_footprints = footprints.copy()
    buffered_footprints['geometry'] = footprints.geometry.buffer(0.25)
    
    # Flatten grid coordinates for mask computation
    X, Y = np.meshgrid(grid_x_coords, grid_y_coords)
    grid_x_flat = X.flatten()
    grid_y_flat = Y.flatten()
    
    ground_mask_flat = compute_ground_mask(
        grid_x_flat, grid_y_flat, buffered_footprints
    )
    
    # Reshape to 2D
    ground_mask = ground_mask_flat.reshape(hotspot_index.shape)
    
    # Invert mask: True = ground (keep), False = building (mask)
    building_mask = ~ground_mask
    
    # Apply mask: set building pixels to NaN
    masked_index[building_mask] = np.nan
    logger.info(f"  Masked {building_mask.sum()} building pixels ({building_mask.sum() / building_mask.size * 100:.1f}%)")
    
    return masked_index


def classify_hotspot_raster(
    hotspot_index: np.ndarray
) -> np.ndarray:
    """
    Classify hotspot index into categories.
    
    Args:
        hotspot_index: 2D array of hotspot indices
        
    Returns:
        2D array of classifications (0=Baseline, 1=High deprivation, 2=Extreme hotspot)
    """
    valid_values = hotspot_index[~np.isnan(hotspot_index)]
    
    if len(valid_values) == 0:
        return np.full_like(hotspot_index, 0)
    
    p90 = np.percentile(valid_values, 90)
    p80 = np.percentile(valid_values, 80)
    
    classified = np.full_like(hotspot_index, 0)  # Baseline
    classified[hotspot_index >= p80] = 1  # High deprivation
    classified[hotspot_index >= p90] = 2  # Extreme hotspot
    classified[np.isnan(hotspot_index)] = -1  # Masked (buildings)
    
    return classified


def plot_continuous_raster(
    hotspot_index: np.ndarray,
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    output_path: Path
) -> None:
    """
    Create continuous heatmap of hotspot index.
    
    Args:
        hotspot_index: 2D array of hotspot indices
        grid_x_coords: 1D array of X coordinates
        grid_y_coords: 1D array of Y coordinates
        output_path: Path to save visualization
    """
    logger.info("Creating continuous raster visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    extent = [
        grid_x_coords.min(), grid_x_coords.max(),
        grid_y_coords.min(), grid_y_coords.max()
    ]
    
    im = ax.imshow(
        hotspot_index,
        extent=extent,
        cmap='RdYlGn_r',  # Red (high) to Yellow to Green (low)
        vmin=0.0,
        vmax=1.0,
        origin='lower',
        interpolation='nearest'
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Environmental Deprivation Index (Continuous Raster)\nHigher values = Greater deprivation', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Hotspot Index (0-1)', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved continuous raster to {output_path}")


def plot_classified_raster(
    classified: np.ndarray,
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    output_path: Path
) -> None:
    """
    Create classified map with thresholds.
    
    Args:
        classified: 2D array of classifications
        grid_x_coords: 1D array of X coordinates
        grid_y_coords: 1D array of Y coordinates
        output_path: Path to save visualization
    """
    logger.info("Creating classified raster visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    extent = [
        grid_x_coords.min(), grid_x_coords.max(),
        grid_y_coords.min(), grid_y_coords.max()
    ]
    
    # Custom colormap
    from matplotlib.colors import ListedColormap
    colors = ['#D3D3D3', '#FF8C00', '#8B0000', '#000000']  # Grey, Orange, Dark red, Black
    cmap = ListedColormap(colors)
    
    im = ax.imshow(
        classified,
        extent=extent,
        cmap=cmap,
        vmin=-1,
        vmax=2,
        origin='lower',
        interpolation='nearest'
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Environmental Deprivation Hotspots (Classified)\nTop 10%: Extreme, 10-20%: High, Remaining: Baseline', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Custom colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[-0.5, 0.5, 1.5, 2.5])
    cbar.set_ticklabels(['Buildings (masked)', 'Baseline', 'High Deprivation', 'Extreme Hotspot'])
    cbar.set_label('Hotspot Classification', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved classified raster to {output_path}")


def aggregate_to_units(
    hotspot_index: np.ndarray,
    grid_x_coords: np.ndarray,
    grid_y_coords: np.ndarray,
    analysis_units: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate raster values to analysis units for comparison.
    
    Args:
        hotspot_index: 2D array of hotspot indices
        grid_x_coords: 1D array of X coordinates
        grid_y_coords: 1D array of Y coordinates
        analysis_units: GeoDataFrame with unit polygons
        
    Returns:
        GeoDataFrame with aggregated hotspot indices
    """
    logger.info("Aggregating raster to analysis units...")
    
    result = analysis_units.copy()
    result['hotspot_index_mean'] = 0.0
    result['hotspot_index_max'] = 0.0
    
    for idx, unit_row in tqdm(result.iterrows(), total=len(result), desc="  Aggregating"):
        unit_geom = unit_row.geometry
        
        values_in_unit = []
        for i, y in enumerate(grid_y_coords):
            for j, x in enumerate(grid_x_coords):
                point = Point(x, y)
                if unit_geom.contains(point) or unit_geom.touches(point):
                    if not np.isnan(hotspot_index[i, j]):
                        values_in_unit.append(hotspot_index[i, j])
        
        if values_in_unit:
            result.loc[idx, 'hotspot_index_mean'] = np.mean(values_in_unit)
            result.loc[idx, 'hotspot_index_max'] = np.max(values_in_unit)
    
    return result


def plot_unit_level_map(
    analysis_units: gpd.GeoDataFrame,
    output_path: Path
) -> None:
    """
    Create unit-level choropleth map.
    
    Args:
        analysis_units: GeoDataFrame with hotspot_index_mean column
        output_path: Path to save visualization
    """
    logger.info("Creating unit-level visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    analysis_units.plot(
        column='hotspot_index_mean',
        ax=ax,
        cmap='RdYlGn_r',
        legend=True,
        edgecolor='black',
        linewidth=0.5,
        vmin=0.0,
        vmax=1.0,
        legend_kwds={
            'label': 'Mean Hotspot Index (0-1)',
            'shrink': 0.8
        }
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Environmental Deprivation Index (Unit-Level Aggregation)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved unit-level map to {output_path}")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute Morphological Environmental Deprivation Index (Raster Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes a continuous 2D raster of environmental deprivation.
All metrics are relative and distribution-based. No causality is inferred.
        """
    )
    parser.add_argument('--solar', type=str, required=True, help='Path to solar access raster (.npy)')
    parser.add_argument('--svf', type=str, required=True, help='Path to SVF raster (.npy)')
    parser.add_argument('--porosity', type=str, required=True, help='Path to porosity raster (.npy)')
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file (for volume computation)')
    parser.add_argument('--footprints', type=str, required=True, help='Path to building footprints shapefile')
    parser.add_argument('--units', type=str, default=None, help='Optional path to analysis units for aggregation')
    parser.add_argument('--solar-reference', type=float, default=None, help='Reference solar hours (default: median)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/deprivation_raster)')
    
    args = parser.parse_args()
    
    # Setup paths
    solar_path = Path(args.solar)
    svf_path = Path(args.svf)
    porosity_path = Path(args.porosity)
    stl_path = Path(args.stl)
    footprints_path = Path(args.footprints)
    
    # Validate inputs
    for path, name in [(solar_path, 'solar'), (svf_path, 'svf'), 
                       (porosity_path, 'porosity'), (stl_path, 'stl'),
                       (footprints_path, 'footprints')]:
        if not path.exists():
            logger.error(f"{name} file not found: {path}")
            sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "deprivation_raster"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MORPHOLOGICAL ENVIRONMENTAL DEPRIVATION INDEX (RASTER)")
    print("=" * 60)
    print(f"Solar access: {solar_path}")
    print(f"SVF: {svf_path}")
    print(f"Porosity: {porosity_path}")
    print(f"STL: {stl_path}")
    print(f"Footprints: {footprints_path}")
    if args.units:
        print(f"Analysis units: {args.units}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load data
    logger.info("Loading data...")
    full_mesh = load_mesh(stl_path)
    
    footprints = load_building_footprints(
        footprints_path,
        terrain_bounds=full_mesh.bounds,
        buffer_distance=0.0  # No buffer for occupancy computation
    )
    
    # Get common bounds (use porosity as reference since it's highest resolution)
    porosity_map = np.load(porosity_path)
    porosity_bounds = footprints.total_bounds  # Use footprint bounds
    
    # Load rasters with coordinates
    solar_map, solar_x, solar_y = load_raster_with_coords(solar_path, porosity_bounds)
    svf_map, svf_x, svf_y = load_raster_with_coords(svf_path, porosity_bounds)
    porosity_map, porosity_x, porosity_y = load_raster_with_coords(porosity_path, porosity_bounds)
    
    # Use porosity grid as reference (highest resolution)
    grid_x_coords = porosity_x
    grid_y_coords = porosity_y
    
    # Handle different raster resolutions by coordinate-based resampling
    logger.info(f"Raster shapes - Solar: {solar_map.shape}, SVF: {svf_map.shape}, Porosity: {porosity_map.shape}")
    
    # Resample solar and SVF to match porosity grid using coordinate-based lookup
    if solar_map.shape != porosity_map.shape:
        logger.info(f"Resampling solar map from {solar_map.shape} to {porosity_map.shape}...")
        solar_resampled = np.full_like(porosity_map, np.nan)
        for i, y in enumerate(tqdm(porosity_y, desc="  Resampling solar")):
            for j, x in enumerate(porosity_x):
                # Find nearest solar grid point
                y_idx = np.argmin(np.abs(solar_y - y))
                x_idx = np.argmin(np.abs(solar_x - x))
                if 0 <= y_idx < solar_map.shape[0] and 0 <= x_idx < solar_map.shape[1]:
                    solar_resampled[i, j] = solar_map[y_idx, x_idx]
        solar_map = solar_resampled
        logger.info(f"  Resampled solar map to {solar_map.shape}")
    
    if svf_map.shape != porosity_map.shape:
        logger.info(f"Resampling SVF map from {svf_map.shape} to {porosity_map.shape}...")
        svf_resampled = np.full_like(porosity_map, np.nan)
        for i, y in enumerate(tqdm(porosity_y, desc="  Resampling SVF")):
            for j, x in enumerate(porosity_x):
                y_idx = np.argmin(np.abs(svf_y - y))
                x_idx = np.argmin(np.abs(svf_x - x))
                if 0 <= y_idx < svf_map.shape[0] and 0 <= x_idx < svf_map.shape[1]:
                    svf_resampled[i, j] = svf_map[y_idx, x_idx]
        svf_map = svf_resampled
        logger.info(f"  Resampled SVF map to {svf_map.shape}")
    
    # Compute pixel-level occupancy pressure
    grid_spacing = grid_x_coords[1] - grid_x_coords[0] if len(grid_x_coords) > 1 else 2.0
    occupancy_map = compute_pixel_level_occupancy(
        grid_x_coords, grid_y_coords, footprints, full_mesh,
        z_threshold=None, grid_spacing=grid_spacing
    )
    
    # Compute deficits
    solar_deficit, ventilation_deficit, occupancy_score = compute_raster_deficits(
        solar_map, svf_map, porosity_map, occupancy_map,
        solar_reference=args.solar_reference
    )
    
    # Compute hotspot index
    hotspot_index = compute_hotspot_index_raster(
        solar_deficit, ventilation_deficit, occupancy_score
    )
    
    # Apply building mask
    hotspot_index_masked = apply_building_mask(
        hotspot_index, grid_x_coords, grid_y_coords, footprints
    )
    
    # Classify hotspots
    classified = classify_hotspot_raster(hotspot_index_masked)
    
    # Save raster outputs
    np.save(output_dir / "hotspot_index.npy", hotspot_index_masked)
    logger.info(f"  Saved hotspot index raster to {output_dir / 'hotspot_index.npy'}")
    
    # Create visualizations
    plot_continuous_raster(
        hotspot_index_masked, grid_x_coords, grid_y_coords,
        output_dir / "hotspot_continuous.png"
    )
    
    plot_classified_raster(
        classified, grid_x_coords, grid_y_coords,
        output_dir / "hotspot_classified.png"
    )
    
    # Aggregate to units if provided
    if args.units:
        units_path = Path(args.units)
        if units_path.exists():
            analysis_units = gpd.read_file(str(units_path))
            units_with_index = aggregate_to_units(
                hotspot_index_masked, grid_x_coords, grid_y_coords, analysis_units
            )
            
            output_gpkg = output_dir / "deprivation_units.gpkg"
            units_with_index.to_file(output_gpkg, driver="GPKG")
            logger.info(f"  Saved unit aggregation to {output_gpkg}")
            
            plot_unit_level_map(units_with_index, output_dir / "hotspot_units.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

