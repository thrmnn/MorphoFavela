#!/usr/bin/env python3
"""
Sectional Porosity Computation for Wind Access Analysis

This script computes sectional porosity as a plan-view proxy for wind access,
defined as the fraction of open (void) area within a horizontal slice of the
urban fabric at a specified height.

Sectional porosity at height z is defined as:
    P_s = A_void / A_total = 1 - (A_built / A_total)

where:
    A_void = area not occupied by buildings at height z
    A_total = total area of the analysis domain

Since most buildings extend above pedestrian height, footprints are assumed
to fully block the section at height z.

This is a geometric proxy for wind access analysis, NOT a CFD simulation.

Usage:
    python scripts/compute_sectional_porosity.py --footprints data/raw/vidigal_buildings.shp --grid-spacing 2.0 --height 1.5
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import logging
from shapely.geometry import box
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_footprints(
    footprints_path: Path,
    buffer_distance: float = 0.25
) -> gpd.GeoDataFrame:
    """
    Load and prepare building footprints for porosity computation.
    
    Args:
        footprints_path: Path to building footprints shapefile
        buffer_distance: Optional outward buffer distance in meters
        
    Returns:
        GeoDataFrame with prepared building footprints
    """
    logger.info(f"Loading building footprints from {footprints_path}...")
    footprints = gpd.read_file(str(footprints_path))
    
    logger.info(f"  Loaded {len(footprints)} building footprints")
    logger.info(f"  CRS: {footprints.crs}")
    logger.info(f"  Bounds: {footprints.total_bounds}")
    
    # Ensure CRS is projected (required for accurate area calculations)
    if footprints.crs is None:
        logger.warning("  No CRS defined, assuming local coordinates")
    elif not footprints.crs.is_projected:
        logger.warning(f"  CRS is not projected: {footprints.crs}")
        logger.warning("  Converting to UTM Zone 23S (EPSG:32723) for analysis")
        footprints = footprints.to_crs("EPSG:32723")
    
    # Apply buffer if specified
    if buffer_distance > 0:
        logger.info(f"  Applying {buffer_distance}m buffer...")
        buffered_geoms = footprints.geometry.buffer(buffer_distance)
        footprints = footprints.set_geometry(buffered_geoms)
    
    logger.info(f"  Prepared {len(footprints)} footprints for analysis")
    return footprints


def generate_grid(
    bounds: tuple,
    grid_spacing: float
) -> tuple:
    """
    Generate a regular 2D grid covering the analysis domain.
    
    Args:
        bounds: Bounding box (minx, miny, maxx, maxy)
        grid_spacing: Grid cell spacing in meters
        
    Returns:
        Tuple of (grid_cells, x_coords, y_coords):
        - grid_cells: GeoDataFrame with grid cell polygons
        - x_coords: 1D array of X coordinates for grid reconstruction
        - y_coords: 1D array of Y coordinates for grid reconstruction
    """
    logger.info(f"Generating grid with spacing {grid_spacing}m...")
    
    minx, miny, maxx, maxy = bounds
    
    # Generate grid coordinates (aligned to grid_spacing)
    x_coords = np.arange(minx, maxx, grid_spacing)
    y_coords = np.arange(miny, maxy, grid_spacing)
    
    # Create grid cell polygons using vectorized approach
    n_x = len(x_coords)
    n_y = len(y_coords)
    
    # Create all grid cells at once
    grid_cells = []
    for y in y_coords:
        for x in x_coords:
            cell = box(x, y, x + grid_spacing, y + grid_spacing)
            grid_cells.append(cell)
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=None)
    
    logger.info(f"  Generated {len(grid_cells)} grid cells ({n_y} × {n_x})")
    
    return grid_gdf, x_coords, y_coords


def compute_sectional_porosity(
    grid_cells: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    grid_spacing: float
) -> np.ndarray:
    """
    Compute sectional porosity for each grid cell.
    
    Porosity = 1 - (built_area / cell_area)
    where built_area is the area of intersection between grid cell and building footprints.
    
    Args:
        grid_cells: GeoDataFrame with grid cell polygons
        footprints: GeoDataFrame with building footprints
        grid_spacing: Grid cell spacing (for cell area calculation)
        
    Returns:
        2D NumPy array of porosity values (0-1), shape (n_y, n_x)
    """
    logger.info("Computing sectional porosity...")
    
    # Set CRS for grid cells to match footprints
    grid_cells = grid_cells.set_crs(footprints.crs, allow_override=True)
    
    # Build spatial index for efficient intersection queries
    if not footprints.sindex:
        logger.info("  Building spatial index...")
        footprints.sindex
    
    # Cell area (all cells are square)
    cell_area = grid_spacing * grid_spacing
    
    # Compute porosity for each grid cell
    porosity_values = []
    
    # Use spatial index for efficient intersection
    logger.info("  Computing intersections...")
    for idx, cell_row in tqdm(grid_cells.iterrows(), total=len(grid_cells), desc="  Computing porosity"):
        cell_geom = cell_row.geometry
        
        # Find buildings that intersect this cell using spatial index
        try:
            possible_matches = list(footprints.sindex.query(cell_geom, predicate='intersects'))
        except:
            # Fallback if sindex query fails
            possible_matches = footprints[footprints.geometry.intersects(cell_geom)].index.tolist()
        
        if len(possible_matches) == 0:
            # No buildings in this cell - fully open
            porosity = 1.0
        else:
            # Compute intersection area with all intersecting buildings
            built_area = 0.0
            for match_idx in possible_matches:
                building_geom = footprints.iloc[match_idx].geometry
                intersection = cell_geom.intersection(building_geom)
                if not intersection.is_empty and intersection.area > 0:
                    built_area += intersection.area
            
            # Porosity = void fraction
            porosity = 1.0 - (built_area / cell_area)
            
            # Clamp to [0, 1]
            porosity = max(0.0, min(1.0, porosity))
        
        porosity_values.append(porosity)
    
    # Reshape to 2D array
    # Grid was created row by row (y first, then x in inner loop)
    # So porosity_values is in row-major order: [row0_col0, row0_col1, ..., row1_col0, ...]
    
    # Determine grid dimensions from bounds
    bounds = grid_cells.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    n_x = int(np.ceil(width / grid_spacing))
    n_y = int(np.ceil(height / grid_spacing))
    
    # Ensure dimensions match actual number of cells
    if n_x * n_y != len(porosity_values):
        # Adjust: might be off by one due to rounding
        # Try to infer from actual data
        actual_n = len(porosity_values)
        # Try to find factors
        for test_n_y in range(int(np.sqrt(actual_n)), actual_n + 1):
            if actual_n % test_n_y == 0:
                n_y = test_n_y
                n_x = actual_n // test_n_y
                break
    
    # Reshape porosity values to 2D (row-major order: y first, then x)
    porosity_2d = np.array(porosity_values).reshape(n_y, n_x)
    
    logger.info(f"  Computed porosity for {len(porosity_values)} grid cells")
    logger.info(f"  Mean porosity: {np.mean(porosity_2d):.3f}")
    logger.info(f"  Min porosity: {np.min(porosity_2d):.3f}")
    logger.info(f"  Max porosity: {np.max(porosity_2d):.3f}")
    
    return porosity_2d


def plot_porosity_map(
    porosity_2d: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    output_path: Path
) -> None:
    """
    Create a top-down heatmap visualization of sectional porosity.
    
    Args:
        porosity_2d: 2D NumPy array of porosity values (0-1)
        x_coords: 1D array of X coordinates
        y_coords: 1D array of Y coordinates
        output_path: Path to save the visualization
    """
    logger.info("Creating porosity map visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define extent for imshow
    extent = [
        x_coords.min(), x_coords.max() + (x_coords[1] - x_coords[0]) if len(x_coords) > 1 else x_coords.max(),
        y_coords.min(), y_coords.max() + (y_coords[1] - y_coords[0]) if len(y_coords) > 1 else y_coords.max()
    ]
    
    # Plot porosity heatmap
    im = ax.imshow(
        porosity_2d,
        extent=extent,
        cmap='RdYlGn',  # Red (low porosity) to Yellow to Green (high porosity)
        vmin=0.0,
        vmax=1.0,
        origin='lower',  # North-up orientation
        interpolation='nearest'  # Square pixels
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Sectional Porosity Map\n(Geometric Proxy for Wind Access)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')  # Square pixels
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sectional Porosity', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved porosity map to {output_path}")


def print_summary_statistics(porosity_2d: np.ndarray) -> None:
    """
    Print summary statistics for sectional porosity.
    
    Args:
        porosity_2d: 2D NumPy array of porosity values
    """
    porosity_flat = porosity_2d.flatten()
    
    print("\n" + "=" * 60)
    print("SECTIONAL POROSITY SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total grid cells: {len(porosity_flat)}")
    print(f"Mean porosity: {np.mean(porosity_flat):.3f}")
    print(f"Std porosity: {np.std(porosity_flat):.3f}")
    print(f"Min porosity: {np.min(porosity_flat):.3f}")
    print(f"Max porosity: {np.max(porosity_flat):.3f}")
    print(f"Median porosity: {np.median(porosity_flat):.3f}")
    print(f"10th percentile: {np.percentile(porosity_flat, 10):.3f} (critical low-porosity zones)")
    print(f"90th percentile: {np.percentile(porosity_flat, 90):.3f}")
    
    # Count cells by porosity category
    low_porosity = np.sum(porosity_flat < 0.3)
    medium_porosity = np.sum((porosity_flat >= 0.3) & (porosity_flat < 0.7))
    high_porosity = np.sum(porosity_flat >= 0.7)
    
    print(f"\nPorosity categories:")
    print(f"  Low (<0.3): {low_porosity} cells ({low_porosity/len(porosity_flat)*100:.1f}%)")
    print(f"  Medium (0.3-0.7): {medium_porosity} cells ({medium_porosity/len(porosity_flat)*100:.1f}%)")
    print(f"  High (≥0.7): {high_porosity} cells ({high_porosity/len(porosity_flat)*100:.1f}%)")
    print("=" * 60 + "\n")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute sectional porosity for wind access analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sectional porosity is a geometric proxy for wind access, defined as the fraction
of open (void) area within a horizontal slice of the urban fabric at a specified height.

This is a geometric analysis based on building footprints only, NOT a CFD simulation.
        """
    )
    parser.add_argument('--footprints', type=str, required=True, help='Path to building footprints shapefile')
    parser.add_argument('--grid-spacing', type=float, default=2.0, help='Grid cell spacing in meters (default: 2.0)')
    parser.add_argument('--height', type=float, default=1.5, help='Section height in meters (default: 1.5, pedestrian level)')
    parser.add_argument('--buffer', type=float, default=0.25, help='Footprint buffer distance in meters (default: 0.25)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/porosity)')
    
    args = parser.parse_args()
    
    # Setup paths
    footprints_path = Path(args.footprints)
    if not footprints_path.exists():
        logger.error(f"Footprints file not found: {footprints_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "porosity"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SECTIONAL POROSITY COMPUTATION")
    print("=" * 60)
    print(f"Building footprints: {footprints_path}")
    print(f"Grid spacing: {args.grid_spacing}m")
    print(f"Section height: {args.height}m")
    print(f"Footprint buffer: {args.buffer}m")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load footprints
    footprints = load_footprints(footprints_path, buffer_distance=args.buffer)
    
    # Generate grid
    bounds = footprints.total_bounds  # (minx, miny, maxx, maxy)
    grid_cells, x_coords, y_coords = generate_grid(bounds, args.grid_spacing)
    
    # Set CRS for grid cells
    grid_cells = grid_cells.set_crs(footprints.crs, allow_override=True)
    
    # Compute sectional porosity
    porosity_2d = compute_sectional_porosity(grid_cells, footprints, args.grid_spacing)
    
    # Print summary statistics
    print_summary_statistics(porosity_2d)
    
    # Save porosity raster
    porosity_npy_path = output_dir / "porosity.npy"
    np.save(porosity_npy_path, porosity_2d)
    logger.info(f"  Saved porosity raster to {porosity_npy_path}")
    
    # Save porosity as CSV (x, y, porosity)
    porosity_csv_path = output_dir / "porosity.csv"
    X, Y = np.meshgrid(x_coords, y_coords)
    porosity_df = {
        'x': X.flatten(),
        'y': Y.flatten(),
        'porosity': porosity_2d.flatten()
    }
    import pandas as pd
    pd.DataFrame(porosity_df).to_csv(porosity_csv_path, index=False)
    logger.info(f"  Saved porosity CSV to {porosity_csv_path}")
    
    # Create visualization
    porosity_map_path = output_dir / "porosity_map.png"
    plot_porosity_map(porosity_2d, x_coords, y_coords, porosity_map_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

