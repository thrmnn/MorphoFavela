#!/usr/bin/env python3
"""
Sky Exposure Plane Exceedance Analysis

This script quantifies how much of the existing built volume in an informal settlement
exceeds a reference sky exposure plane envelope. The sky exposure plane is an environmental
performance envelope that defines the maximum allowable built form based on an inclined plane
rising from building footprint boundaries at a specified angle.

The analysis evaluates environmental implications (solar access and ventilation) by comparing
actual built form against this geometric envelope, NOT legal code compliance.

Usage:
    python scripts/analyze_sky_exposure.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --angle 45.0
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import argparse
import sys
from tqdm import tqdm
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_utils import load_mesh, load_building_footprints

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_building_meshes(
    full_mesh: pv.PolyData,
    footprints: gpd.GeoDataFrame,
    terrain_bounds: tuple,
    z_threshold: float = None
) -> dict:
    """
    Extract individual building meshes from the full scene mesh using footprints.
    
    Args:
        full_mesh: Full scene mesh (terrain + buildings)
        footprints: GeoDataFrame with building footprint polygons
        terrain_bounds: Terrain bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
        z_threshold: Optional Z threshold to separate terrain from buildings
        
    Returns:
        Dictionary mapping building index to PyVista mesh
    """
    logger.info("Extracting building meshes from scene...")
    
    if z_threshold is None:
        # Estimate terrain level as bottom 10% of mesh Z coordinates
        z_coords = full_mesh.points[:, 2]
        z_threshold = np.percentile(z_coords, 10)
        logger.info(f"  Estimated terrain level: {z_threshold:.2f}m")
    
    building_meshes = {}
    
    # Get mesh bounds
    mesh_bounds = full_mesh.bounds
    
    # For each building footprint, extract the corresponding mesh portion
    for idx, footprint in tqdm(footprints.iterrows(), total=len(footprints), desc="  Extracting buildings"):
        # Get footprint bounds
        geom = footprint.geometry
        if geom is None or geom.is_empty:
            continue
        
        # Get footprint bounding box
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        
        # Create a bounding box filter for this building
        # Include some vertical range above terrain
        x_min, y_min, x_max, y_max = bounds
        z_min = z_threshold
        z_max = mesh_bounds[5]  # Top of mesh
        
        # Extract points within footprint bounding box
        mask = (
            (full_mesh.points[:, 0] >= x_min) & (full_mesh.points[:, 0] <= x_max) &
            (full_mesh.points[:, 1] >= y_min) & (full_mesh.points[:, 1] <= y_max) &
            (full_mesh.points[:, 2] >= z_min) & (full_mesh.points[:, 2] <= z_max)
        )
        
        if not mask.any():
            continue
        
        # Extract cells that have at least one point in the bounding box
        building_points = full_mesh.points[mask]
        
        if len(building_points) == 0:
            continue
        
        # Create a simple mesh for this building
        # For now, we'll use a bounding box approximation
        # In a more sophisticated version, we could use actual footprint clipping
        building_mesh = pv.PolyData(building_points)
        
        if building_mesh.n_points > 0:
            building_meshes[idx] = {
                'mesh': building_mesh,
                'footprint': geom,
                'bounds': bounds
            }
    
    logger.info(f"  Extracted {len(building_meshes)} building meshes")
    return building_meshes


def get_setback_distance_at_point(
    point: Point,
    footprint: Polygon,
    front_setback: float,
    side_rear_setback: float
) -> float:
    """
    Determine the appropriate setback distance for a point based on its position.
    
    Since we don't have building orientation, we use a simplified approach:
    - Points closer to the longer side use front_setback
    - Points closer to shorter sides use side_rear_setback
    
    Args:
        point: Point to check
        footprint: Building footprint polygon
        front_setback: Front setback distance (meters)
        side_rear_setback: Side/rear setback distance (meters)
        
    Returns:
        Appropriate setback distance for this point
    """
    # Get footprint bounds to determine orientation
    bounds = footprint.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    # Determine which dimension is longer (front side)
    if width >= height:
        # Front is along X axis (longer dimension)
        # Check if point is closer to front/back (X edges) or sides (Y edges)
        dist_to_x_edge = min(abs(point.x - bounds[0]), abs(point.x - bounds[2]))
        dist_to_y_edge = min(abs(point.y - bounds[1]), abs(point.y - bounds[3]))
        
        # If closer to X edges (front/back), use front setback
        # If closer to Y edges (sides), use side setback
        if dist_to_x_edge < dist_to_y_edge:
            return front_setback
        else:
            return side_rear_setback
    else:
        # Front is along Y axis (longer dimension)
        dist_to_x_edge = min(abs(point.x - bounds[0]), abs(point.x - bounds[2]))
        dist_to_y_edge = min(abs(point.y - bounds[1]), abs(point.y - bounds[3]))
        
        # If closer to Y edges (front/back), use front setback
        # If closer to X edges (sides), use side setback
        if dist_to_y_edge < dist_to_x_edge:
            return front_setback
        else:
            return side_rear_setback


def create_setback_polygon(
    footprint: Polygon,
    front_setback: float,
    side_rear_setback: float
) -> Polygon:
    """
    Create a setback polygon by shrinking the footprint.
    
    Uses different setbacks for front (larger) vs side/rear (smaller).
    Since we don't have orientation information, we'll use a simplified approach:
    - Use the minimum setback (side_rear) for all sides as a conservative estimate
    - Or use front_setback as maximum if we want to be more permissive
    
    Args:
        footprint: Building footprint polygon
        front_setback: Front setback distance (meters)
        side_rear_setback: Side/rear setback distance (meters)
        
    Returns:
        Setback polygon (shrunk footprint)
    """
    # For now, use the minimum setback (side_rear) for all sides
    # This is conservative and ensures we don't overestimate allowable height
    # In a full implementation, would need building orientation to determine front vs side/rear
    setback_distance = min(front_setback, side_rear_setback)
    
    # Create setback by buffering inward (negative buffer)
    setback_poly = footprint.buffer(-setback_distance)
    
    # If buffer results in empty or invalid geometry, use a very small buffer
    if setback_poly.is_empty or not setback_poly.is_valid:
        # Use a minimal setback to avoid invalid geometry
        setback_poly = footprint.buffer(-0.1)
    
    return setback_poly


def get_envelope_height_at_point(
    point: Point,
    footprint: Polygon,
    angle_degrees: float,
    base_height: float = 0.0,
    front_setback: float = 5.0,
    side_rear_setback: float = 3.0,
    max_height: float = None,
    origin_z: float = 0.0
) -> float:
    """
    Calculate the sky exposure plane envelope height at a given (x, y) point.
    
    The envelope allows full height up to base_height, then applies the sky exposure
    plane from setback boundaries: height = base_height + (distance_from_setback * tan(angle))
    
    Args:
        point: Point geometry (x, y)
        footprint: Building footprint polygon
        angle_degrees: Sky exposure plane angle in degrees
        base_height: Base height allowed before sky plane applies (meters)
        front_setback: Front setback distance (meters)
        side_rear_setback: Side/rear setback distance (meters)
        max_height: Optional maximum height cap
        origin_z: Z coordinate of footprint base
        
    Returns:
        Envelope Z coordinate at this point
    """
    angle_rad = np.deg2rad(angle_degrees)
    tan_angle = np.tan(angle_rad)
    
    # Get appropriate setback distance for this point
    setback_dist = get_setback_distance_at_point(point, footprint, front_setback, side_rear_setback)
    
    # Create setback polygon using the appropriate setback
    setback_poly = footprint.buffer(-setback_dist)
    
    # Handle invalid geometry
    if setback_poly.is_empty or not setback_poly.is_valid:
        setback_poly = footprint.buffer(-0.1)
        setback_dist = 0.1
    
    # Get footprint boundary
    if hasattr(footprint, 'exterior'):
        footprint_boundary = footprint.exterior
    else:
        footprint_boundary = footprint
    
    # Calculate distance from footprint edge
    dist_to_edge = footprint_boundary.distance(point)
    
    # If point is within setback area (between edge and setback line)
    if dist_to_edge < setback_dist:
        # Within setback area: allowed height = base_height
        envelope_height = base_height
    else:
        # Outside setback area: calculate distance from setback boundary
        if hasattr(setback_poly, 'exterior'):
            setback_boundary = setback_poly.exterior
        else:
            setback_boundary = setback_poly
        
        dist_to_setback = setback_boundary.distance(point)
        
        # Envelope height = base_height + (distance from setback * tan(angle))
        envelope_height = base_height + (dist_to_setback * tan_angle)
    
    if max_height is not None:
        envelope_height = min(envelope_height, max_height)
    
    return origin_z + envelope_height


def compute_building_volume(mesh: pv.PolyData) -> float:
    """
    Compute the volume of a building mesh.
    
    Args:
        mesh: Building mesh
        
    Returns:
        Volume in cubic meters
    """
    if mesh.n_points < 4:
        return 0.0
    
    # Use convex hull volume as approximation
    # For more accurate results, could use actual mesh volume calculation
    try:
        hull = mesh.convex_hull()
        if hull.n_cells > 0:
            return hull.volume
    except:
        pass
    
    # Fallback: bounding box volume
    bounds = mesh.bounds
    width = bounds[1] - bounds[0]
    depth = bounds[3] - bounds[2]
    height = bounds[5] - bounds[4]
    return width * depth * height


def compute_exceedance(
    building_mesh: pv.PolyData,
    footprint: Polygon,
    angle_degrees: float,
    base_height: float = 0.0,
    front_setback: float = 5.0,
    side_rear_setback: float = 3.0,
    max_height: float = None,
    origin_z: float = 0.0
) -> dict:
    """
    Compute volume exceedance for a building against sky exposure plane.
    
    Args:
        building_mesh: Building mesh
        footprint: Building footprint polygon
        angle_degrees: Sky exposure plane angle
        base_height: Base height allowed before sky plane applies (meters)
        front_setback: Front setback distance (meters)
        side_rear_setback: Side/rear setback distance (meters)
        max_height: Optional maximum height cap
        origin_z: Z coordinate of footprint base
        
    Returns:
        Dictionary with exceedance metrics
    """
    # Compute total building volume
    total_volume = compute_building_volume(building_mesh)
    
    # Get building bounds
    bounds = building_mesh.bounds
    z_min = bounds[4]
    z_max = bounds[5]
    
    # Exceedance calculation:
    # For each point in the building mesh, check if it's above the envelope
    # The envelope height at any (x,y) point is determined by the distance
    # from the footprint edge and the sky plane angle
    
    building_points = building_mesh.points
    exceedance_points = []
    max_exceedance_height = 0.0
    
    for point_coords in building_points:
        x, y, z = point_coords
        
        # Calculate envelope height at this (x,y) location
        point_geom = Point(x, y)
        envelope_z = get_envelope_height_at_point(
            point_geom, footprint, angle_degrees, base_height,
            front_setback, side_rear_setback, max_height, origin_z
        )
        
        # Check if point exceeds envelope
        if z > envelope_z:
            exceedance = z - envelope_z
            exceedance_points.append(exceedance)
            max_exceedance_height = max(max_exceedance_height, exceedance)
    
    # Estimate exceeding volume
    # Simplified: use ratio of exceeding points
    if len(building_points) > 0:
        exceedance_ratio = len(exceedance_points) / len(building_points)
        exceeding_volume = total_volume * exceedance_ratio
    else:
        exceedance_ratio = 0.0
        exceeding_volume = 0.0
    
    return {
        'total_volume': total_volume,
        'exceeding_volume': exceeding_volume,
        'exceedance_ratio': exceedance_ratio * 100,  # Percentage
        'max_exceedance_height': max_exceedance_height,
        'z_min': z_min,
        'z_max': z_max
    }


def create_exceedance_map(
    footprints: gpd.GeoDataFrame,
    exceedance_results: dict,
    output_path: Path
) -> None:
    """
    Create a plan view map showing buildings colored by exceedance ratio.
    
    Args:
        footprints: Building footprints GeoDataFrame
        exceedance_results: Dictionary mapping building index to exceedance metrics
        output_path: Path to save the map
    """
    logger.info("Creating exceedance map...")
    
    # Add exceedance data to footprints
    gdf = footprints.copy()
    gdf['exceedance_ratio'] = 0.0
    gdf['exceeding_volume'] = 0.0
    gdf['max_exceedance_height'] = 0.0
    
    for idx, metrics in exceedance_results.items():
        if idx in gdf.index:
            gdf.loc[idx, 'exceedance_ratio'] = metrics['exceedance_ratio']
            gdf.loc[idx, 'exceeding_volume'] = metrics['exceeding_volume']
            gdf.loc[idx, 'max_exceedance_height'] = metrics['max_exceedance_height']
    
    # Create map
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot buildings colored by exceedance ratio
    gdf.plot(
        column='exceedance_ratio',
        ax=ax,
        cmap='YlOrRd',
        legend=True,
        legend_kwds={'label': 'Exceedance Ratio (%)', 'shrink': 0.8}
    )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Sky Exposure Plane Exceedance Map\n(Percentage of Volume Exceeding Envelope)', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved exceedance map to {output_path}")


def create_vertical_sections(
    full_mesh: pv.PolyData,
    footprints: gpd.GeoDataFrame,
    exceedance_results: dict,
    building_meshes: dict,
    angle_degrees: float,
    base_height: float,
    front_setback: float,
    side_rear_setback: float,
    output_dir: Path,
    n_sections: int = 3
) -> None:
    """
    Generate vertical sections showing actual built form vs sky exposure plane.
    
    Args:
        full_mesh: Full scene mesh
        footprints: Building footprints
        exceedance_results: Exceedance metrics per building
        building_meshes: Extracted building meshes
        angle_degrees: Sky exposure plane angle
        output_dir: Output directory
        n_sections: Number of sections to generate
    """
    logger.info(f"Creating {n_sections} vertical sections...")
    
    # Select buildings with highest exceedance for sections
    sorted_buildings = sorted(
        exceedance_results.items(),
        key=lambda x: x[1]['exceedance_ratio'],
        reverse=True
    )[:n_sections]
    
    for section_idx, (building_idx, metrics) in enumerate(sorted_buildings):
        if building_idx not in building_meshes:
            continue
        
        building_data = building_meshes[building_idx]
        footprint = building_data['footprint']
        building_mesh = building_data['mesh']
        
        # Get footprint bounds for section
        bounds = footprint.bounds
        centroid = footprint.centroid
        
        # Create section line (through centroid, along longest dimension)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        if width > height:
            # Section along X axis
            section_x = np.linspace(bounds[0], bounds[2], 100)
            section_y = np.full_like(section_x, centroid.y)
        else:
            # Section along Y axis
            section_y = np.linspace(bounds[1], bounds[3], 100)
            section_x = np.full_like(section_y, centroid.x)
        
        # Extract building points along section
        z_base = metrics['z_min']
        z_top = metrics['z_max']
        
        # Sample building mesh points near section line
        building_points = building_mesh.points
        section_building_z = []
        section_distances = []
        
        for i, (x, y) in enumerate(zip(section_x, section_y)):
            # Find points near this section location
            point_geom = Point(x, y)
            
            # Find actual building height at this (x,y)
            # Get points within small distance of section line
            mask = (
                (np.abs(building_points[:, 0] - x) < 1.0) &
                (np.abs(building_points[:, 1] - y) < 1.0)
            )
            
            if mask.any():
                building_z_at_point = np.max(building_points[mask, 2])
            else:
                building_z_at_point = z_base
            
            section_building_z.append(building_z_at_point)
            section_distances.append(i * (width if width > height else height) / len(section_x))
        
        # Create section plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot actual building profile
        ax.fill_between(
            section_distances, z_base, section_building_z,
            alpha=0.6, color='#d62728', label='Actual Built Form'
        )
        
        # Plot sky exposure plane envelope
        # Get parameters from main function (will be passed in)
        # For now, use defaults - these should be passed as parameters
        envelope_heights = [
            get_envelope_height_at_point(
                Point(x, y), footprint, angle_degrees,
                base_height, front_setback, side_rear_setback,
                max_height=None, origin_z=z_base
            )
            for x, y in zip(section_x, section_y)
        ]
        
        ax.plot(
            section_distances, envelope_heights,
            'k--', linewidth=2.5, label='Sky Exposure Plane Envelope'
        )
        
        # Highlight exceedance areas
        exceedance_mask = np.array(section_building_z) > np.array(envelope_heights)
        if exceedance_mask.any():
            ax.fill_between(
                np.array(section_distances)[exceedance_mask],
                np.array(envelope_heights)[exceedance_mask],
                np.array(section_building_z)[exceedance_mask],
                alpha=0.4, color='#ff4444', label='Exceedance Area'
            )
        
        ax.set_xlabel('Distance Along Section (meters)', fontsize=12)
        ax.set_ylabel('Height (meters)', fontsize=12)
        ax.set_title(
            f'Vertical Section {section_idx + 1}\n'
            f'Building {building_idx}: {metrics["exceedance_ratio"]:.1f}% exceedance, '
            f'Max exceedance: {metrics["max_exceedance_height"]:.2f}m',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
        
        output_path = output_dir / f"section_{section_idx + 1}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved section {section_idx + 1} to {output_path}")


def print_summary_statistics(exceedance_results: dict) -> None:
    """
    Print summary statistics for exceedance analysis.
    
    Args:
        exceedance_results: Dictionary mapping building index to exceedance metrics
    """
    if not exceedance_results:
        logger.warning("No exceedance results to summarize")
        return
    
    exceedance_ratios = [m['exceedance_ratio'] for m in exceedance_results.values()]
    buildings_with_exceedance = sum(1 for r in exceedance_ratios if r > 0)
    total_buildings = len(exceedance_ratios)
    
    print("\n" + "=" * 60)
    print("SKY EXPOSURE PLANE EXCEEDANCE SUMMARY")
    print("=" * 60)
    print(f"Total buildings analyzed: {total_buildings}")
    print(f"Buildings with exceedance: {buildings_with_exceedance} ({buildings_with_exceedance/total_buildings*100:.1f}%)")
    print(f"Mean exceedance ratio: {np.mean(exceedance_ratios):.2f}%")
    print(f"Max exceedance ratio: {np.max(exceedance_ratios):.2f}%")
    print(f"Median exceedance ratio: {np.median(exceedance_ratios):.2f}%")
    
    total_volumes = [m['total_volume'] for m in exceedance_results.values()]
    exceeding_volumes = [m['exceeding_volume'] for m in exceedance_results.values()]
    
    print(f"\nTotal built volume: {sum(total_volumes):.2f} m³")
    print(f"Total exceeding volume: {sum(exceeding_volumes):.2f} m³")
    print(f"Overall exceedance ratio: {sum(exceeding_volumes)/sum(total_volumes)*100:.2f}%")
    print("=" * 60 + "\n")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Analyze sky exposure plane exceedances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The sky exposure plane is an environmental performance envelope that defines the maximum
allowable built form based on an inclined plane rising from building footprint boundaries.
This analysis evaluates environmental implications (solar access and ventilation), NOT
legal code compliance.
        """
    )
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file')
    parser.add_argument('--footprints', type=str, required=True, help='Path to building footprints shapefile')
    parser.add_argument('--angle', type=float, default=45.0, help='Sky exposure plane angle in degrees (default: 45.0)')
    parser.add_argument('--base-height', type=float, default=7.5, help='Base height allowed before sky plane applies, meters (default: 7.5, ~2-3 floors, range: 6-9m)')
    parser.add_argument('--front-setback', type=float, default=5.0, help='Front setback distance in meters (default: 5.0)')
    parser.add_argument('--side-setback', type=float, default=3.0, help='Side/rear setback distance in meters (default: 3.0)')
    parser.add_argument('--max-height', type=float, default=None, help='Optional maximum height cap (meters)')
    parser.add_argument('--z-threshold', type=float, default=None, help='Z threshold to separate terrain from buildings')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/sky_exposure)')
    parser.add_argument('--n-sections', type=int, default=3, help='Number of vertical sections to generate (default: 3)')
    
    args = parser.parse_args()
    
    # Setup paths
    stl_path = Path(args.stl)
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        sys.exit(1)
    
    footprints_path = Path(args.footprints)
    if not footprints_path.exists():
        logger.error(f"Footprints file not found: {footprints_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "sky_exposure"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SKY EXPOSURE PLANE EXCEEDANCE ANALYSIS")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    print(f"Building footprints: {footprints_path}")
    print(f"Sky exposure angle: {args.angle}°")
    print(f"Base height: {args.base_height}m (2-3 floors)")
    print(f"Front setback: {args.front_setback}m")
    print(f"Side/rear setback: {args.side_setback}m")
    if args.max_height:
        print(f"Maximum height cap: {args.max_height}m")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load mesh
    mesh = load_mesh(stl_path)
    
    # Load building footprints
    footprints = load_building_footprints(
        footprints_path,
        terrain_bounds=mesh.bounds,
        buffer_distance=0.0  # No buffer for this analysis
    )
    
    # Extract building meshes
    building_meshes = extract_building_meshes(
        mesh, footprints, mesh.bounds, z_threshold=args.z_threshold
    )
    
    # Compute exceedances
    logger.info("Computing exceedances...")
    exceedance_results = {}
    
    for building_idx, building_data in tqdm(building_meshes.items(), desc="Computing exceedances"):
        building_mesh = building_data['mesh']
        footprint = building_data['footprint']
        
        # Get base Z from mesh
        z_base = building_mesh.bounds[4]
        
        metrics = compute_exceedance(
            building_mesh,
            footprint,
            args.angle,
            base_height=args.base_height,
            front_setback=args.front_setback,
            side_rear_setback=args.side_setback,
            max_height=args.max_height,
            origin_z=z_base
        )
        
        exceedance_results[building_idx] = metrics
    
    # Print summary statistics
    print_summary_statistics(exceedance_results)
    
    # Create visualizations
    logger.info("Generating visualizations...")
    
    # Exceedance map
    exceedance_map_path = output_dir / "exceedance_map.png"
    create_exceedance_map(footprints, exceedance_results, exceedance_map_path)
    
    # Vertical sections
    create_vertical_sections(
        mesh, footprints, exceedance_results, building_meshes,
        args.angle, args.base_height, args.front_setback, args.side_setback,
        output_dir, n_sections=args.n_sections
    )
    
    # Save results to CSV
    results_df = pd.DataFrame(exceedance_results).T
    results_df.index.name = 'building_idx'
    results_csv = output_dir / "exceedance_results.csv"
    results_df.to_csv(results_csv)
    logger.info(f"  Saved results to {results_csv}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

