#!/usr/bin/env python3
"""
Compute Solar Access along street centerlines.

This script computes solar access (hours of direct sunlight) at sample points along 
street centerlines, providing street-level environmental analysis. It complements 
the grid-based solar access analysis by focusing specifically on pedestrian 
experience along street networks.

Usage:
    python scripts/compute_solar_access_streets.py \
        --stl data/vidigal/raw/full_scan.stl \
        --roads data/vidigal/raw/roads_vidigal.shp \
        --spacing 3.0 \
        --height 1.5 \
        --area vidigal
"""

import numpy as np
import pyvista as pv
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import sys
from shapely.geometry import Point, LineString
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from src.svf_utils import load_mesh, extract_terrain_surface
from scripts.compute_svf_streets import (
    sample_points_along_line,
    extract_elevation_from_mesh
)
from scripts.compute_solar_access import compute_sun_positions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_solar_access_for_street_points(
    observer_points: np.ndarray,
    sun_directions: np.ndarray,
    full_mesh: pv.PolyData,
    ray_length: float = 10000.0
) -> np.ndarray:
    """
    Compute solar access (hours of direct sunlight) for street observer points.
    
    For each point and each sun position:
    1. Cast a ray toward the sun
    2. Test intersection against the STL mesh
    3. If unobstructed → sunlit, else → shaded
    
    Args:
        observer_points: Array of shape (N, 3) with observer point coordinates (already at evaluation height)
        sun_directions: Array of shape (M, 3) with normalized sun direction vectors
        full_mesh: Full scene mesh (terrain + buildings)
        ray_length: Maximum ray length for intersection test (meters)
        
    Returns:
        Array of solar access values (number of unobstructed sun positions) for each point
    """
    logger.info(f"Computing solar access for {len(observer_points)} street points...")
    logger.info(f"  Sun positions: {len(sun_directions)}")
    
    solar_access_steps = []
    
    pbar = tqdm(total=len(observer_points), desc="Computing solar access", unit="points")
    
    for observer in observer_points:
        unobstructed_count = 0
        
        for sun_dir in sun_directions:
            # Ray end point (far away in sun direction)
            ray_end = observer + sun_dir * ray_length
            
            # Cast ray and check for intersection
            intersection, cell_id = full_mesh.ray_trace(observer, ray_end)
            
            # If no intersection, sun is visible (unobstructed)
            if len(intersection) == 0:
                unobstructed_count += 1
        
        solar_access_steps.append(unobstructed_count)
        
        # Update progress bar
        if len(solar_access_steps) > 0:
            current_mean = np.mean(solar_access_steps)
            pbar.set_postfix({
                'mean_steps': f'{current_mean:.2f}',
                'current': f'{unobstructed_count}'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    return np.array(solar_access_steps)


def sample_street_points(
    roads_gdf: gpd.GeoDataFrame,
    spacing: float,
    terrain: pv.PolyData = None
) -> gpd.GeoDataFrame:
    """
    Generate sample points along all street centerlines.
    
    Args:
        roads_gdf: GeoDataFrame with LineString geometries
        spacing: Distance between sample points (meters)
        terrain: Terrain mesh for elevation extraction
        
    Returns:
        GeoDataFrame with Point geometries, elevation, and street metadata
    """
    logger.info(f"Sampling points along streets with {spacing}m spacing...")
    
    all_points = []
    metadata = []
    
    for idx, row in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="  Processing streets"):
        line = row.geometry
        
        if not isinstance(line, LineString):
            logger.warning(f"  Skipping non-LineString geometry at index {idx}")
            continue
        
        # Sample points along this line
        points_with_distances = sample_points_along_line(line, spacing)
        
        # Extract elevation for each point
        for point, distance_along in points_with_distances:
            # Extract elevation from terrain mesh
            if terrain is not None:
                z = extract_elevation_from_mesh(point, terrain)
            else:
                z = 0.0
            
            # Create 3D point
            point_3d = Point(point.x, point.y, z)
            all_points.append(point_3d)
            
            # Store metadata
            metadata.append({
                'segment_idx': idx,
                'distance_along': distance_along,
                'street_name': row.get('nome', row.get('tipo_logra', f'Street_{idx}')),
                'street_type': row.get('tipo_logra', 'Unknown'),
                'hierarchy': row.get('hierarquia', None)
            })
    
    # Create GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        metadata,
        geometry=all_points,
        crs=roads_gdf.crs
    )
    
    logger.info(f"  Generated {len(points_gdf)} sample points")
    return points_gdf


def aggregate_segment_statistics(
    points_gdf: gpd.GeoDataFrame,
    solar_hours: np.ndarray,
    roads_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate solar access statistics per street segment.
    
    Args:
        points_gdf: GeoDataFrame with street sample points
        solar_hours: Array of solar hours values for each point
        roads_gdf: Original roads GeoDataFrame
        
    Returns:
        GeoDataFrame with segment-level statistics
    """
    logger.info("Aggregating segment-level statistics...")
    
    # Add solar values to points
    points_gdf = points_gdf.copy()
    points_gdf['solar_hours'] = solar_hours
    
    # Group by segment
    segment_stats = []
    
    for idx, row in roads_gdf.iterrows():
        segment_points = points_gdf[points_gdf['segment_idx'] == idx]
        
        if len(segment_points) == 0:
            continue
        
        stats = {
            'segment_idx': idx,
            'street_name': row.get('nome', row.get('tipo_logra', f'Street_{idx}')),
            'street_type': row.get('tipo_logra', 'Unknown'),
            'hierarchy': row.get('hierarquia', None),
            'length': row.geometry.length,
            'n_points': len(segment_points),
            'solar_mean': float(segment_points['solar_hours'].mean()),
            'solar_min': float(segment_points['solar_hours'].min()),
            'solar_max': float(segment_points['solar_hours'].max()),
            'solar_std': float(segment_points['solar_hours'].std()),
            'solar_median': float(segment_points['solar_hours'].median())
        }
        segment_stats.append(stats)
    
    # Create GeoDataFrame with original geometries
    segments_gdf = gpd.GeoDataFrame(
        segment_stats,
        geometry=[roads_gdf.loc[s['segment_idx']].geometry for s in segment_stats],
        crs=roads_gdf.crs
    )
    
    logger.info(f"  Aggregated {len(segments_gdf)} street segments")
    return segments_gdf


def create_street_solar_map(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame = None,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None
):
    """
    Create a map showing streets colored by solar access values.
    """
    logger.info("Creating street solar access map...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot building footprints as background (if available)
    if building_footprints is not None:
        building_footprints.plot(
            ax=ax, facecolor='lightgrey', edgecolor='black', 
            linewidth=0.5, alpha=0.5, label='Buildings'
        )
    
    # Plot street segments colored by mean solar access
    max_solar = segments_gdf['solar_mean'].max()
    segments_gdf.plot(
        ax=ax, column='solar_mean', cmap='YlOrRd',
        vmin=0, vmax=max_solar, linewidth=3, legend=True,
        legend_kwds={'label': 'Mean Solar Hours', 'shrink': 0.8},
        label='Street segments'
    )
    
    # Optionally plot points (for detailed view)
    if points_gdf is not None and len(points_gdf) < 1000:
        points_gdf.plot(
            ax=ax, column='solar_hours', cmap='YlOrRd',
            vmin=0, vmax=max_solar, markersize=2, alpha=0.6, zorder=10
        )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Street-Level Solar Access (Winter Solstice)\nColored by Mean Hours per Segment', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved street solar map to {output_path}")
    
    plt.close()


def create_solar_distribution_plot(
    points_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    output_path: Path = None
):
    """
    Create distribution plots for street-level solar access.
    """
    logger.info("Creating solar access distribution plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Point-level distribution
    ax1 = axes[0]
    solar_hours = points_gdf['solar_hours'].values
    ax1.hist(solar_hours, bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(solar_hours), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(solar_hours):.2f}h')
    ax1.axvline(np.median(solar_hours), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {np.median(solar_hours):.2f}h')
    ax1.set_xlabel('Solar Hours', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Point-Level Solar Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Segment-level distribution
    ax2 = axes[1]
    segment_solar = segments_gdf['solar_mean'].values
    ax2.hist(segment_solar, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(segment_solar), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(segment_solar):.2f}h')
    ax2.axvline(np.median(segment_solar), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {np.median(segment_solar):.2f}h')
    ax2.set_xlabel('Mean Solar Hours', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Segment-Level Solar Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved solar distribution plot to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute solar access along street centerlines')
    parser.add_argument('--stl', type=str, required=True, help='Path to STL mesh file')
    parser.add_argument('--roads', type=str, required=True, help='Path to roads shapefile')
    parser.add_argument('--footprints', type=str, default=None, help='Path to building footprints (for visualization)')
    parser.add_argument('--spacing', type=float, default=3.0, help='Point spacing along streets (meters)')
    parser.add_argument('--height', type=float, default=1.5, help='Evaluation height above ground (meters)')
    parser.add_argument('--area', type=str, default=None, help='Area name for output organization')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--latitude', type=float, default=-22.9519, help='Site latitude (default: Rio de Janeiro)')
    parser.add_argument('--longitude', type=float, default=-43.2105, help='Site longitude (default: Rio de Janeiro)')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.area:
        output_dir = PROJECT_ROOT / 'outputs' / args.area / 'solar_streets'
    else:
        output_dir = PROJECT_ROOT / 'outputs' / 'solar_streets'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("STREET-LEVEL SOLAR ACCESS ANALYSIS")
    print("=" * 60)
    print(f"STL file: {args.stl}")
    print(f"Road network: {args.roads}")
    print(f"Point spacing: {args.spacing}m")
    print(f"Evaluation height: {args.height}m")
    print(f"Location: ({args.latitude:.4f}°, {args.longitude:.4f}°)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load mesh and extract terrain
    mesh = load_mesh(args.stl)
    terrain = extract_terrain_surface(mesh)
    
    # Load road network
    logger.info(f"Loading road network from {args.roads}...")
    roads_gdf = gpd.read_file(args.roads)
    logger.info(f"  Loaded {len(roads_gdf)} road segments")
    
    # Check coordinate system
    mesh_center_x = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh_center_y = (mesh.bounds[2] + mesh.bounds[3]) / 2
    road_bounds = roads_gdf.total_bounds
    road_center_x = (road_bounds[0] + road_bounds[2]) / 2
    road_center_y = (road_bounds[1] + road_bounds[3]) / 2
    
    logger.info(f"  Road bounds: {road_bounds}")
    
    # Transform roads if needed
    if abs(road_center_x - mesh_center_x) > 1000 or abs(road_center_y - mesh_center_y) > 1000:
        dx = mesh_center_x - road_center_x
        dy = mesh_center_y - road_center_y
        logger.info(f"  Detected coordinate system mismatch - transforming to local coordinates")
        logger.info(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
        roads_gdf.geometry = roads_gdf.geometry.translate(xoff=dx, yoff=dy)
        logger.info(f"  Transformed road bounds: {roads_gdf.total_bounds}")
    
    # Load building footprints if provided (for visualization)
    building_footprints = None
    if args.footprints:
        logger.info(f"Loading building footprints from {args.footprints}...")
        building_footprints = gpd.read_file(args.footprints)
        # Transform if needed
        fp_bounds = building_footprints.total_bounds
        fp_center_x = (fp_bounds[0] + fp_bounds[2]) / 2
        fp_center_y = (fp_bounds[1] + fp_bounds[3]) / 2
        if abs(fp_center_x - mesh_center_x) > 1000 or abs(fp_center_y - mesh_center_y) > 1000:
            dx = mesh_center_x - fp_center_x
            dy = mesh_center_y - fp_center_y
            building_footprints.geometry = building_footprints.geometry.translate(xoff=dx, yoff=dy)
    
    # Generate sample points along streets
    points_gdf = sample_street_points(roads_gdf, args.spacing, terrain)
    
    # Extract 3D coordinates for computation
    ground_points = np.array([[p.x, p.y, p.z] for p in points_gdf.geometry])
    
    # Add evaluation height
    observer_points = ground_points.copy()
    observer_points[:, 2] += args.height
    
    # Compute sun positions for winter solstice
    sun_directions, sun_times = compute_sun_positions(
        latitude=args.latitude,
        longitude=args.longitude,
        timestep_minutes=60
    )
    
    # Compute solar access
    solar_steps = compute_solar_access_for_street_points(
        observer_points, sun_directions, mesh
    )
    
    # Convert to hours (assuming 1-hour timestep)
    solar_hours = solar_steps.astype(float)  # Each step is 1 hour
    
    # Add to points GeoDataFrame
    points_gdf['solar_hours'] = solar_hours
    
    # Aggregate segment statistics
    segments_gdf = aggregate_segment_statistics(points_gdf, solar_hours, roads_gdf)
    
    # Save results
    logger.info("Saving results...")
    
    points_path = output_dir / 'street_solar_points.gpkg'
    points_gdf.to_file(points_path, driver='GPKG')
    logger.info(f"  Saved point-level results to {points_path}")
    
    segments_path = output_dir / 'street_solar_segments.gpkg'
    segments_gdf.to_file(segments_path, driver='GPKG')
    logger.info(f"  Saved segment-level results to {segments_path}")
    
    # Generate statistics
    stats = {
        'n_points': len(points_gdf),
        'n_segments': len(segments_gdf),
        'solar_mean': float(points_gdf['solar_hours'].mean()),
        'solar_std': float(points_gdf['solar_hours'].std()),
        'solar_min': float(points_gdf['solar_hours'].min()),
        'solar_max': float(points_gdf['solar_hours'].max()),
        'solar_median': float(points_gdf['solar_hours'].median()),
        'below_2h_pct': float((points_gdf['solar_hours'] < 2).mean() * 100),
        'below_3h_pct': float((points_gdf['solar_hours'] < 3).mean() * 100)
    }
    
    stats_df = pd.DataFrame([stats])
    stats_path = output_dir / 'street_solar_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"  Saved statistics to {stats_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    create_street_solar_map(
        segments_gdf, points_gdf, building_footprints,
        output_dir / 'street_solar_map.png'
    )
    
    create_solar_distribution_plot(
        points_gdf, segments_gdf,
        output_dir / 'street_solar_distribution.png'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("STREET-LEVEL SOLAR ACCESS SUMMARY")
    print("=" * 60)
    print(f"Total street segments: {len(segments_gdf)}")
    print(f"Total sample points: {len(points_gdf)}")
    print(f"\nPoint-level statistics:")
    print(f"  Mean solar hours: {stats['solar_mean']:.2f}h")
    print(f"  Median solar hours: {stats['solar_median']:.2f}h")
    print(f"  Min solar hours: {stats['solar_min']:.2f}h")
    print(f"  Max solar hours: {stats['solar_max']:.2f}h")
    print(f"  Points below 2h: {stats['below_2h_pct']:.1f}%")
    print(f"  Points below 3h: {stats['below_3h_pct']:.1f}%")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()


