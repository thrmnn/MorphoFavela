#!/usr/bin/env python3
"""
Compute Sky View Factor (SVF) along street centerlines.

This script computes SVF at sample points along street centerlines, providing
street-level environmental analysis. It complements the grid-based SVF analysis
by focusing specifically on pedestrian experience along street networks.

Usage:
    python scripts/compute_svf_streets.py \
        --stl data/vidigal/raw/full_scan.stl \
        --roads data/vidigal/raw/roads_vidigal.shp \
        --spacing 3.0 \
        --height 1.5
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
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not available - DTM elevation extraction will be disabled. Install with: pip install rasterio")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
from src.svf_utils import (
    load_mesh,
    extract_terrain_surface
)
from scripts.compute_svf import generate_sky_patches

# Import compute_svf function - defined in compute_svf.py
def compute_svf(
    ground_points: np.ndarray,
    sky_patches: np.ndarray,
    full_mesh: pv.PolyData,
    evaluation_height: float
) -> np.ndarray:
    """
    Compute SVF for each ground point (reused from compute_svf.py).
    
    For each point and each sky patch:
    1. Cast a ray toward the patch centroid
    2. Use PyVista ray intersection to test obstruction
    3. Compute SVF as: number of visible patches / total patches
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates
        sky_patches: Array of shape (M, 3) with sky patch centroids
        full_mesh: Full scene mesh (terrain + buildings)
        evaluation_height: Height above ground for evaluation (meters)
        
    Returns:
        Array of SVF values (0-1) for each ground point
    """
    logger.info(f"Computing SVF for {len(ground_points)} points...")
    logger.info(f"  Evaluation height: {evaluation_height}m")
    logger.info(f"  Sky patches: {len(sky_patches)}")
    
    svf_values = []
    
    # Create observer points (ground points + evaluation height)
    observer_points = ground_points.copy()
    observer_points[:, 2] += evaluation_height
    
    # Compute SVF for each observer point
    pbar = tqdm(total=len(observer_points), desc="Computing SVF", unit="points")
    
    for i, observer in enumerate(observer_points):
        visible_patches = 0
        
        for patch_centroid in sky_patches:
            # Ray direction from observer to patch centroid
            ray_direction = patch_centroid - observer
            ray_length = np.linalg.norm(ray_direction)
            ray_direction = ray_direction / ray_length  # Normalize
            
            # Cast ray and check for intersection
            ray_end = observer + ray_direction * ray_length
            intersection, cell_id = full_mesh.ray_trace(observer, ray_end)
            
            # If no intersection, patch is visible
            if len(intersection) == 0:
                visible_patches += 1
        
        # SVF = visible patches / total patches
        svf = visible_patches / len(sky_patches)
        svf_values.append(svf)
        
        # Update progress bar
        if len(svf_values) > 0:
            current_mean = np.mean(svf_values)
            pbar.set_postfix({
                'mean_svf': f'{current_mean:.3f}',
                'current': f'{svf:.3f}'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    return np.array(svf_values)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sample_points_along_line(line: LineString, spacing: float) -> list:
    """
    Sample points along a LineString at regular intervals.
    
    Args:
        line: Shapely LineString geometry
        spacing: Distance between sample points (meters)
        
    Returns:
        List of tuples: (Point geometry, distance_along)
    """
    points = []
    length = line.length
    
    if length < spacing:
        # Line is shorter than spacing, just use midpoint
        point = line.interpolate(length / 2)
        points.append((point, length / 2))
        return points
    
    # Generate points at regular intervals
    distance = 0.0
    while distance <= length:
        point = line.interpolate(distance)
        points.append((point, distance))
        distance += spacing
    
    # Always include the end point if not already included
    if len(points) > 0 and points[-1][0].distance(line.interpolate(length)) > spacing / 2:
        end_point = line.interpolate(length)
        points.append((end_point, length))
    
    return points


def extract_elevation_from_dtm(point: Point, dtm_path: Path) -> float:
    """
    Extract elevation from DTM raster at a given point.
    
    Args:
        point: Shapely Point with x, y coordinates (in DTM's coordinate system)
        dtm_path: Path to DTM raster file
        
    Returns:
        Elevation (Z coordinate) in meters, or NaN if NoData/invalid
    """
    if not HAS_RASTERIO:
        logger.warning("rasterio not available - cannot extract elevation from DTM")
        return np.nan
    
    try:
        with rasterio.open(dtm_path) as src:
            # Sample the raster at the point location
            values = list(src.sample([(point.x, point.y)]))
            
            if not values:
                return np.nan
            
            value = float(values[0])
            
            # Check for NoData value (common values: -9999, very large numbers, NaN)
            nodata = src.nodata
            if nodata is not None:
                # Handle very large NoData values (like 3.4e+38)
                if abs(value - nodata) < 1e-6 or value > 1e6:
                    return np.nan
            
            # Check for NaN or invalid values
            if np.isnan(value) or not np.isfinite(value):
                return np.nan
            
            # Sanity check: elevation should be reasonable (between -100m and 10000m)
            if value < -100 or value > 10000:
                return np.nan
            
            return value
    except Exception as e:
        logger.warning(f"  Error reading DTM for point ({point.x:.1f}, {point.y:.1f}): {e}")
        return np.nan


def extract_elevation_from_mesh(point: Point, terrain: pv.PolyData) -> float:
    """
    Extract elevation from terrain mesh at a given point.
    
    Args:
        point: Shapely Point with x, y coordinates
        terrain: PyVista terrain surface mesh
        
    Returns:
        Elevation (Z coordinate) in meters
    """
    # Create 3D point (use low Z to project upward)
    point_3d = np.array([point.x, point.y, terrain.bounds[4] - 100])
    
    # Find closest point on terrain
    closest_idx = terrain.find_closest_point(point_3d)
    closest_point = terrain.points[closest_idx]
    
    return float(closest_point[2])


def sample_street_points(
    roads_gdf: gpd.GeoDataFrame,
    spacing: float,
    dtm_path: Path = None,
    terrain: pv.PolyData = None
) -> gpd.GeoDataFrame:
    """
    Generate sample points along all street centerlines.
    
    Args:
        roads_gdf: GeoDataFrame with LineString geometries
        spacing: Distance between sample points (meters)
        dtm_path: Optional path to DTM raster for elevation extraction
        terrain: Optional terrain mesh for elevation extraction (fallback)
        
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
            # Extract elevation
            # Prefer terrain mesh interpolation (works with transformed coordinates)
            # DTM would require coordinate transformation which is more complex
            if terrain is not None:
                z = extract_elevation_from_mesh(point, terrain)
            elif dtm_path and dtm_path.exists() and use_dtm_original_crs:
                # Only use DTM if explicitly requested and original CRS coordinates available
                z = extract_elevation_from_dtm(point, dtm_path)
            else:
                z = np.nan
            
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
    svf_values: np.ndarray,
    roads_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate SVF statistics per street segment.
    
    Args:
        points_gdf: GeoDataFrame with street sample points
        svf_values: Array of SVF values for each point
        roads_gdf: Original roads GeoDataFrame
        
    Returns:
        GeoDataFrame with segment-level statistics
    """
    logger.info("Aggregating segment-level statistics...")
    
    # Add SVF values to points
    points_gdf = points_gdf.copy()
    points_gdf['svf'] = svf_values
    
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
            'svf_mean': float(segment_points['svf'].mean()),
            'svf_min': float(segment_points['svf'].min()),
            'svf_max': float(segment_points['svf'].max()),
            'svf_std': float(segment_points['svf'].std()),
            'svf_median': float(segment_points['svf'].median())
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


def create_street_svf_map(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame = None,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None
):
    """
    Create a map showing streets colored by SVF values.
    
    Args:
        segments_gdf: GeoDataFrame with segment-level SVF statistics
        points_gdf: Optional GeoDataFrame with point-level SVF (for detailed view)
        building_footprints: Optional building footprints for context
        output_path: Path to save the map
    """
    logger.info("Creating street SVF map...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot building footprints as background (if available)
    if building_footprints is not None:
        building_footprints.plot(
            ax=ax, facecolor='lightgrey', edgecolor='black', 
            linewidth=0.5, alpha=0.5, label='Buildings'
        )
    
    # Plot street segments colored by mean SVF
    segments_gdf.plot(
        ax=ax, column='svf_mean', cmap='RdYlGn',
        vmin=0, vmax=1, linewidth=3, legend=True,
        legend_kwds={'label': 'Mean SVF (0-1)', 'shrink': 0.8},
        label='Street segments'
    )
    
    # Optionally plot points (for detailed view)
    if points_gdf is not None and len(points_gdf) < 1000:  # Only if not too many points
        points_gdf.plot(
            ax=ax, column='svf', cmap='RdYlGn',
            vmin=0, vmax=1, markersize=2, alpha=0.6, zorder=10
        )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Street-Level Sky View Factor (SVF)\nColored by Mean SVF per Segment', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if building_footprints is not None:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved street SVF map to {output_path}")


def create_statistics_plots(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    output_dir: Path
):
    """
    Create statistical distribution plots.
    
    Args:
        segments_gdf: GeoDataFrame with segment-level statistics
        points_gdf: GeoDataFrame with point-level SVF
        output_dir: Output directory
    """
    logger.info("Creating statistics plots...")
    
    # Distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(points_gdf['svf'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('SVF Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Street-Level SVF Values', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_svf = points_gdf['svf'].mean()
    median_svf = points_gdf['svf'].median()
    ax.axvline(mean_svf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_svf:.3f}')
    ax.axvline(median_svf, color='green', linestyle='--', linewidth=2, label=f'Median: {median_svf:.3f}')
    ax.legend()
    
    plt.tight_layout()
    hist_path = output_dir / "street_svf_distribution.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved distribution plot to {hist_path}")
    
    # Box plot by street (if not too many streets)
    if len(segments_gdf) <= 20:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create box plot data
        box_data = []
        labels = []
        for idx, row in segments_gdf.iterrows():
            segment_points = points_gdf[points_gdf['segment_idx'] == row['segment_idx']]
            if len(segment_points) > 0:
                box_data.append(segment_points['svf'].values)
                name = row['street_name'] if pd.notna(row['street_name']) else f"Segment {row['segment_idx']}"
                labels.append(name[:30])  # Truncate long names
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_ylabel('SVF Value', fontsize=12)
            ax.set_title('SVF Distribution by Street Segment', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            box_path = output_dir / "street_svf_by_segment.png"
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved box plot to {box_path}")


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute Sky View Factor along street centerlines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file')
    parser.add_argument('--roads', type=str, required=True, help='Path to road network shapefile')
    parser.add_argument('--footprints', type=str, default=None, help='Path to building footprints (optional, for visualization)')
    parser.add_argument('--dtm', type=str, default=None, help='Path to DTM raster (optional, preferred for elevation)')
    parser.add_argument('--spacing', type=float, default=3.0, help='Distance between sample points (meters, default: 3.0)')
    parser.add_argument('--height', type=float, default=1.5, help='Evaluation height above ground (meters, default: 1.5)')
    parser.add_argument('--sky-patches', type=int, default=145, help='Number of sky patches (default: 145)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/svf_streets)')
    parser.add_argument('--area', type=str, default=None, help='Area name (e.g., vidigal, copacabana) for automatic path resolution')
    
    args = parser.parse_args()
    
    # Setup paths
    stl_path = Path(args.stl)
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        sys.exit(1)
    
    roads_path = Path(args.roads)
    if not roads_path.exists():
        logger.error(f"Road network file not found: {roads_path}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.area:
        from src.config import get_area_output_dir
        output_dir = get_area_output_dir(args.area) / "svf_streets"
    else:
        output_dir = PROJECT_ROOT / "outputs" / "svf_streets"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STREET-LEVEL SKY VIEW FACTOR (SVF) COMPUTATION")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    print(f"Road network: {roads_path}")
    print(f"Point spacing: {args.spacing}m")
    print(f"Evaluation height: {args.height}m (pedestrian eye level)")
    print(f"Sky patches: {args.sky_patches}")
    if args.dtm:
        print(f"DTM raster: {args.dtm}")
    if args.footprints:
        print(f"Building footprints: {args.footprints}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load mesh and terrain
    mesh = load_mesh(stl_path)
    terrain = extract_terrain_surface(mesh)
    
    # Load road network
    logger.info(f"Loading road network from {roads_path}...")
    roads_gdf = gpd.read_file(roads_path)
    logger.info(f"  Loaded {len(roads_gdf)} road segments")
    logger.info(f"  Original CRS: {roads_gdf.crs}")
    logger.info(f"  Road bounds: {roads_gdf.total_bounds}")
    
    # Handle coordinate system transformation if needed
    # STL mesh is typically in local coordinates, roads may be in UTM
    # Detect and apply translation similar to building footprints
    terrain_bounds = terrain.bounds
    stl_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
    stl_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
    
    roads_center_x = (roads_gdf.total_bounds[0] + roads_gdf.total_bounds[2]) / 2
    roads_center_y = (roads_gdf.total_bounds[1] + roads_gdf.total_bounds[3]) / 2
    
    dx = stl_center_x - roads_center_x
    dy = stl_center_y - roads_center_y
    
    # Check if translation is needed (if centers are very far apart)
    if abs(dx) > 100 or abs(dy) > 100:  # Threshold: 100m
        logger.info(f"  Detected coordinate system mismatch - transforming to local coordinates")
        logger.info(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
        roads_gdf.geometry = roads_gdf.geometry.translate(xoff=dx, yoff=dy)
        logger.info(f"  Transformed road bounds: {roads_gdf.total_bounds}")
    else:
        logger.info(f"  Coordinate systems appear aligned (translation < 100m)")
    
    # Sample points along streets
    dtm_path = Path(args.dtm) if args.dtm else None
    points_gdf = sample_street_points(
        roads_gdf, args.spacing, dtm_path=dtm_path, terrain=terrain
    )
    
    # Remove points with invalid elevation
    valid_mask = ~points_gdf.geometry.apply(lambda p: np.isnan(p.z))
    if not valid_mask.all():
        logger.warning(f"  Removed {np.sum(~valid_mask)} points with invalid elevation")
        points_gdf = points_gdf[valid_mask].copy()
    
    # Convert points to numpy array for SVF computation
    # Extract coordinates, handling coordinate system transformation if needed
    # For now, assume STL and roads are in compatible coordinate systems
    # If not, we'd need to transform here
    
    street_points_3d = np.array([
        [geom.x, geom.y, geom.z] for geom in points_gdf.geometry
    ])
    
    logger.info(f"Computing SVF for {len(street_points_3d)} street points...")
    
    # Generate sky patches
    sky_patches, _ = generate_sky_patches(args.sky_patches)
    
    # Compute SVF
    svf_values = compute_svf(street_points_3d, sky_patches, mesh, args.height)
    
    # Add SVF values to points GeoDataFrame
    points_gdf = points_gdf.copy()
    points_gdf['svf'] = svf_values
    
    # Aggregate to segment level
    segments_gdf = aggregate_segment_statistics(points_gdf, svf_values, roads_gdf)
    
    # Save results
    logger.info("Saving results...")
    
    # Point-level output
    points_output = output_dir / "street_svf_points.gpkg"
    points_gdf.to_file(points_output, driver='GPKG')
    logger.info(f"  Saved point-level results to {points_output}")
    
    # Segment-level output
    segments_output = output_dir / "street_svf_segments.gpkg"
    segments_gdf.to_file(segments_output, driver='GPKG')
    logger.info(f"  Saved segment-level results to {segments_output}")
    
    # CSV summary
    csv_output = output_dir / "street_svf_statistics.csv"
    segments_gdf.drop(columns=['geometry']).to_csv(csv_output, index=False)
    logger.info(f"  Saved statistics to {csv_output}")
    
    # Visualizations
    logger.info("Generating visualizations...")
    
    # Load building footprints if provided
    building_footprints = None
    if args.footprints:
        footprints_path = Path(args.footprints)
        if footprints_path.exists():
            from src.svf_utils import load_building_footprints
            building_footprints = load_building_footprints(
                footprints_path,
                terrain_bounds=terrain.bounds,
                buffer_distance=0.0  # No buffer needed for visualization
            )
    
    # Street SVF map
    map_path = output_dir / "street_svf_map.png"
    create_street_svf_map(segments_gdf, points_gdf, building_footprints, map_path)
    
    # Statistics plots
    create_statistics_plots(segments_gdf, points_gdf, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("STREET SVF SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total street segments: {len(segments_gdf)}")
    print(f"Total sample points: {len(points_gdf)}")
    print(f"\nPoint-level SVF:")
    print(f"  Mean: {points_gdf['svf'].mean():.4f}")
    print(f"  Std:  {points_gdf['svf'].std():.4f}")
    print(f"  Min:  {points_gdf['svf'].min():.4f}")
    print(f"  Max:  {points_gdf['svf'].max():.4f}")
    print(f"  Median: {points_gdf['svf'].median():.4f}")
    print(f"\nSegment-level SVF (mean values):")
    print(f"  Mean: {segments_gdf['svf_mean'].mean():.4f}")
    print(f"  Min:  {segments_gdf['svf_mean'].min():.4f}")
    print(f"  Max:  {segments_gdf['svf_mean'].max():.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

