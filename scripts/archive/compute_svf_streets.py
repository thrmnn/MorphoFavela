#!/usr/bin/env python3
"""
Compute Sky View Factor (SVF) along street centerlines.

This script computes SVF at sample points along street centerlines, providing
street-level environmental analysis. It complements the grid-based SVF analysis
by focusing specifically on pedestrian experience along street networks.

Usage:
    python scripts/compute_svf_streets.py \
        --stl data/vidigal_tls/raw/full_scan.stl \
        --roads data/vidigal_tls/raw/roads_vidigal.shp \
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
from matplotlib import colors, colormaps
from matplotlib.colors import LinearSegmentedColormap
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
from src.svf_compute import generate_sky_patches, compute_svf, get_compute_backend

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


def interpolate_svf_along_segments(
    points_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    interpolation_spacing: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Create continuous SVF representation by interpolating along each street segment.
    
    For each segment, interpolates SVF values between sampled points to create
    a dense, continuous representation of SVF along the street.
    
    Args:
        points_gdf: GeoDataFrame with sampled points and their SVF values
        roads_gdf: GeoDataFrame with street segments
        interpolation_spacing: Distance between interpolated points (meters, default: 0.5)
    
    Returns:
        GeoDataFrame with dense interpolated points along streets with SVF values
    """
    from scipy.interpolate import interp1d
    
    logger.info(f"Interpolating SVF along segments with {interpolation_spacing}m spacing...")
    
    all_interpolated_points = []
    all_metadata = []
    
    for idx, row in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="  Interpolating"):
        line = row.geometry
        
        if not isinstance(line, LineString):
            continue
        
        # Get all points for this segment
        segment_points = points_gdf[points_gdf['segment_idx'] == idx].copy()
        
        if len(segment_points) == 0:
            continue
        
        # Sort by distance_along
        segment_points = segment_points.sort_values('distance_along')
        
        # Extract distances and SVF values
        distances = segment_points['distance_along'].values
        svf_values = segment_points['svf'].values
        
        # Generate dense points along the line
        line_length = line.length
        interp_distances = np.arange(0, line_length + interpolation_spacing, interpolation_spacing)
        interp_distances = np.clip(interp_distances, 0, line_length)  # Ensure within bounds
        
        # Remove duplicates (in case of rounding)
        interp_distances = np.unique(interp_distances)
        
        # Interpolate SVF values
        if len(distances) == 1:
            # Single point - use constant value
            interp_svf = np.full(len(interp_distances), svf_values[0], dtype=float)
        elif len(distances) == 2:
            # Two points - linear interpolation
            interp_func = interp1d(distances, svf_values, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            interp_svf = interp_func(interp_distances)
        else:
            # Multiple points - linear interpolation (preserves min/max, no overshoot)
            interp_func = interp1d(distances, svf_values, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
            interp_svf = interp_func(interp_distances)
        
        # Ensure interp_svf is an array
        interp_svf = np.atleast_1d(interp_svf)
        
        # Clamp SVF to valid range [0, 1]
        interp_svf = np.clip(interp_svf, 0.0, 1.0)
        
        # Create points and metadata
        for dist, svf_val in zip(interp_distances, interp_svf):
            point_2d = line.interpolate(dist)
            # Get elevation from original point if available, or use interpolation
            # Find closest original point for elevation
            closest_idx = np.argmin(np.abs(distances - dist))
            closest_point = segment_points.iloc[closest_idx]
            z = closest_point.geometry.z if hasattr(closest_point.geometry, 'z') else 0.0
            
            point_3d = Point(point_2d.x, point_2d.y, z)
            all_interpolated_points.append(point_3d)
            
            all_metadata.append({
                'segment_idx': idx,
                'distance_along': dist,
                'svf': float(svf_val),
                'is_interpolated': True,
                'street_name': row.get('nome', row.get('tipo_logra', f'Street_{idx}')),
                'street_type': row.get('tipo_logra', 'Unknown'),
                'hierarchy': row.get('hierarquia', None)
            })
    
    # Create GeoDataFrame
    interpolated_gdf = gpd.GeoDataFrame(
        all_metadata,
        geometry=all_interpolated_points,
        crs=points_gdf.crs
    )
    
    logger.info(f"  Generated {len(interpolated_gdf)} interpolated points")
    return interpolated_gdf


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


def create_3d_alignment_debug(
    mesh: pv.PolyData,
    terrain: pv.PolyData,
    roads_gdf: gpd.GeoDataFrame,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None
):
    """
    Create a 3D visualization showing STL mesh, building footprints, and roads for alignment debugging.
    
    Args:
        mesh: Full STL mesh (georeferenced)
        terrain: Terrain surface mesh
        roads_gdf: Road network GeoDataFrame
        building_footprints: Building footprints GeoDataFrame
        output_path: Path to save the visualization
    """
    import pyvista as pv
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # Add terrain surface (semi-transparent)
    terrain_actor = plotter.add_mesh(
        terrain,
        color='lightgray',
        opacity=0.6,
        show_edges=False,
        label='Terrain'
    )
    
    # Add full mesh (buildings) - extract non-terrain faces
    # We'll use a simple approach: show all faces with higher opacity
    buildings_mesh = mesh.copy()
    # Remove terrain faces by checking normals
    buildings_mesh = buildings_mesh.compute_normals(point_normals=False, cell_normals=True)
    # Select faces that are NOT upward-facing (terrain)
    building_mask = buildings_mesh.cell_normals[:, 2] <= 0.5
    if building_mask.any():
        buildings_only = buildings_mesh.extract_cells(building_mask)
        plotter.add_mesh(
            buildings_only,
            color='tan',
            opacity=0.8,
            show_edges=False,
            label='Buildings (STL)'
        )
    
    # Add building footprints as extruded polygons
    if building_footprints is not None and len(building_footprints) > 0:
        # Sample a subset for visualization (too many can be slow)
        n_buildings = min(100, len(building_footprints))
        sample_buildings = building_footprints.sample(n=n_buildings) if len(building_footprints) > n_buildings else building_footprints
        
        for idx, row in sample_buildings.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                # Get exterior coordinates
                coords = np.array(geom.exterior.coords)
                if len(coords) >= 3:
                    # Get base Z from terrain (use first point's elevation)
                    z_base = 0.0  # Simplified - could sample from terrain
                    z_top = 5.0  # 5m height for visualization
                    
                    # Create polygon points (X, Y, Z_base)
                    points_2d = coords[:, :2]  # X, Y only
                    # Create PyVista polygon from 2D points
                    try:
                        # Create a simple polygon mesh
                        poly_points = np.column_stack([points_2d, np.full(len(points_2d), z_base)])
                        # Create faces (triangulate if needed)
                        if len(poly_points) >= 3:
                            # Simple approach: create a surface from the polygon
                            # Use PyVista's built-in polygon creation
                            poly = pv.Polygon(points_2d)
                            # Extrude upward
                            extruded = poly.extrude((0, 0, z_top - z_base))
                            if extruded.n_points > 0:
                                plotter.add_mesh(
                                    extruded,
                                    color='red',
                                    opacity=0.5,
                                    show_edges=True,
                                    edge_color='darkred',
                                    line_width=1
                                )
                    except Exception as e:
                        logger.warning(f"  Failed to visualize building {idx}: {e}")
                        continue
    
    # Add roads as lines elevated above terrain
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = np.array(geom.coords)
            if len(coords) > 0:
                # Create line in 3D (elevate slightly above terrain)
                z_offset = 0.5  # 0.5m above terrain
                coords_3d = np.column_stack([coords[:, 0], coords[:, 1], np.full(len(coords), z_offset)])
                line = pv.Spline(coords_3d)
                plotter.add_mesh(
                    line,
                    color='blue',
                    line_width=3,
                    label='Roads' if idx == roads_gdf.index[0] else None
                )
    
    # Set camera to show overview
    plotter.camera.position = (
        (mesh.bounds[1] + mesh.bounds[0]) / 2 + 200,
        (mesh.bounds[3] + mesh.bounds[2]) / 2 + 200,
        mesh.bounds[5] + 100
    )
    plotter.camera.focal_point = (
        (mesh.bounds[1] + mesh.bounds[0]) / 2,
        (mesh.bounds[3] + mesh.bounds[2]) / 2,
        (mesh.bounds[4] + mesh.bounds[5]) / 2
    )
    plotter.camera.up = (0, 0, 1)
    plotter.camera.zoom(0.8)
    
    # Add axes
    plotter.add_axes()
    
    # Add legend
    plotter.add_legend()
    
    # Save screenshot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    
    logger.info(f"  Saved 3D debug visualization to {output_path}")


def plot_street_debug(
    roads_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    building_footprints: gpd.GeoDataFrame = None,
    isolated_buildings: gpd.GeoDataFrame = None,
    area_filtered_buildings: gpd.GeoDataFrame = None,
    output_path: Path = None,
    spacing: float = None,
    height: float = None,
    building_buffer: float = None,
    building_buffer_geom = None,
    filtered_roads_gdf: gpd.GeoDataFrame = None,
    original_roads_count: int = None,
    intersection_info: gpd.GeoDataFrame = None,
    covered_roads: gpd.GeoDataFrame = None
):
    """
    Create a debug plot showing street centerlines, sample points, and building footprints.
    
    Args:
        roads_gdf: GeoDataFrame with street centerlines
        points_gdf: GeoDataFrame with sample points along streets
        building_footprints: GeoDataFrame with main cluster building footprints
        isolated_buildings: GeoDataFrame with isolated buildings (for visualization)
        area_filtered_buildings: GeoDataFrame with area-filtered buildings (for visualization)
        output_path: Path to save the plot
        spacing: Point spacing used for sampling
        height: Evaluation height
    """
    logger.info(f"Creating street debug plot: {output_path}")
    
    from src.config import MAX_FILTER_AREA, BUILDING_CLUSTER_BUFFER
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot area-filtered buildings in yellow (first, so they appear behind)
    if area_filtered_buildings is not None and len(area_filtered_buildings) > 0:
        if area_filtered_buildings.crs != roads_gdf.crs:
            area_filtered_buildings = area_filtered_buildings.to_crs(roads_gdf.crs)
        area_filtered_buildings.plot(ax=ax, facecolor='yellow', edgecolor='orange', 
                                     linewidth=1.5, alpha=0.5, 
                                     label=f'Area-filtered buildings (area > {MAX_FILTER_AREA}m², n={len(area_filtered_buildings)})')
    
    # Plot isolated buildings (gray/transparent)
    if isolated_buildings is not None and len(isolated_buildings) > 0:
        if isolated_buildings.crs != roads_gdf.crs:
            isolated_buildings = isolated_buildings.to_crs(roads_gdf.crs)
        isolated_buildings.plot(ax=ax, facecolor='lightgray', edgecolor='gray', 
                               linewidth=1, alpha=0.3, 
                               label=f'Isolated buildings (n={len(isolated_buildings)})')
    
    # Plot building buffer zone if provided
    if building_buffer_geom is not None and building_buffer is not None:
        from shapely.geometry import shape
        buffer_gdf = gpd.GeoDataFrame(geometry=[building_buffer_geom], crs=roads_gdf.crs)
        buffer_gdf.plot(ax=ax, facecolor='none', edgecolor='orange', linestyle='--', 
                       linewidth=1.5, alpha=0.7, 
                       label=f'Building buffer zone ({building_buffer}m)')
    
    # Plot building footprints (red) - main cluster
    if building_footprints is not None and len(building_footprints) > 0:
        if building_footprints.crs != roads_gdf.crs:
            building_footprints = building_footprints.to_crs(roads_gdf.crs)
        building_footprints.plot(ax=ax, facecolor='red', edgecolor='darkred', 
                                linewidth=1.5, alpha=0.4, 
                                label=f'Main cluster buildings (n={len(building_footprints)})')
    
    # Plot filtered-out streets (light gray, if any)
    if filtered_roads_gdf is not None and original_roads_count is not None:
        if len(filtered_roads_gdf) < original_roads_count:
            # Find streets that were filtered out
            filtered_out_mask = ~roads_gdf.index.isin(filtered_roads_gdf.index)
            filtered_out_roads = roads_gdf[filtered_out_mask]
            if len(filtered_out_roads) > 0:
                filtered_out_roads.plot(ax=ax, color='lightgray', linewidth=1, 
                                       alpha=0.3, linestyle=':', 
                                       label=f'Filtered-out streets (n={len(filtered_out_roads)})')
    
    # Plot filtered street centerlines (blue) - first so redirected roads appear on top
    if filtered_roads_gdf is not None:
        # Exclude redirected roads from filtered roads to avoid overlap
        if intersection_info is not None and len(intersection_info) > 0:
            redirected_indices = set(intersection_info['road_idx'].values)
            filtered_non_redirected = filtered_roads_gdf[~filtered_roads_gdf.index.isin(redirected_indices)]
            if len(filtered_non_redirected) > 0:
                filtered_non_redirected.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.6, 
                                           label=f'Filtered street centerlines (n={len(filtered_non_redirected)})')
        else:
            filtered_roads_gdf.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.6, 
                                   label=f'Filtered street centerlines (n={len(filtered_roads_gdf)})')
    else:
        # Exclude redirected roads from regular roads
        if intersection_info is not None and len(intersection_info) > 0:
            redirected_indices = set(intersection_info['road_idx'].values)
            non_redirected = roads_gdf[~roads_gdf.index.isin(redirected_indices)]
            if len(non_redirected) > 0:
                non_redirected.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.6, 
                                  label=f'Street centerlines (n={len(non_redirected)})')
        else:
            roads_gdf.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.6, 
                          label=f'Street centerlines (n={len(roads_gdf)})')
    
    # Plot covered roads (roads that still intersect buildings) in dark red
    if covered_roads is not None and len(covered_roads) > 0:
        covered_indices = covered_roads['road_idx'].values
        covered_roads_geom = roads_gdf.loc[covered_indices]
        if len(covered_roads_geom) > 0:
            covered_roads_geom.plot(ax=ax, color='darkred', linewidth=1.0, alpha=0.8, 
                                   linestyle=':', zorder=9,
                                   label=f'Covered roads (SVF=0, n={len(covered_roads_geom)})')
    
    # Plot redirected roads (if any) in purple - on top so they're visible
    if intersection_info is not None and len(intersection_info) > 0:
        redirected_indices = intersection_info['road_idx'].values
        # Exclude covered roads from redirected roads visualization
        if covered_roads is not None and len(covered_roads) > 0:
            covered_indices_set = set(covered_roads['road_idx'].values)
            redirected_indices = [idx for idx in redirected_indices if idx not in covered_indices_set]
        
        if len(redirected_indices) > 0:
            redirected_roads = roads_gdf.loc[redirected_indices]
            if len(redirected_roads) > 0:
                redirected_roads.plot(ax=ax, color='purple', linewidth=1.2, alpha=0.9, 
                                     linestyle='--', zorder=10,
                                     label=f'Redirected roads (n={len(redirected_roads)})')
    
    # Points removed from debug visualization as requested
    
    # Add statistics text
    stats_text = []
    if original_roads_count is not None:
        stats_text.append(f"Original streets: {original_roads_count:,}")
        if filtered_roads_gdf is not None:
            stats_text.append(f"Filtered streets: {len(filtered_roads_gdf):,}")
            if len(filtered_roads_gdf) < original_roads_count:
                reduction = 100 * (1 - len(filtered_roads_gdf) / original_roads_count)
                stats_text.append(f"Filtered out: {reduction:.1f}%")
    else:
        stats_text.append(f"Street segments: {len(roads_gdf):,}")
    stats_text.append(f"Sample points: {len(points_gdf):,}")
    if spacing is not None:
        stats_text.append(f"Point spacing: {spacing}m")
    if height is not None:
        stats_text.append(f"Evaluation height: {height}m")
    total_street_length = roads_gdf.geometry.length.sum()
    stats_text.append(f"Total street length: {total_street_length:.1f}m")
    if len(points_gdf) > 0:
        avg_points_per_segment = len(points_gdf) / len(roads_gdf)
        stats_text.append(f"Avg points/segment: {avg_points_per_segment:.1f}")
    if building_footprints is not None and len(building_footprints) > 0:
        stats_text.append(f"Main cluster: {len(building_footprints):,} buildings")
    if area_filtered_buildings is not None and len(area_filtered_buildings) > 0:
        stats_text.append(f"Area-filtered: {len(area_filtered_buildings):,} buildings")
    if isolated_buildings is not None and len(isolated_buildings) > 0:
        stats_text.append(f"Isolated: {len(isolated_buildings):,} buildings")
    if intersection_info is not None and len(intersection_info) > 0:
        stats_text.append(f"Redirected roads: {len(intersection_info):,}")
    if covered_roads is not None and len(covered_roads) > 0:
        stats_text.append(f"Covered roads (SVF=0): {len(covered_roads):,}")
    
    if stats_text:
        stats_str = "\n".join(stats_text)
        ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    
    title = 'Street-Level SVF Debug Plot'
    if spacing is not None:
        title += f'\n(Point spacing: {spacing}m)'
    if height is not None:
        title += f'\n(Evaluation height: {height}m)'
    if building_buffer is not None and building_buffer > 0:
        title += f'\n(Street filter: {building_buffer}m buffer)'
    if MAX_FILTER_AREA is not None:
        title += f'\n(Area filter: >{MAX_FILTER_AREA}m² excluded)'
    title += f'\n(Cluster filter: {BUILDING_CLUSTER_BUFFER}m buffer)'
    title += '\nBlue = Filtered streets, Purple dashed = Redirected roads, Dark red dotted = Covered (SVF=0)'
    title += '\nGray dotted = Filtered-out streets, Red = Main cluster, Yellow = Area-filtered'
    title += '\nGray = Isolated, Orange dashed = Buffer zone'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved street debug plot to {output_path}")


def create_street_svf_map(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame = None,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None,
    covered_roads: gpd.GeoDataFrame = None,
    roads_gdf: gpd.GeoDataFrame = None
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
    
    # Building footprints should already be in the correct coordinate system
    # (aligned in the main script before calling this function)
    # No need for additional alignment here
    
    # Plot building footprints as background (if available)
    if building_footprints is not None and not building_footprints.empty:
        logger.info(f"  Plotting {len(building_footprints)} building footprints")
        # Ensure buildings are in the same CRS as segments
        if building_footprints.crs != segments_gdf.crs:
            building_footprints = building_footprints.to_crs(segments_gdf.crs)
        building_footprints.plot(
            ax=ax, facecolor='lightgrey', edgecolor='darkgrey', 
            linewidth=0.3, alpha=0.8, label='Buildings', zorder=1
        )
    else:
        logger.warning("  No building footprints to plot")
    
    # Plot covered roads first (in dark red) so they're visible
    if covered_roads is not None and len(covered_roads) > 0 and roads_gdf is not None:
        covered_indices = covered_roads['road_idx'].values
        covered_segments = segments_gdf[segments_gdf['segment_idx'].isin(covered_indices)]
        if len(covered_segments) > 0:
            covered_segments.plot(ax=ax, color='darkred', linewidth=3, alpha=0.9, 
                                 linestyle=':', zorder=10,
                                 label=f'Covered roads (SVF=0, n={len(covered_segments)})')
    
    # Plot street segments colored by mean SVF with fixed 0-1 range
    # Exclude covered roads from the main plot
    if covered_roads is not None and len(covered_roads) > 0:
        covered_indices = set(covered_roads['road_idx'].values)
        non_covered_segments = segments_gdf[~segments_gdf['segment_idx'].isin(covered_indices)]
    else:
        non_covered_segments = segments_gdf
    
    # Use interpolated points for continuous visualization if available
    # Check if points_gdf has interpolated points (look for is_interpolated column or use all points)
    use_interpolated = points_gdf is not None and 'is_interpolated' in points_gdf.columns
    
    if use_interpolated and len(points_gdf) > 0:
        # Plot interpolated points colored by SVF for continuous gradient visualization
        # Exclude covered roads from main plot
        if covered_roads is not None and len(covered_roads) > 0:
            covered_indices = set(covered_roads['road_idx'].values)
            non_covered_points = points_gdf[~points_gdf['segment_idx'].isin(covered_indices)]
        else:
            non_covered_points = points_gdf
        
        if len(non_covered_points) > 0:
            # Create small line segments from interpolated points for smooth visualization
            # Group by segment and create line segments between consecutive points
            from shapely.geometry import LineString as ShapelyLineString
            
            line_segments = []
            line_svf_values = []
            
            for seg_idx in non_covered_points['segment_idx'].unique():
                seg_points = non_covered_points[non_covered_points['segment_idx'] == seg_idx].sort_values('distance_along')
                if len(seg_points) < 2:
                    continue
                
                for i in range(len(seg_points) - 1):
                    p1 = seg_points.iloc[i].geometry
                    p2 = seg_points.iloc[i+1].geometry
                    line_seg = ShapelyLineString([(p1.x, p1.y), (p2.x, p2.y)])
                    line_segments.append(line_seg)
                    # Use average SVF of the two points for the segment
                    line_svf_values.append((seg_points.iloc[i]['svf'] + seg_points.iloc[i+1]['svf']) / 2.0)
            
            if len(line_segments) > 0:
                line_segments_gdf = gpd.GeoDataFrame(
                    {'svf': line_svf_values},
                    geometry=line_segments,
                    crs=points_gdf.crs
                )
                line_segments_gdf.plot(
                    ax=ax, column='svf', cmap='RdYlGn',
                    vmin=0.0, vmax=1.0, linewidth=2.0, alpha=0.75, legend=True,
                    legend_kwds={'label': 'SVF (0.0-1.0)', 'shrink': 0.8},
                    zorder=2
                )
    elif len(non_covered_segments) > 0:
        # Fallback to segment-based visualization
        non_covered_segments.plot(
            ax=ax, column='svf_mean', cmap='RdYlGn',
            vmin=0.0, vmax=1.0, linewidth=3, legend=True,
            legend_kwds={'label': 'Mean SVF (0.0-1.0)', 'shrink': 0.8},
            label='Street segments', zorder=3
        )
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Street-Level Sky View Factor (SVF)\nColored by Mean SVF per Segment', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if building_footprints is not None and not building_footprints.empty:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved street SVF map to {output_path}")


def create_street_svf_map_minmax(
    segments_gdf: gpd.GeoDataFrame,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None,
    dtm_path: Path = None,
    clip_percentile: float = 95.0,
    show_isolines: bool = False,
):
    """
    Create an additional min-max scaled SVF map.

    The default scaling is robust to upper-tail outliers by clipping SVF at p95.
    Color convention: low SVF (0) is brown, high SVF (~1) is sand.
    """
    logger.info("Creating min-max street SVF map...")

    if 'svf_mean' not in segments_gdf.columns:
        raise ValueError("GeoDataFrame must have 'svf_mean' column")

    plot_segments = segments_gdf.copy()
    svf = plot_segments['svf_mean'].astype(float)
    svf_min = float(svf.min())
    svf_high = float(np.percentile(svf, clip_percentile))
    svf_clipped = svf.clip(lower=svf_min, upper=svf_high)

    if svf_high > svf_min:
        plot_segments['svf_minmax'] = (svf_clipped - svf_min) / (svf_high - svf_min)
    else:
        plot_segments['svf_minmax'] = 0.5

    fig, ax = plt.subplots(figsize=(12, 10))

    # Low SVF -> brown, high SVF -> sand.
    brown_to_sand = LinearSegmentedColormap.from_list(
        'brown_to_sand',
        ['#6E4423', '#9A6A3A', '#C69A5B', '#DFC08A', '#F2E3B3'],
        N=256
    )

    # Optional terrain isolines only for dedicated isoline variants.
    dtm_bounds_aligned = None
    if show_isolines and dtm_path is not None and dtm_path.exists() and HAS_RASTERIO:
        try:
            with rasterio.open(dtm_path) as src:
                dem = src.read(1, masked=True)
                if src.nodata is not None:
                    dem = np.ma.masked_equal(dem, src.nodata)
                transform = src.transform

            rows, cols = np.meshgrid(
                np.arange(dem.shape[0], dtype=float),
                np.arange(dem.shape[1], dtype=float),
                indexing='ij'
            )
            xs = transform.a * cols + transform.b * rows + transform.c
            ys = transform.d * cols + transform.e * rows + transform.f

            dem_arr = np.ma.filled(dem, np.nan)
            seg_bounds = plot_segments.total_bounds
            dtm_bounds = np.array([np.nanmin(xs), np.nanmin(ys), np.nanmax(xs), np.nanmax(ys)])

            # If DTM and streets are in different coordinate spaces, align by center shift.
            def _center(bounds):
                return (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0

            def _overlap(a, b):
                return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

            if not _overlap(seg_bounds, dtm_bounds):
                dx = _center(seg_bounds)[0] - _center(dtm_bounds)[0]
                dy = _center(seg_bounds)[1] - _center(dtm_bounds)[1]
                xs = xs + dx
                ys = ys + dy
                dtm_bounds = dtm_bounds + np.array([dx, dy, dx, dy])
            dtm_bounds_aligned = dtm_bounds

            zmin = float(np.nanmin(dem_arr))
            zmax = float(np.nanmax(dem_arr))
            levels_minor = np.linspace(zmin, zmax, 24)
            levels_major = np.linspace(zmin, zmax, 12)

            ax.contour(
                xs, ys, dem_arr, levels=levels_minor,
                colors='#596b7a', linewidths=0.55,
                linestyles='dotted', alpha=0.70, zorder=1
            )
            ax.contour(
                xs, ys, dem_arr, levels=levels_major,
                colors='#435564', linewidths=0.95,
                linestyles='solid', alpha=0.60, zorder=1
            )
        except Exception as exc:
            logger.warning(f"  Could not draw terrain isolines from DTM: {exc}")

    # Building footprints should already be in the correct coordinate system
    # (aligned in the main script before calling this function)
    # No need for additional alignment here

    if building_footprints is not None and not building_footprints.empty:
        building_footprints.plot(
            ax=ax,
            facecolor='lightgrey',
            edgecolor='black',
            linewidth=0.35,
            alpha=0.48,
            zorder=2
        )

    plot_segments.plot(
        ax=ax,
        column='svf_minmax',
        cmap=brown_to_sand,
        vmin=0,
        vmax=1,
        linewidth=3.0,
        zorder=4
    )

    sm = plt.cm.ScalarMappable(
        cmap=brown_to_sand,
        norm=colors.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)

    norm_ticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    svf_ticks = svf_min + norm_ticks * (svf_high - svf_min)
    cbar.set_ticks(norm_ticks)
    cbar.set_ticklabels([f"{s:.3f}" for s in svf_ticks])
    cbar.set_label(f"SVF (p{int(clip_percentile)} capped min-max mapping)")

    # Set map extent from polygon bbox (building footprints), with segments fallback.
    # This prevents DTM-driven over-zoom into a much larger surrounding area.
    bounds_to_merge = [plot_segments.total_bounds]
    if building_footprints is not None and not building_footprints.empty:
        bounds_to_merge = [building_footprints.total_bounds, plot_segments.total_bounds]

    merged = np.array(bounds_to_merge, dtype=float)
    minx = float(np.min(merged[:, 0]))
    miny = float(np.min(merged[:, 1]))
    maxx = float(np.max(merged[:, 2]))
    maxy = float(np.max(merged[:, 3]))

    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    pad_x = 0.07 * width
    pad_y = 0.07 * height

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_title('Street-Level SVF (Min-Max Scaled)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_axis_off()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved min-max street SVF map to {output_path}")


def create_street_svf_map_minmax_purple(
    segments_gdf: gpd.GeoDataFrame,
    building_footprints: gpd.GeoDataFrame = None,
    output_path: Path = None,
    dtm_path: Path = None,
    clip_percentile: float = 95.0,
    show_isolines: bool = False,
):
    """Backward-compatible wrapper for older calls."""
    return create_street_svf_map_minmax(
        segments_gdf=segments_gdf,
        building_footprints=building_footprints,
        output_path=output_path,
        dtm_path=dtm_path,
        clip_percentile=clip_percentile,
        show_isolines=show_isolines,
    )


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
    
    # Calculate histogram - use weights to get proportions as percentages
    svf_values = points_gdf['svf'].values
    total_points = len(svf_values)
    
    # Create histogram with percentage on y-axis
    counts, bins, patches = ax.hist(svf_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue', 
                                    weights=np.ones_like(svf_values) * (100.0 / total_points))
    
    ax.set_xlabel('SVF Value', fontsize=12)
    ax.set_ylabel('Proportion of street network (%)', fontsize=12)
    ax.set_title('Distribution of Street-Level SVF Values', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add comfort SVF threshold at 0.3.
    comfort_threshold = 0.3
    ax.axvline(
        comfort_threshold,
        color='orange',
        linestyle=':',
        linewidth=2,
        label=f'Comfort threshold (SVF = {comfort_threshold:.1f})'
    )
    
    # Add statistics
    mean_svf = points_gdf['svf'].mean()
    median_svf = points_gdf['svf'].median()
    ax.axvline(mean_svf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_svf:.3f}')
    ax.axvline(median_svf, color='green', linestyle='--', linewidth=2, label=f'Median: {median_svf:.3f}')

    # Add segment-level comfort share as a legend entry.
    pct_segments_below = 100.0 * float((segments_gdf['svf_mean'] < comfort_threshold).mean())
    from matplotlib.lines import Line2D
    comfort_handle = Line2D(
        [0], [0],
        color='none',
        label=f'Segments below comfort threshold: {pct_segments_below:.1f}%'
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(comfort_handle)
    labels.append(comfort_handle.get_label())
    ax.legend(handles, labels)
    
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
    parser.add_argument('--stl', type=str, required=False, help='Path to STL file (auto-resolved if --area provided)')
    parser.add_argument('--roads', type=str, required=False, help='Path to road network shapefile (auto-resolved if --area provided)')
    parser.add_argument('--footprints', type=str, default=None, help='Path to building footprints (optional, for visualization)')
    parser.add_argument('--dtm', type=str, default=None, help='Path to DTM raster (optional, preferred for elevation)')
    parser.add_argument('--spacing', type=float, default=3.0, help='Distance between sample points (meters, default: 3.0)')
    parser.add_argument('--height', type=float, default=1.5, help='Evaluation height above ground (meters, default: 1.5)')
    parser.add_argument('--sky-patches', type=int, default=145, help='Number of sky patches (default: 145)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: outputs/svf_street)')
    parser.add_argument('--area', type=str, default=None, help='Area name (e.g., vidigal_tls, copacabana) for automatic path resolution')
    parser.add_argument('--building-buffer', type=float, default=30.0, help='Buffer distance from buildings to filter streets (meters, default: 30.0)')
    parser.add_argument('--redirect-roads', action='store_true', default=False, help='Redirect roads to avoid building intersections (default: False)')
    parser.add_argument('--set-covered-svf-zero', action='store_true', default=False, help='Set SVF=0 for roads that cannot be redirected (default: False)')
    parser.add_argument('--debug-only', action='store_true', help='Only generate debug plot and exit, do not compute SVF')
    parser.add_argument('--use-gpu', action='store_true', default=None, help='Use GPU acceleration if available (auto-detected by default)')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU computation even if GPU is available')
    
    args = parser.parse_args()
    
    # Determine GPU usage
    use_gpu = None
    if args.no_gpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True
    # else: use_gpu = None (auto-detect)
    
    # Setup paths
    if args.area:
        # Auto-resolve paths from area name
        from src.config import get_area_data_dir
        area_data_dir = get_area_data_dir(args.area)
        
        # Find STL file
        if args.stl:
            stl_path = Path(args.stl)
        else:
            stl_candidates = list(area_data_dir.glob("*.stl"))
            if not stl_candidates:
                logger.error(f"No STL file found in {area_data_dir}")
                sys.exit(1)
            stl_path = stl_candidates[0]
            logger.info(f"Auto-resolved STL file: {stl_path}")
        
        # Find roads file
        if args.roads:
            roads_path = Path(args.roads)
        else:
            # Try common road file patterns
            road_patterns = ["*roads*.shp", "*roads*.gpkg", "*street*.shp", "*street*.gpkg"]
            roads_candidates = []
            for pattern in road_patterns:
                roads_candidates.extend(list(area_data_dir.glob(pattern)))
            if not roads_candidates:
                logger.error(f"No road network file found in {area_data_dir}")
                sys.exit(1)
            roads_path = roads_candidates[0]
            logger.info(f"Auto-resolved roads file: {roads_path}")
        
        # Auto-resolve DTM if not provided
        if not args.dtm:
            dtm_patterns = ["*dtm*.tif", "*dtm*.tiff", "*DTM*.tif", "*DTM*.tiff", "*dem*.tif", "*DEM*.tif"]
            dtm_candidates = []
            for pattern in dtm_patterns:
                dtm_candidates.extend(list(area_data_dir.glob(pattern)))
            if dtm_candidates:
                args.dtm = str(dtm_candidates[0])
                logger.info(f"Auto-resolved DTM file: {args.dtm}")
    else:
        # Require explicit paths if no area specified
        if not args.stl or not args.roads:
            logger.error("Either --area must be provided, or both --stl and --roads must be specified")
            sys.exit(1)
        stl_path = Path(args.stl)
        roads_path = Path(args.roads)
    
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        sys.exit(1)
    
    if not roads_path.exists():
        logger.error(f"Road network file not found: {roads_path}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.area:
        from src.config import get_area_output_dir
        output_dir = get_area_output_dir(args.area) / "svf_street"
    else:
        output_dir = PROJECT_ROOT / "outputs" / "svf_street"
    
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
    
    # Load mesh and terrain (in local coordinates)
    mesh = load_mesh(stl_path)
    terrain = extract_terrain_surface(mesh)
    
    # Get STL bounds (in local coordinates)
    terrain_bounds = terrain.bounds  # (x_min, x_max, y_min, y_max, z_min, z_max)
    stl_bounds = terrain_bounds
    
    # Setup DTM path (required for georeferencing)
    dtm_path = Path(args.dtm) if args.dtm else None
    if not dtm_path or not dtm_path.exists():
        logger.error("DTM is required for STL georeferencing. Please provide --dtm argument or use --area for auto-resolution.")
        sys.exit(1)
    
    logger.info("Georeferencing STL using DTM bounds...")
    from src.data_alignment_utils import georeference_stl_using_dtm
    
    # Calculate translation to transform STL FROM local TO world coordinates (X, Y, Z)
    dx, dy, dz, dtm_bounds = georeference_stl_using_dtm(
        stl_bounds=stl_bounds,
        dtm_path=dtm_path
    )
    
    # Transform STL mesh vertices to world coordinates
    logger.info(f"  Transforming STL mesh to world coordinates...")
    mesh.points[:, 0] += dx  # Transform X coordinates
    mesh.points[:, 1] += dy  # Transform Y coordinates
    mesh.points[:, 2] += dz  # Transform Z coordinates (align with DTM elevation)
    
    # Re-extract terrain after transformation (to ensure it's correct)
    logger.info(f"  Re-extracting terrain surface after transformation...")
    terrain = extract_terrain_surface(mesh)
    
    # Get transformed STL bounds (now in world coordinates)
    transformed_stl_bounds = (
        stl_bounds[0] + dx, stl_bounds[1] + dx,
        stl_bounds[2] + dy, stl_bounds[3] + dy,
        stl_bounds[4] + dz, stl_bounds[5] + dz
    )
    logger.info(f"  STL bounds (world): ({transformed_stl_bounds[0]:.2f}, {transformed_stl_bounds[2]:.2f}) to ({transformed_stl_bounds[1]:.2f}, {transformed_stl_bounds[3]:.2f})")
    
    # Load road network (already in world coordinates, no transformation needed)
    logger.info(f"Loading road network from {roads_path}...")
    roads_gdf = gpd.read_file(roads_path)
    logger.info(f"  Loaded {len(roads_gdf)} road segments")
    logger.info(f"  CRS: {roads_gdf.crs}")
    logger.info(f"  Road bounds: {roads_gdf.total_bounds}")
    
    # Validate alignment: check that STL and roads overlap
    road_bounds = roads_gdf.total_bounds
    overlap_x = min(transformed_stl_bounds[1], road_bounds[2]) - max(transformed_stl_bounds[0], road_bounds[0])
    overlap_y = min(transformed_stl_bounds[3], road_bounds[3]) - max(transformed_stl_bounds[2], road_bounds[1])
    
    if overlap_x < 0 or overlap_y < 0:
        logger.warning(f"  Warning: STL and roads may not overlap after georeferencing")
        logger.warning(f"    STL bounds: ({transformed_stl_bounds[0]:.2f}, {transformed_stl_bounds[2]:.2f}) to ({transformed_stl_bounds[1]:.2f}, {transformed_stl_bounds[3]:.2f})")
        logger.warning(f"    Road bounds: ({road_bounds[0]:.2f}, {road_bounds[1]:.2f}) to ({road_bounds[2]:.2f}, {road_bounds[3]:.2f})")
    else:
        logger.info(f"  Alignment validated: STL and roads overlap ({overlap_x:.2f}m x {overlap_y:.2f}m)")
    
    # Now load building footprints (they will use the same transformation)
    building_footprints = None
    isolated_buildings = None
    area_filtered_buildings = None
    building_buffer_geom = None
    
    # Auto-resolve footprints path if --area is provided
    if args.area and not args.footprints:
        from src.config import get_area_data_dir
        area_data_dir = get_area_data_dir(args.area)
        # Try common building footprint file patterns
        footprint_patterns = ["*buildings*.shp", "*buildings*.gpkg", "*footprints*.shp", "*footprints*.gpkg"]
        footprints_candidates = []
        for pattern in footprint_patterns:
            footprints_candidates.extend(list(area_data_dir.glob(pattern)))
        if footprints_candidates:
            args.footprints = str(footprints_candidates[0])
            logger.info(f"Auto-resolved footprints file: {args.footprints}")
    
    if args.footprints:
        footprints_path = Path(args.footprints)
        if footprints_path.exists():
            # Load buildings directly (they're already in world coordinates, don't transform)
            from src.svf_utils import filter_buildings
            buildings_raw = gpd.read_file(str(footprints_path))
            logger.info(f"  Loaded {len(buildings_raw)} building footprints")
            logger.info(f"  Building CRS: {buildings_raw.crs}")
            logger.info(f"  Building bounds: {buildings_raw.total_bounds}")
            
            # Apply filtering (but don't transform - they're already georeferenced)
            building_footprints, isolated_buildings, area_filtered_buildings = filter_buildings(
                buildings_raw, area=args.area
            )
            
            logger.info(f"  Prepared {len(building_footprints)} building footprints (main cluster)")
            
            # Create buffer geometry for street filtering
            if building_footprints is not None and len(building_footprints) > 0 and args.building_buffer > 0:
                from shapely.ops import unary_union
                # Repair invalid polygons before union (common with large city-wide datasets).
                geom_series = building_footprints.geometry.copy()
                invalid_mask = ~geom_series.is_valid
                if invalid_mask.any():
                    n_invalid = int(invalid_mask.sum())
                    logger.warning(f"  Found {n_invalid} invalid building geometries; attempting repair with buffer(0)")
                    geom_series.loc[invalid_mask] = geom_series.loc[invalid_mask].buffer(0)
                    geom_series = geom_series[~geom_series.is_empty]
                    geom_series = geom_series[geom_series.is_valid]
                    logger.info(f"  Using {len(geom_series)} valid geometries after repair")

                building_union = unary_union(geom_series.values)
                building_buffer_geom = building_union.buffer(args.building_buffer)
                logger.info(f"  Created building buffer zone ({args.building_buffer}m) for street filtering")
            
            # Redirect roads to avoid building intersections (OPTIONAL preprocessing step)
            intersection_info = None
            intersections_after = None
            
            if args.redirect_roads and building_footprints is not None and len(building_footprints) > 0:
                from src.data_alignment_utils import redirect_roads, detect_road_building_intersections
                
                # Detect intersections before redirection
                intersections_before = detect_road_building_intersections(roads_gdf, building_footprints)
                
                if len(intersections_before) > 0:
                    logger.warning(f"  Found {len(intersections_before)} road segments intersecting buildings")
                    logger.info(f"  Redirecting roads to avoid collisions (--redirect-roads enabled)...")
                    
                    # Redirect roads (using parallel offset method)
                    roads_gdf, intersection_info = redirect_roads(
                        roads_gdf,
                        building_footprints,
                        method='parallel_offset',
                        offset_distance=2.0
                    )
                    
                    # Verify redirection success
                    intersections_after = detect_road_building_intersections(roads_gdf, building_footprints)
                    if len(intersections_after) == 0:
                        logger.info(f"  Successfully redirected all {len(intersection_info)} intersecting roads")
                    else:
                        logger.warning(f"  {len(intersections_after)} roads still intersect after redirection")
                        if args.set_covered_svf_zero:
                            logger.info(f"  These roads will be treated as covered (SVF=0, --set-covered-svf-zero enabled)")
                        else:
                            logger.info(f"  These roads will be processed normally (use --set-covered-svf-zero to set SVF=0)")
                else:
                    intersection_info = gpd.GeoDataFrame(columns=['road_idx'], crs=roads_gdf.crs)
                    intersections_after = gpd.GeoDataFrame(columns=['road_idx'], crs=roads_gdf.crs)
                    logger.info(f"  No road-building intersections detected")
            elif building_footprints is not None and len(building_footprints) > 0:
                # Check for intersections but don't redirect (unless flag is set)
                from src.data_alignment_utils import detect_road_building_intersections
                intersections_before = detect_road_building_intersections(roads_gdf, building_footprints)
                if len(intersections_before) > 0:
                    logger.info(f"  Found {len(intersections_before)} road segments intersecting buildings")
                    logger.info(f"  Use --redirect-roads to redirect them, or --set-covered-svf-zero to set SVF=0")
                intersection_info = None
                intersections_after = None
            else:
                intersection_info = None
                intersections_after = None
    else:
        intersection_info = None
        intersections_after = None
    
    # Filter streets to only keep those within building buffer
    # Use a stricter criterion: require at least 50% of the street segment to be within the buffer
    original_roads_count = len(roads_gdf)
    filtered_roads_gdf = roads_gdf.copy()
    if building_buffer_geom is not None and args.building_buffer > 0:
        import numpy as np
        from shapely.geometry import Point
        
        # Check each street segment to see what percentage is within the buffer
        street_mask = np.zeros(len(roads_gdf), dtype=bool)
        min_percentage_within = 0.7  # Require at least 70% of street to be within buffer (more conservative)
        
        logger.info(f"  Checking street segments for buffer inclusion (min {min_percentage_within*100:.0f}% within buffer)...")
        for idx, row in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="  Filtering streets"):
            line = row.geometry
            line_length = line.length
            
            # Sample points along the line to check what percentage is within buffer
            # Sample every 5 meters or at least 10 points
            num_samples = max(10, int(line_length / 5.0))
            sample_distances = np.linspace(0, line_length, num_samples)
            
            points_within = 0
            for dist in sample_distances:
                point = line.interpolate(dist)
                if building_buffer_geom.contains(point):
                    points_within += 1
            
            percentage_within = points_within / num_samples
            if percentage_within >= min_percentage_within:
                street_mask[idx] = True
        
        filtered_roads_gdf = roads_gdf[street_mask].copy()
        
        n_filtered = original_roads_count - len(filtered_roads_gdf)
        if n_filtered > 0:
            logger.info(f"  Filtered out {n_filtered} street segments (<{min_percentage_within*100:.0f}% within {args.building_buffer}m buffer)")
            logger.info(f"  Remaining streets: {len(filtered_roads_gdf)} / {original_roads_count}")
        else:
            logger.info(f"  All {original_roads_count} street segments meet the buffer criterion")
    else:
        logger.info(f"  No building buffer applied - processing all {original_roads_count} street segments")
    
    # Sample points along filtered streets
    dtm_path = Path(args.dtm) if args.dtm else None
    points_gdf = sample_street_points(
        filtered_roads_gdf, args.spacing, dtm_path=dtm_path, terrain=terrain
    )
    
    # Remove points with invalid elevation
    import numpy as np  # Ensure np is available
    def _check_nan_z(p):
        if hasattr(p, 'z'):
            return np.isnan(p.z)
        return True
    valid_mask = ~points_gdf.geometry.apply(_check_nan_z)
    if not valid_mask.all():
        logger.warning(f"  Removed {np.sum(~valid_mask)} points with invalid elevation")
        points_gdf = points_gdf[valid_mask].copy()
    
    # Create 2D debug plot
    debug_plot_path = output_dir / "street_debug.png"
    plot_street_debug(
        roads_gdf, points_gdf, 
        building_footprints=building_footprints,
        isolated_buildings=isolated_buildings,
        intersection_info=intersection_info if 'intersection_info' in locals() else None,
        area_filtered_buildings=area_filtered_buildings,
        output_path=debug_plot_path,
        spacing=args.spacing,
        height=args.height,
        building_buffer=args.building_buffer if args.building_buffer > 0 else None,
        building_buffer_geom=building_buffer_geom,
        filtered_roads_gdf=filtered_roads_gdf,
        original_roads_count=original_roads_count
    )
    
    # Create 3D debug visualization showing STL mesh, buildings, and roads
    debug_3d_path = output_dir / "street_debug_3d.png"
    logger.info(f"Creating 3D debug visualization: {debug_3d_path}")
    create_3d_alignment_debug(
        mesh=mesh,
        terrain=terrain,
        roads_gdf=roads_gdf,
        building_footprints=building_footprints,
        output_path=debug_3d_path
    )
    
    # If debug-only mode, stop here
    if args.debug_only:
        print("\n" + "=" * 60)
        print("DEBUG PLOT GENERATION COMPLETE")
        print(f"Debug plot saved to: {debug_plot_path}")
        print("=" * 60)
        return
    
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
    
    # Compute SVF (with automatic GPU detection)
    svf_values = compute_svf(street_points_3d, sky_patches, mesh, args.height, use_gpu=use_gpu)
    
    # Add SVF values to points GeoDataFrame
    points_gdf = points_gdf.copy()
    points_gdf['svf'] = svf_values
    
    # Set SVF to 0 for roads that still intersect buildings (OPTIONAL, only if flag is set)
    if args.set_covered_svf_zero and intersections_after is not None and len(intersections_after) > 0:
        covered_road_indices = set(intersections_after['road_idx'].values)
        covered_mask = points_gdf['segment_idx'].isin(covered_road_indices)
        n_covered_points = covered_mask.sum()
        if n_covered_points > 0:
            points_gdf.loc[covered_mask, 'svf'] = 0.0
            logger.info(f"  Set SVF=0 for {n_covered_points} points on {len(covered_road_indices)} covered roads (--set-covered-svf-zero enabled)")
    elif intersections_after is not None and len(intersections_after) > 0:
        covered_road_indices = set(intersections_after['road_idx'].values)
        n_covered_points = (points_gdf['segment_idx'].isin(covered_road_indices)).sum()
        if n_covered_points > 0:
            logger.info(f"  {n_covered_points} points on {len(covered_road_indices)} roads still intersect buildings")
            logger.info(f"  Use --set-covered-svf-zero to set their SVF=0")
    
    # Create continuous SVF representation by interpolating along segments
    logger.info("Creating continuous SVF representation...")
    interpolated_points_gdf = interpolate_svf_along_segments(
        points_gdf, filtered_roads_gdf, interpolation_spacing=0.5
    )
    
    # Set SVF to 0 for interpolated points on covered roads (OPTIONAL, only if flag is set)
    if args.set_covered_svf_zero and intersections_after is not None and len(intersections_after) > 0:
        covered_road_indices = set(intersections_after['road_idx'].values)
        covered_interp_mask = interpolated_points_gdf['segment_idx'].isin(covered_road_indices)
        if covered_interp_mask.any():
            interpolated_points_gdf.loc[covered_interp_mask, 'svf'] = 0.0
    
    # Aggregate to segment level using interpolated points for more accurate statistics
    segments_gdf = aggregate_segment_statistics(
        interpolated_points_gdf, interpolated_points_gdf['svf'].values, filtered_roads_gdf
    )
    
    # Set SVF to 0 for segments that are completely covered (OPTIONAL, only if flag is set)
    if args.set_covered_svf_zero and intersections_after is not None and len(intersections_after) > 0:
        covered_road_indices = set(intersections_after['road_idx'].values)
        covered_segment_mask = segments_gdf['segment_idx'].isin(covered_road_indices)
        if covered_segment_mask.any():
            # Set mean SVF to 0 for covered segments
            segments_gdf.loc[covered_segment_mask, 'svf_mean'] = 0.0
            segments_gdf.loc[covered_segment_mask, 'svf_min'] = 0.0
            segments_gdf.loc[covered_segment_mask, 'svf_max'] = 0.0
            segments_gdf.loc[covered_segment_mask, 'svf_std'] = 0.0
            logger.info(f"  Set segment-level SVF=0 for {covered_segment_mask.sum()} covered road segments (--set-covered-svf-zero enabled)")
    
    # All data is already in world coordinates (STL was georeferenced, roads/buildings were never transformed)
    # No back-transformation needed
    points_gdf_world = points_gdf.copy()
    interpolated_points_gdf_world = interpolated_points_gdf.copy()
    segments_gdf_world = segments_gdf.copy()
    building_footprints_world = building_footprints.copy() if building_footprints is not None else None

    # Save results
    logger.info("Saving results...")
    
    # Point-level output
    points_output = output_dir / "street_svf_points.gpkg"
    points_gdf_world.to_file(points_output, driver='GPKG')
    logger.info(f"  Saved point-level results to {points_output}")
    
    # Save interpolated points (continuous representation)
    interpolated_output = output_dir / "street_svf_points_interpolated.gpkg"
    interpolated_points_gdf_world.to_file(interpolated_output, driver='GPKG')
    logger.info(f"  Saved interpolated point-level results to {interpolated_output}")
    
    # Segment-level output
    segments_output = output_dir / "street_svf_segments.gpkg"
    segments_gdf_world.to_file(segments_output, driver='GPKG')
    logger.info(f"  Saved segment-level results to {segments_output}")
    
    # CSV summary
    csv_output = output_dir / "street_svf_statistics.csv"
    segments_gdf_world.drop(columns=['geometry']).to_csv(csv_output, index=False)
    logger.info(f"  Saved statistics to {csv_output}")
    
    # Visualizations
    logger.info("Generating visualizations...")
    
    # Street SVF map (use world coordinates for buildings)
    map_path = output_dir / "street_svf_map.png"
    # Translate covered roads info to world coordinates if needed (only if flag is set)
    covered_roads_world = None
    if args.set_covered_svf_zero and intersections_after is not None and len(intersections_after) > 0:
        covered_roads_world = intersections_after.copy()
    
    # Debug: Check if buildings are available
    if building_footprints_world is not None:
        logger.info(f"  Building footprints for visualization: {len(building_footprints_world)} buildings")
    else:
        logger.warning("  No building footprints available for visualization")
    
    create_street_svf_map(
        segments_gdf_world, interpolated_points_gdf_world, building_footprints_world, map_path,
        covered_roads=covered_roads_world,
        roads_gdf=filtered_roads_gdf if 'filtered_roads_gdf' in locals() else None
    )

    # Additional robust min-max map (brown->sand, kept as a separate output).
    minmax_map_path = output_dir / "street_svf_map_minmax.png"
    create_street_svf_map_minmax(
        segments_gdf_world,
        building_footprints_world,
        minmax_map_path,
        dtm_path=dtm_path,
        clip_percentile=95.0,
        show_isolines=False,
    )
    
    # Statistics plots
    create_statistics_plots(segments_gdf_world, points_gdf_world, output_dir)
    
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

