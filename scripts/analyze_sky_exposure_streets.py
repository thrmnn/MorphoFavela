#!/usr/bin/env python3
"""
Compute Sky Exposure Plane Exceedance along street centerlines.

This script computes sky exposure plane exceedance at sample points along street
centerlines, implementing Brazilian building code rulesets (Rio de Janeiro and São Paulo).
This provides street-level analysis of building code compliance.

Usage:
    python scripts/analyze_sky_exposure_streets.py \
        --stl data/vidigal/raw/full_scan.stl \
        --roads data/vidigal/raw/roads_vidigal.shp \
        --footprints data/vidigal/raw/vidigal_buildings.shp \
        --ruleset rio
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
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_utils import load_mesh, extract_terrain_surface, load_building_footprints
from src.metrics import normalize_height_columns
from src.config import MIN_BUILDING_AREA, MAX_FILTER_AREA, is_formal_area
from scripts.compute_svf_streets import sample_points_along_line, extract_elevation_from_mesh
from scripts.analyze_sky_exposure import extract_building_meshes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rio_setback(building_height: float) -> float:
    """
    Calculate setback distance for Rio de Janeiro ruleset.
    
    Rio: setback = max(2.5m, H/5)
    
    Args:
        building_height: Building height in meters
        
    Returns:
        Setback distance in meters
    """
    calculated_setback = building_height / 5.0
    return max(2.5, calculated_setback)


def calculate_saopaulo_setback(building_height: float) -> float:
    """
    Calculate setback distance for São Paulo ruleset.
    
    São Paulo: A = (H - 6) / 10, setback = max(3.0m, A)
    
    Args:
        building_height: Building height in meters
        
    Returns:
        Setback distance in meters (or None if building <= 10m)
    """
    if building_height <= 10.0:
        return None  # No envelope restriction for buildings <= 10m
    A = (building_height - 6.0) / 10.0
    return max(3.0, A)


def calculate_rio_envelope_height(
    point: Point,
    footprint: Polygon,
    base_height: float,
    building_height: float,
    origin_z: float = 0.0
) -> float:
    """
    Calculate Rio de Janeiro ruleset envelope height at a street point.
    
    Rio: 1/5 ratio - building recedes 1m for every 5m of height
    Envelope: base_height + (distance_to_setback × 5)
    
    Args:
        point: Street point geometry (x, y)
        footprint: Building footprint polygon
        base_height: Base height (first ventilated floor level, absolute Z coordinate)
        building_height: Total building height (top_height - base_height)
        origin_z: Z coordinate of footprint base (should match base_height for absolute coords)
        
    Returns:
        Envelope Z coordinate at this point (absolute)
    """
    # Calculate setback
    setback_dist = calculate_rio_setback(building_height)
    
    # Create setback polygon
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
    
    # If point is within setback area
    if dist_to_edge < setback_dist:
        # Within setback: allowed height = base_height
        envelope_height = base_height
    else:
        # Calculate distance from setback boundary
        if hasattr(setback_poly, 'exterior'):
            setback_boundary = setback_poly.exterior
        else:
            setback_boundary = setback_poly
        
        dist_to_setback = setback_boundary.distance(point)
        
        # Rio: envelope = base_height + (distance × 5)
        # base_height is already absolute Z, so envelope_height is also absolute
        envelope_height = base_height + (dist_to_setback * 5.0)
    
    # base_height is already absolute, so return envelope_height directly
    # (origin_z is only used if base_height is relative, but in our case it's absolute)
    return envelope_height


def calculate_saopaulo_envelope_height(
    point: Point,
    footprint: Polygon,
    base_height: float,
    building_height: float,
    origin_z: float = 0.0
) -> float:
    """
    Calculate São Paulo ruleset envelope height at a street point.
    
    São Paulo: 
    - If H ≤ 10m: envelope = 10m (no recession)
    - If H > 10m: 1/10 ratio, envelope = 10 + (distance × 10)
    
    Args:
        point: Street point geometry (x, y)
        footprint: Building footprint polygon
        base_height: Base height (terrain level)
        building_height: Total building height
        origin_z: Z coordinate of footprint base
        
    Returns:
        Envelope Z coordinate at this point
    """
    # Check threshold
    if building_height <= 10.0:
        # No envelope restriction, allow up to 10m
        return origin_z + 10.0
    
    # Calculate setback
    setback_dist = calculate_saopaulo_setback(building_height)
    
    if setback_dist is None:
        return origin_z + 10.0
    
    # Create setback polygon
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
    
    # If point is within setback area
    if dist_to_edge < setback_dist:
        envelope_height = 10.0
    else:
        # Calculate distance from setback boundary
        if hasattr(setback_poly, 'exterior'):
            setback_boundary = setback_poly.exterior
        else:
            setback_boundary = setback_poly
        
        dist_to_setback = setback_boundary.distance(point)
        
        # São Paulo: envelope = 10 + (distance × 10)
        envelope_height = 10.0 + (dist_to_setback * 10.0)
    
    return origin_z + envelope_height


def extract_building_height_at_point(
    street_point: Point,
    building_footprint: Polygon,
    building_mesh: pv.PolyData,
    terrain: pv.PolyData
) -> float:
    """
    Extract actual building height at a street point location.
    
    For street points, we want to find the building height that affects the sky exposure
    at that location. Since street points are outside buildings, we:
    1. Find the closest point on the building footprint to the street point
    2. Extract the building height at that facade location
    
    Args:
        street_point: Street point (x, y)
        building_footprint: Building footprint polygon
        building_mesh: Building mesh from STL
        terrain: Terrain mesh for elevation reference
        
    Returns:
        Building height (absolute Z coordinate) at the closest facade point, or NaN if no building
    """
    # Get building mesh points
    if building_mesh.n_points == 0:
        return np.nan
    
    building_points = building_mesh.points
    
    # Find the closest point on the building footprint boundary to the street point
    # This represents the facade that would affect sky exposure from the street
    # Handle both Polygon and MultiPolygon geometries
    if hasattr(building_footprint, 'exterior'):
        # Single Polygon
        closest_point_on_footprint = building_footprint.exterior.interpolate(
            building_footprint.exterior.project(street_point)
        )
    else:
        # MultiPolygon - find closest polygon first
        from shapely.ops import nearest_points
        closest_poly = min(
            building_footprint.geoms if hasattr(building_footprint, 'geoms') else [building_footprint],
            key=lambda p: p.distance(street_point)
        )
        closest_point_on_footprint = closest_poly.exterior.interpolate(
            closest_poly.exterior.project(street_point)
        )
    
    # Find building mesh points near this closest facade point
    # Use a smaller search radius (e.g., 5-10m) to get points on the facade
    search_radius = 10.0  # meters
    
    distances_to_facade = np.sqrt(
        (building_points[:, 0] - closest_point_on_footprint.x)**2 + 
        (building_points[:, 1] - closest_point_on_footprint.y)**2
    )
    
    facade_mask = distances_to_facade <= search_radius
    
    if not facade_mask.any():
        # No facade points found, try a larger search or use entire building
        # Fallback: find points within building footprint
        from shapely.geometry import Point as ShapelyPoint
        within_footprint = [
            building_footprint.contains(ShapelyPoint(p[0], p[1])) 
            for p in building_points
        ]
        facade_mask = np.array(within_footprint)
    
    if not facade_mask.any():
        # Still no points, use all building points but prioritize closer ones
        distances_to_street = np.sqrt(
            (building_points[:, 0] - street_point.x)**2 + 
            (building_points[:, 1] - street_point.y)**2
        )
        # Use points within 20m of street point
        facade_mask = distances_to_street <= 20.0
    
    if not facade_mask.any():
        return np.nan
    
    # Get maximum Z coordinate of facade points
    # This represents the building height at the facade facing the street
    facade_points = building_points[facade_mask]
    max_z = np.max(facade_points[:, 2])
    
    return float(max_z)


def find_nearby_buildings(
    street_point: Point,
    buildings_gdf: gpd.GeoDataFrame,
    search_radius: float = 75.0
) -> list:
    """
    Find buildings within search radius of a street point.
    
    Args:
        street_point: Street point geometry
        buildings_gdf: GeoDataFrame with building footprints
        search_radius: Search radius in meters
        
    Returns:
        List of building indices within search radius
    """
    # Create buffer around street point
    buffer_zone = street_point.buffer(search_radius)
    
    # Find buildings that intersect or are within buffer
    nearby = buildings_gdf[buildings_gdf.geometry.intersects(buffer_zone)]
    
    return nearby.index.tolist()


def compute_street_exceedance(
    street_points_3d: np.ndarray,
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    building_meshes: dict,
    terrain: pv.PolyData,
    ruleset: str = 'rio',
    search_radius: float = 75.0
) -> tuple:
    """
    Compute sky exposure plane exceedance for street points.
    
    Args:
        street_points_3d: Array of (x, y, z) street point coordinates
        roads_gdf: GeoDataFrame with road segments
        buildings_gdf: GeoDataFrame with building footprints (must have base_height, top_height)
        building_meshes: Dictionary of building meshes extracted from STL
        terrain: Terrain mesh
        ruleset: 'rio' or 'saopaulo'
        search_radius: Radius for finding nearby buildings (meters)
        
    Returns:
        Tuple of (exceedance_values, metadata_list)
        - exceedance_values: Array of maximum exceedance per point (meters)
        - metadata_list: List of dicts with point metadata
    """
    logger.info(f"Computing street exceedance using {ruleset.upper()} ruleset...")
    logger.info(f"  Search radius: {search_radius}m")
    logger.info(f"  Street points: {len(street_points_3d)}")
    
    # Validate building data
    required_cols = ['base_height', 'top_height']
    missing = [col for col in required_cols if col not in buildings_gdf.columns]
    if missing:
        raise ValueError(f"Building footprints must have columns: {missing}")
    
    # Calculate building heights
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['building_height'] = buildings_gdf['top_height'] - buildings_gdf['base_height']
    
    # Build spatial index for efficient queries
    logger.info("Building spatial index for building queries...")
    building_tree = STRtree(buildings_gdf.geometry.values)
    
    exceedance_values = []
    metadata_list = []
    
    # Process each street point
    pbar = tqdm(total=len(street_points_3d), desc="Computing exceedance", unit="points")
    
    for i, (x, y, z) in enumerate(street_points_3d):
        street_point = Point(x, y)
        
        # Find nearby buildings
        nearby_indices = find_nearby_buildings(street_point, buildings_gdf, search_radius)
        
        if len(nearby_indices) == 0:
            # No buildings nearby, no exceedance
            exceedance_values.append(0.0)
            metadata_list.append({
                'n_buildings': 0,
                'max_exceedance': 0.0,
                'max_envelope_height': np.nan,
                'max_actual_height': np.nan
            })
            pbar.update(1)
            continue
        
        # Sort buildings by distance from street point (closest first)
        # This ensures we check the front-most buildings first
        building_distances = []
        for building_idx in nearby_indices:
            if building_idx not in building_meshes:
                continue
            building_row = buildings_gdf.loc[building_idx]
            footprint = building_row.geometry
            # Calculate distance from street point to building footprint
            distance = footprint.distance(street_point)
            building_distances.append((building_idx, distance))
        
        # Sort by distance (closest first)
        building_distances.sort(key=lambda x: x[1])
        
        # Find the first (front-most) building that intersects the sky exposure plane
        # Buildings behind it are occluded and do not contribute unless the front building is lower than the plane
        max_exceedance = 0.0
        max_envelope = np.nan
        max_actual = np.nan
        front_building_found = False
        
        for building_idx, distance in building_distances:
            if building_idx not in building_meshes:
                continue
            
            building_row = buildings_gdf.loc[building_idx]
            
            footprint = building_row.geometry
            base_height = float(building_row['base_height'])
            building_height = float(building_row['building_height'])
            top_height = float(building_row['top_height'])  # Use attribute directly
            
            # Calculate envelope height based on ruleset
            # base_height is already in absolute Z coordinates, so pass it directly
            if ruleset.lower() == 'rio':
                envelope_height = calculate_rio_envelope_height(
                    street_point, footprint, base_height, building_height, origin_z=base_height
                )
            elif ruleset.lower() == 'saopaulo':
                envelope_height = calculate_saopaulo_envelope_height(
                    street_point, footprint, base_height, building_height, origin_z=base_height
                )
            else:
                raise ValueError(f"Unknown ruleset: {ruleset}")
            
            # Use top_height from building attributes directly (not mesh extraction)
            # This is more reliable and consistent with the envelope calculation
            actual_height = top_height
            
            # Check if this building intersects the sky exposure plane
            # A building intersects the plane if its actual height exceeds the envelope
            if actual_height > envelope_height:
                # This is the front-most building that intersects the plane
                # Calculate exceedance and stop (buildings behind are occluded)
                exceedance = actual_height - envelope_height
                max_exceedance = exceedance
                max_envelope = envelope_height
                max_actual = actual_height
                front_building_found = True
                break
            # If actual_height <= envelope_height, the building doesn't intersect the plane
            # Continue to check next building (it might be behind but taller, intersecting the plane)
        
        exceedance_values.append(max_exceedance)
        metadata_list.append({
            'n_buildings': len(nearby_indices),
            'n_buildings_checked': len(building_distances),
            'front_building_found': front_building_found,
            'max_exceedance': max_exceedance,
            'max_envelope_height': max_envelope,
            'max_actual_height': max_actual
        })
        
        # Update progress
        if len(exceedance_values) > 0 and len(exceedance_values) % 50 == 0:
            current_mean = np.mean([e for e in exceedance_values if e > 0] or [0])
            pbar.set_postfix({
                'mean_exceedance': f'{current_mean:.2f}m',
                'current': f'{max_exceedance:.2f}m'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    return np.array(exceedance_values), metadata_list


def sample_street_points(
    roads_gdf: gpd.GeoDataFrame,
    spacing: float,
    terrain: pv.PolyData
) -> gpd.GeoDataFrame:
    """
    Generate sample points along all street centerlines.
    
    Args:
        roads_gdf: GeoDataFrame with LineString geometries (already transformed)
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
            z = extract_elevation_from_mesh(point, terrain)
            
            if np.isnan(z):
                continue
            
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
    exceedance_values: np.ndarray,
    roads_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Aggregate exceedance statistics per street segment.
    
    Args:
        points_gdf: GeoDataFrame with street sample points
        exceedance_values: Array of exceedance values for each point
        roads_gdf: Original roads GeoDataFrame
        
    Returns:
        GeoDataFrame with segment-level statistics
    """
    logger.info("Aggregating segment-level statistics...")
    
    # Add exceedance values to points
    points_gdf = points_gdf.copy()
    points_gdf['exceedance'] = exceedance_values
    
    # Group by segment
    segment_stats = []
    
    for idx, row in roads_gdf.iterrows():
        segment_points = points_gdf[points_gdf['segment_idx'] == idx]
        
        if len(segment_points) == 0:
            continue
        
        # Calculate statistics
        exceedance_array = segment_points['exceedance'].values
        points_with_exceedance = (exceedance_array > 0).sum()
        
        stats = {
            'segment_idx': idx,
            'street_name': row.get('nome', row.get('tipo_logra', f'Street_{idx}')),
            'street_type': row.get('tipo_logra', 'Unknown'),
            'hierarchy': row.get('hierarquia', None),
            'length': row.geometry.length,
            'n_points': len(segment_points),
            'mean_exceedance': float(np.mean(exceedance_array)),
            'max_exceedance': float(np.max(exceedance_array)),
            'min_exceedance': float(np.min(exceedance_array)),
            'std_exceedance': float(np.std(exceedance_array)),
            'points_exceeding': points_with_exceedance,
            'exceedance_ratio': float(points_with_exceedance / len(segment_points) * 100.0)
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


def create_street_exceedance_map(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame = None,
    building_footprints: gpd.GeoDataFrame = None,
    ruleset: str = 'rio',
    output_path: Path = None,
    selected_points_gdf: gpd.GeoDataFrame = None
):
    """
    Create a map showing streets colored by exceedance values.
    
    Args:
        segments_gdf: GeoDataFrame with segment-level exceedance statistics
        points_gdf: Optional GeoDataFrame with point-level exceedance
        building_footprints: Optional building footprints for context
        ruleset: Ruleset name for title
        output_path: Path to save the map
        selected_points_gdf: Optional GeoDataFrame with selected section points to highlight
    """
    logger.info("Creating street exceedance map...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot building footprints as background (if available)
    if building_footprints is not None:
        building_footprints.plot(
            ax=ax, facecolor='lightgrey', edgecolor='black', 
            linewidth=0.5, alpha=0.5, label='Buildings'
        )
    
    # Determine color scale based on exceedance values
    max_exceed = segments_gdf['mean_exceedance'].max()
    vmax = max(10.0, max_exceed * 1.1)  # At least 10m range, or 10% above max
    
    # Plot street segments colored by mean exceedance
    segments_gdf.plot(
        ax=ax, column='mean_exceedance', cmap='Reds',
        vmin=0, vmax=vmax, linewidth=3, legend=True,
        legend_kwds={'label': 'Mean Exceedance (m)', 'shrink': 0.8},
        label='Street segments'
    )
    
    # Optionally plot points (for detailed view, if not too many)
    if points_gdf is not None and len(points_gdf) < 1000:
        points_gdf.plot(
            ax=ax, column='exceedance', cmap='Reds',
            vmin=0, vmax=vmax, markersize=2, alpha=0.6, zorder=10
        )
    
    # Highlight selected section points
    if selected_points_gdf is not None:
        colors = {'high': 'red', 'mean': 'orange', 'low': 'green'}
        markers = {'high': '^', 'mean': 's', 'low': 'o'}
        labels_map = {'high': 'High Exceedance', 'mean': 'Mean Exceedance', 'low': 'Low Exceedance'}
        
        for label in ['high', 'mean', 'low']:
            point_subset = selected_points_gdf[selected_points_gdf['label'] == label]
            if len(point_subset) > 0:
                point_subset.plot(
                    ax=ax, color=colors[label], marker=markers[label],
                    markersize=200, edgecolor='black', linewidth=2,
                    zorder=20, label=labels_map[label]
                )
    
    ruleset_name = {'rio': 'Rio de Janeiro', 'saopaulo': 'São Paulo'}.get(ruleset.lower(), ruleset)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Street-Level Sky Exposure Plane Exceedance\n{ruleset_name} Ruleset', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved street exceedance map to {output_path}")


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


def compute_building_exceedance(
    building_idx: int,
    building_mesh: pv.PolyData,
    footprint: Polygon,
    base_height: float,
    top_height: float,
    ruleset: str
) -> dict:
    """
    Compute exceedance for a building against sky exposure plane using Rio/São Paulo ruleset.
    
    Args:
        building_idx: Building index
        building_mesh: Building mesh from STL
        footprint: Building footprint polygon
        base_height: Base height (absolute Z coordinate)
        top_height: Top height (absolute Z coordinate)
        ruleset: Ruleset name ('rio' or 'saopaulo')
        
    Returns:
        Dictionary with exceedance metrics
    """
    # Compute total building volume
    total_volume = compute_building_volume(building_mesh)
    
    # Get building bounds
    bounds = building_mesh.bounds
    z_min = bounds[4]
    z_max = bounds[5]
    
    # Building height attribute (relative height)
    building_height = top_height - base_height
    
    # Exceedance calculation:
    # For each point in the building mesh, check if it's above the envelope
    building_points = building_mesh.points
    exceedance_points = []
    max_exceedance_height = 0.0
    
    for point_coords in building_points:
        x, y, z = point_coords
        point_geom = Point(x, y)
        
        # Calculate envelope height at this (x,y) location using ruleset-specific function
        if ruleset.lower() == 'rio':
            envelope_z = calculate_rio_envelope_height(
                point_geom, footprint, base_height, building_height, origin_z=base_height
            )
        elif ruleset.lower() == 'saopaulo':
            envelope_z = calculate_saopaulo_envelope_height(
                point_geom, footprint, base_height, building_height, origin_z=base_height
            )
        else:
            envelope_z = z_max  # No restriction if unknown ruleset
        
        # Check if point exceeds envelope
        if z > envelope_z:
            exceedance = z - envelope_z
            exceedance_points.append(exceedance)
            max_exceedance_height = max(max_exceedance_height, exceedance)
    
    # Calculate exceedance ratio (percentage of points exceeding envelope)
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


def create_building_exceedance_map(
    footprints_gdf: gpd.GeoDataFrame,
    exceedance_results: dict,
    ruleset: str,
    output_path: Path
) -> None:
    """
    Create a plan view map showing buildings colored by exceedance ratio.
    
    Args:
        footprints_gdf: Building footprints GeoDataFrame
        exceedance_results: Dictionary mapping building index to exceedance metrics
        ruleset: Ruleset name
        output_path: Path to save the map
    """
    logger.info("Creating building-level exceedance map...")
    
    # Add exceedance data to footprints
    gdf = footprints_gdf.copy()
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
    ruleset_name = {'rio': 'Rio de Janeiro', 'saopaulo': 'São Paulo'}.get(ruleset.lower(), ruleset)
    ax.set_title(f'Sky Exposure Plane Exceedance Map ({ruleset_name} Ruleset)\nPercentage of Volume Exceeding Envelope', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Saved building exceedance map to {output_path}")


def create_statistics_plots(
    segments_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    ruleset: str,
    output_dir: Path
):
    """
    Create statistical distribution plots.
    
    Args:
        segments_gdf: GeoDataFrame with segment-level statistics
        points_gdf: GeoDataFrame with point-level exceedance
        ruleset: Ruleset name
        output_dir: Output directory
    """
    logger.info("Creating statistics plots...")
    
    # Distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    exceedance_data = points_gdf['exceedance'].values
    exceedance_data = exceedance_data[exceedance_data > 0]  # Only show violations
    
    if len(exceedance_data) > 0:
        ax.hist(exceedance_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Exceedance (meters)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        ruleset_name = {'rio': 'Rio de Janeiro', 'saopaulo': 'São Paulo'}.get(ruleset.lower(), ruleset)
        ax.set_title(f'Distribution of Street-Level Exceedance Values\n{ruleset_name} Ruleset', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(10, exceedance_data.max() * 1.1))
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_exceed = np.mean(exceedance_data)
        median_exceed = np.median(exceedance_data)
        ax.axvline(mean_exceed, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_exceed:.2f}m')
        ax.axvline(median_exceed, color='green', linestyle='--', linewidth=2, label=f'Median: {median_exceed:.2f}m')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No exceedances found', ha='center', va='center', fontsize=14)
        ax.set_xlabel('Exceedance (meters)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    hist_path = output_dir / f"street_exceedance_distribution_{ruleset}.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved distribution plot to {hist_path}")


def create_street_section_views(
    street_points_gdf: gpd.GeoDataFrame,
    mesh: pv.PolyData,
    buildings_gdf: gpd.GeoDataFrame,
    building_meshes: dict,
    terrain: pv.PolyData,
    ruleset: str,
    output_dir: Path,
    section_length: float = 100.0
):
    """
    Create vertical section views at selected street points.
    
    Selects 3 points: high exceedance, mean exceedance, and low exceedance.
    Creates cross-sections showing actual building heights vs envelope heights.
    
    Args:
        street_points_gdf: GeoDataFrame with street sample points and exceedance values
        mesh: Full scene mesh
        buildings_gdf: GeoDataFrame with building footprints
        building_meshes: Dictionary of building meshes
        terrain: Terrain mesh
        ruleset: Ruleset name ('rio' or 'saopaulo')
        output_dir: Output directory
        section_length: Length of section view in meters (default: 100m)
    """
    logger.info("Creating street section views...")
    
    # Filter points with valid exceedance
    valid_points = street_points_gdf[street_points_gdf['exceedance'].notna()].copy()
    
    if len(valid_points) == 0:
        logger.warning("  No valid exceedance data for section views")
        return
    
    exceedance_values = valid_points['exceedance'].values
    
    # Select 3 representative points
    # 1. High exceedance (near maximum)
    high_idx = np.argmax(exceedance_values)
    high_point = valid_points.iloc[high_idx]
    
    # 2. Mean exceedance (closest to mean)
    mean_exceedance = np.mean(exceedance_values[exceedance_values > 0]) if np.any(exceedance_values > 0) else np.mean(exceedance_values)
    mean_idx = np.argmin(np.abs(exceedance_values - mean_exceedance))
    mean_point = valid_points.iloc[mean_idx]
    
    # 3. Low exceedance (near minimum, but preferably > 0 if possible)
    if np.any(exceedance_values > 0):
        low_exceedance_points = valid_points[valid_points['exceedance'] > 0]
        if len(low_exceedance_points) > 0:
            low_idx = np.argmin(low_exceedance_points['exceedance'].values)
            low_point = low_exceedance_points.iloc[low_idx]
        else:
            low_idx = np.argmin(exceedance_values)
            low_point = valid_points.iloc[low_idx]
    else:
        low_idx = np.argmin(exceedance_values)
        low_point = valid_points.iloc[low_idx]
    
    selected_points = [
        (high_point, 'high', high_point['exceedance']),
        (mean_point, 'mean', mean_point['exceedance']),
        (low_point, 'low', low_point['exceedance'])
    ]
    
    logger.info(f"  Selected points: High={high_point['exceedance']:.2f}m, "
                f"Mean={mean_point['exceedance']:.2f}m, Low={low_point['exceedance']:.2f}m")
    
    # Create section views for each point
    for point_row, label, exceedance_value in selected_points:
        street_point = point_row.geometry
        
        # Determine section direction (perpendicular to street)
        # For simplicity, use cardinal directions (North-South or East-West)
        # Find closest street segment to determine direction
        segment_idx = point_row['segment_idx']
        
        # Get street segment from roads (need to find it)
        # For now, use East-West section (we'll improve this later)
        section_angle = 0  # East-West (0 degrees)
        
        # Create section line through street point
        half_length = section_length / 2
        section_start = Point(street_point.x - half_length, street_point.y)
        section_end = Point(street_point.x + half_length, street_point.y)
        
        # Sample points along section
        n_samples = 200
        section_x = np.linspace(section_start.x, section_end.x, n_samples)
        section_y = np.full_like(section_x, street_point.y)
        
        # Get terrain elevation along section
        terrain_z = []
        building_heights = []  # Tallest building at each point (for visualization)
        front_building_heights = []  # Front-most building that intersects plane (for exceedance)
        envelope_heights_section = []
        
        for x, y in zip(section_x, section_y):
            point_2d = Point(x, y)
            
            # Extract terrain elevation
            z_terrain = extract_elevation_from_mesh(point_2d, terrain)
            if np.isnan(z_terrain):
                z_terrain = 0
            terrain_z.append(z_terrain)
            
            # Find buildings near this section point
            nearby_buildings = find_nearby_buildings(point_2d, buildings_gdf, search_radius=50.0)
            
            # Extract actual building height at this (x,y) location
            # Use building footprints and height attributes directly (more reliable than mesh extraction)
            max_building_height = z_terrain  # Start with terrain height
            
            for building_idx in nearby_buildings:
                building_row = buildings_gdf.loc[building_idx]
                footprint = building_row.geometry
                
                # Check if point is within this building's footprint
                if footprint.contains(point_2d) or footprint.touches(point_2d):
                    # Point is within building - use the building's top_height attribute
                    top_height = float(building_row['top_height'])
                    # top_height is already in absolute Z coordinates
                    max_building_height = max(max_building_height, top_height)
            
            # Calculate envelope height at this point
            # Apply occlusion: only consider front-most building that intersects the plane
            envelope_height_at_point = z_terrain  # Default: no restriction beyond terrain
            front_building_height = z_terrain  # Height of front-most building that intersects (for exceedance)
            
            # Check if there's actually a building at this point
            if max_building_height > z_terrain + 1.0:
                # Sort buildings by distance from this point (closest first) for occlusion logic
                building_distances = []
                for building_idx in nearby_buildings:
                    if building_idx not in building_meshes:
                        continue
                    building_row = buildings_gdf.loc[building_idx]
                    footprint = building_row.geometry
                    distance = footprint.distance(point_2d)
                    building_distances.append((building_idx, distance))
                
                building_distances.sort(key=lambda x: x[1])
                
                # Find the first (front-most) building that intersects the sky exposure plane
                for building_idx, distance in building_distances:
                    building_row = buildings_gdf.loc[building_idx]
                    footprint = building_row.geometry
                    base_height = float(building_row['base_height'])
                    top_height = float(building_row['top_height'])
                    building_height_attr = top_height - base_height
                    
                    # Calculate envelope height for this building at this point
                    if ruleset.lower() == 'rio':
                        envelope_height = calculate_rio_envelope_height(
                            point_2d, footprint, base_height, building_height_attr, origin_z=base_height
                        )
                    elif ruleset.lower() == 'saopaulo':
                        envelope_height = calculate_saopaulo_envelope_height(
                            point_2d, footprint, base_height, building_height_attr, origin_z=base_height
                        )
                    else:
                        envelope_height = z_terrain
                    
                    # Use top_height directly from building attributes (consistent with point exceedance)
                    actual_height = top_height
                    
                    # Check if this building intersects the sky exposure plane
                    if actual_height > envelope_height:
                        # This is the front-most building that intersects the plane
                        envelope_height_at_point = envelope_height
                        front_building_height = actual_height  # Store this building's height for exceedance
                        break
            
            # Validate envelope height makes sense
            if np.isnan(envelope_height_at_point):
                envelope_height_at_point = z_terrain
            
            building_heights.append(max_building_height)  # Tallest building (for visualization)
            front_building_heights.append(front_building_height)  # Front-most building that intersects (for exceedance)
            envelope_heights_section.append(envelope_height_at_point)
        
        # Calculate distances along section (relative to street point)
        distances = section_x - street_point.x
        
        # Create section plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convert to numpy arrays for easier manipulation
        terrain_z_array = np.array(terrain_z)
        building_heights_array = np.array(building_heights)  # Tallest building (for visualization)
        front_building_heights_array = np.array(front_building_heights)  # Front-most building that intersects (for exceedance)
        envelope_heights_array = np.array(envelope_heights_section)
        distances_array = np.array(distances)
        
        # Find valid data (non-zero heights)
        valid_mask = building_heights_array > terrain_z_array + 0.5  # Buildings at least 0.5m above terrain
        
        # Plot terrain base (from minimum terrain elevation)
        min_terrain = np.min(terrain_z_array)
        max_terrain = np.max(terrain_z_array)
        terrain_range = max_terrain - min_terrain
        terrain_base = min_terrain - max(5.0, terrain_range * 0.1)  # Base slightly below minimum terrain
        
        # Fill terrain area
        ax.fill_between(distances_array, terrain_base, terrain_z_array, 
                       color='lightgreen', alpha=0.5, label='Terrain')
        ax.plot(distances_array, terrain_z_array, 'g-', linewidth=2, label='Ground level')
        
        # Plot building profiles (filled area from terrain to building top)
        if np.any(valid_mask):
            # Fill building areas
            building_mask = building_heights_array > terrain_z_array
            ax.fill_between(
                distances_array[building_mask],
                terrain_z_array[building_mask],
                building_heights_array[building_mask],
                color='darkgray', alpha=0.6, label='Actual Buildings'
            )
            
            # Plot building outlines
            ax.plot(distances_array, building_heights_array, 'k-', linewidth=1.5, alpha=0.8)
        
        # Plot envelope heights
        ax.plot(distances_array, envelope_heights_array, 'r--', linewidth=2.5, 
               label='Envelope (Allowed)', alpha=0.9)
        
        # Highlight exceedance areas (where front-most building > envelope)
        # Use front_building_heights_array for exceedance calculation (not building_heights_array)
        # IMPORTANT: Only show exceedance where there's actually a building visible at that point
        # This prevents showing exceedance in areas where no building is drawn
        has_building_at_point = building_heights_array > terrain_z_array + 0.5
        exceedance_mask = has_building_at_point & (front_building_heights_array > envelope_heights_array)
        if np.any(exceedance_mask):
            # Fill exceedance areas with red
            # Use the MINIMUM of front_building_heights and building_heights to ensure
            # the red fill never extends beyond the visible building profile
            exceedance_top = np.minimum(front_building_heights_array[exceedance_mask], 
                                        building_heights_array[exceedance_mask])
            ax.fill_between(
                distances_array[exceedance_mask],
                envelope_heights_array[exceedance_mask],
                exceedance_top,
                color='red', alpha=0.4, label='Exceedance Area', zorder=5
            )
            
            # Calculate exceedance metrics
            # Use the POINT EXCEEDANCE value (exceedance_value) for max height
            # This is the correct value from street-level analysis at this exact point
            max_exceedance_height = exceedance_value  # Use the point exceedance, not section max
            
            # Calculate exceedance area (2D area in section view)
            # This represents the cross-sectional area of exceedance
            # Use the bounded exceedance heights (matching the visual fill)
            bounded_top = np.minimum(front_building_heights_array[exceedance_mask], 
                                     building_heights_array[exceedance_mask])
            exceedance_heights = bounded_top - envelope_heights_array[exceedance_mask]
            exceedance_heights = np.maximum(exceedance_heights, 0)  # Ensure non-negative
            sample_spacing = np.abs(distances_array[1] - distances_array[0]) if len(distances_array) > 1 else 1.0
            exceedance_area = np.sum(exceedance_heights) * sample_spacing  # m² (area in 2D section)
            
            # Calculate exceedance volume (approximate as area × unit width)
            # In a 2D section, this represents volume per unit width perpendicular to section
            # For a more accurate volume, we'd need the actual building width, but this gives a good approximation
            exceedance_volume = exceedance_area * 1.0  # m³ per meter width (approximate)
            
            # Add text annotation with exceedance values (sparse annotations)
            for i in np.where(exceedance_mask)[0]:
                if i % 20 == 0:  # Annotate every 20th point to avoid clutter
                    bounded_height = min(front_building_heights_array[i], building_heights_array[i])
                    exceedance_val = bounded_height - envelope_heights_array[i]
                    if exceedance_val > 1.0:  # Only show significant exceedances
                        ax.text(distances_array[i], bounded_height + 2,
                               f'{exceedance_val:.1f}m', fontsize=8, ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            # No exceedance
            max_exceedance_height = exceedance_value  # Use the point exceedance value
            exceedance_area = 0.0
            exceedance_volume = 0.0
        
        # Mark street point location
        ax.axvline(0, color='blue', linestyle=':', linewidth=2.5, label='Street Point', zorder=10)
        street_terrain_z = street_point.z  # Use street point Z as terrain reference
        ax.scatter([0], [street_terrain_z], color='blue', s=150, zorder=15, 
                  marker='o', edgecolors='black', linewidths=2, label='Street Location')
        
        # Add visual arrow showing exceedance height at street point
        if exceedance_value > 0:
            # Use the envelope and actual height from the point exceedance calculation
            # These are stored in the point_row metadata
            envelope_at_street = point_row.get('max_envelope_height', np.nan)
            building_at_street = point_row.get('max_actual_height', np.nan)
            
            # Only draw if we have valid values
            if not np.isnan(envelope_at_street) and not np.isnan(building_at_street):
                # Draw double-headed arrow showing exceedance
                arrow_x = 5.0  # Offset slightly from x=0 for visibility
                
                # Draw vertical line for exceedance
                ax.annotate('', 
                           xy=(arrow_x, building_at_street),
                           xytext=(arrow_x, envelope_at_street),
                           arrowprops=dict(arrowstyle='<->', color='darkred', lw=3, 
                                          shrinkA=0, shrinkB=0),
                           zorder=25)
                
                # Add label with exceedance value
                mid_y = (building_at_street + envelope_at_street) / 2
                ax.text(arrow_x + 3, mid_y, f'{exceedance_value:.2f}m',
                       fontsize=12, fontweight='bold', color='darkred',
                       verticalalignment='center', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='darkred', linewidth=2, alpha=0.95),
                       zorder=26)
                
                # Draw horizontal dashed lines to connect to the arrow
                ax.plot([0, arrow_x], [envelope_at_street, envelope_at_street], 
                       'r:', linewidth=1.5, alpha=0.7, zorder=24)
                ax.plot([0, arrow_x], [building_at_street, building_at_street], 
                       'k:', linewidth=1.5, alpha=0.7, zorder=24)
        
        # Add exceedance metrics text box
        metrics_text = (
            f'Exceedance Metrics:\n'
            f'Height: {max_exceedance_height:.2f} m\n'
            f'Area (2D): {exceedance_area:.1f} m²\n'
            f'Volume (approx): {exceedance_volume:.1f} m³/m'
        )
        
        # Position text box in upper left corner
        text_x = distances_array[0] + (distances_array[-1] - distances_array[0]) * 0.02
        text_y = ax.get_ylim()[1] * 0.95
        
        ax.text(text_x, text_y, metrics_text,
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='red', linewidth=2, alpha=0.9),
               verticalalignment='top', horizontalalignment='left',
               zorder=20)
        
        # Formatting
        ax.set_xlabel('Distance from Street Point (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Elevation (meters, absolute Z)', fontsize=12, fontweight='bold')
        
        ruleset_name = {'rio': 'Rio de Janeiro', 'saopaulo': 'São Paulo'}.get(ruleset.lower(), ruleset)
        ax.set_title(
            f'Street Section View - {label.upper()} Exceedance\n'
            f'Point Exceedance: {exceedance_value:.2f}m | {ruleset_name} Ruleset',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Set reasonable axis limits
        if len(building_heights_array) > 0:
            max_height = max(np.max(building_heights_array), np.max(envelope_heights_array))
            min_height = min(terrain_base, np.min(terrain_z_array) - 2)
            ax.set_ylim(min_height, max_height + 10)
        
        # Add horizontal reference line at street level
        ax.axhline(street_terrain_z, color='blue', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        section_path = output_dir / f"street_section_{label}_{ruleset}.png"
        plt.savefig(section_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved section view ({label}) to {section_path}")
    
    # Save selected point locations to GeoPackage
    selected_points_gdf = gpd.GeoDataFrame([
        {
            'label': 'high',
            'exceedance': high_point['exceedance'],
            'segment_idx': high_point['segment_idx'],
            'street_name': high_point.get('street_name', 'Unknown'),
            'geometry': high_point.geometry
        },
        {
            'label': 'mean',
            'exceedance': mean_point['exceedance'],
            'segment_idx': mean_point['segment_idx'],
            'street_name': mean_point.get('street_name', 'Unknown'),
            'geometry': mean_point.geometry
        },
        {
            'label': 'low',
            'exceedance': low_point['exceedance'],
            'segment_idx': low_point['segment_idx'],
            'street_name': low_point.get('street_name', 'Unknown'),
            'geometry': low_point.geometry
        }
    ], crs=street_points_gdf.crs)
    
    selected_points_path = output_dir / f"selected_section_points_{ruleset}.gpkg"
    selected_points_gdf.to_file(selected_points_path, driver='GPKG')
    logger.info(f"  Saved selected section points to {selected_points_path}")
    
    return selected_points_gdf


def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(
        description='Compute sky exposure plane exceedance along street centerlines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--stl', type=str, required=True, help='Path to STL file')
    parser.add_argument('--roads', type=str, default=None, help='Path to road network shapefile (optional, enables street-level analysis)')
    parser.add_argument('--footprints', type=str, required=True, help='Path to building footprints')
    parser.add_argument('--ruleset', type=str, default='rio', choices=['rio', 'saopaulo'],
                       help='Building code ruleset: rio (default) or saopaulo')
    parser.add_argument('--spacing', type=float, default=3.0, help='Distance between sample points (meters, default: 3.0)')
    parser.add_argument('--search-radius', type=float, default=75.0, help='Search radius for nearby buildings (meters, default: 75.0)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--area', type=str, default=None, help='Area name (e.g., vidigal, copacabana) for automatic path resolution')
    
    args = parser.parse_args()
    
    # Setup paths
    stl_path = Path(args.stl)
    if not stl_path.exists():
        logger.error(f"STL file not found: {stl_path}")
        sys.exit(1)
    
    roads_path = Path(args.roads) if args.roads else None
    if args.roads and roads_path and not roads_path.exists():
        logger.error(f"Road network file not found: {roads_path}")
        sys.exit(1)
    
    footprints_path = Path(args.footprints)
    if not footprints_path.exists():
        logger.error(f"Building footprints file not found: {footprints_path}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.area:
        from src.config import get_area_output_dir
        output_dir = get_area_output_dir(args.area) / "sky_exposure_streets"
    else:
        output_dir = PROJECT_ROOT / "outputs" / "sky_exposure_streets"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STREET-LEVEL SKY EXPOSURE PLANE EXCEEDANCE ANALYSIS")
    print("=" * 60)
    print(f"STL file: {stl_path}")
    print(f"Road network: {roads_path if roads_path else 'Not provided (building-level only)'}")
    print(f"Building footprints: {footprints_path}")
    print(f"Ruleset: {args.ruleset.upper()} (default: Rio de Janeiro)")
    if roads_path:
        print(f"Point spacing: {args.spacing}m")
        print(f"Search radius: {args.search_radius}m")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load mesh and terrain
    mesh = load_mesh(stl_path)
    terrain = extract_terrain_surface(mesh)
    terrain_bounds = terrain.bounds
    
    # Load road network (if provided)
    roads_gdf = None
    if roads_path:
        logger.info(f"Loading road network from {roads_path}...")
        roads_gdf = gpd.read_file(roads_path)
        logger.info(f"  Loaded {len(roads_gdf)} road segments")
        logger.info(f"  Original CRS: {roads_gdf.crs}")
        logger.info(f"  Road bounds: {roads_gdf.total_bounds}")
        
        # Handle coordinate system transformation
        stl_center_x = (terrain_bounds[0] + terrain_bounds[1]) / 2
        stl_center_y = (terrain_bounds[2] + terrain_bounds[3]) / 2
        
        roads_center_x = (roads_gdf.total_bounds[0] + roads_gdf.total_bounds[2]) / 2
        roads_center_y = (roads_gdf.total_bounds[1] + roads_gdf.total_bounds[3]) / 2
        
        dx = stl_center_x - roads_center_x
        dy = stl_center_y - roads_center_y
        
        if abs(dx) > 100 or abs(dy) > 100:
            logger.info(f"  Detected coordinate system mismatch - transforming to local coordinates")
            logger.info(f"  Applying translation: dx={dx:.1f}, dy={dy:.1f}")
            roads_gdf.geometry = roads_gdf.geometry.translate(xoff=dx, yoff=dy)
            logger.info(f"  Transformed road bounds: {roads_gdf.total_bounds}")
    else:
        logger.info("No road network provided - will compute building-level exceedance only")
    
    # Load building footprints
    logger.info(f"Loading building footprints from {footprints_path}...")
    buildings_gdf = load_building_footprints(
        footprints_path,
        terrain_bounds=terrain.bounds,
        buffer_distance=0.0  # No buffer needed
    )
    
    # Normalize height columns (handles 'base'/'altura' -> 'base_height'/'top_height')
    buildings_gdf = normalize_height_columns(buildings_gdf)
    
    # Validate required columns
    if 'base_height' not in buildings_gdf.columns or 'top_height' not in buildings_gdf.columns:
        raise ValueError(f"Building footprints must have 'base_height' and 'top_height' columns. "
                        f"Found columns: {list(buildings_gdf.columns)}")
    
    # Filter outlier buildings
    # These thresholds remove erroneous/very small buildings that can skew the analysis
    MIN_BUILDING_HEIGHT = 1.5  # meters - buildings shorter than this are likely errors
    
    original_count = len(buildings_gdf)
    
    # Calculate building footprint area and height
    buildings_gdf['footprint_area'] = buildings_gdf.geometry.area
    buildings_gdf['building_height'] = buildings_gdf['top_height'] - buildings_gdf['base_height']
    
    # Apply filters
    # 1. Remove buildings with footprint area too small (noise/errors)
    area_mask = buildings_gdf['footprint_area'] >= MIN_BUILDING_AREA
    
    # 2. Remove buildings with height too low (noise/errors)
    height_mask = buildings_gdf['building_height'] >= MIN_BUILDING_HEIGHT
    
    # 3. For informal areas, also filter very large buildings (likely misclassified)
    if args.area and not is_formal_area(args.area):
        max_area_mask = buildings_gdf['footprint_area'] <= MAX_FILTER_AREA
        combined_mask = area_mask & height_mask & max_area_mask
        logger.info(f"Filtering buildings (informal area: {args.area}):")
    else:
        combined_mask = area_mask & height_mask
        logger.info(f"Filtering buildings:")
    
    buildings_gdf = buildings_gdf[combined_mask].copy()
    filtered_count = len(buildings_gdf)
    
    logger.info(f"  Original: {original_count} buildings")
    logger.info(f"  Removed: {original_count - filtered_count} outliers")
    logger.info(f"    - Min area: {MIN_BUILDING_AREA} m² (removed {(~area_mask).sum()})")
    logger.info(f"    - Min height: {MIN_BUILDING_HEIGHT} m (removed {(~height_mask).sum()})")
    if args.area and not is_formal_area(args.area):
        logger.info(f"    - Max area: {MAX_FILTER_AREA} m² (removed {(~max_area_mask).sum()})")
    logger.info(f"  Remaining: {filtered_count} buildings")
    
    # Extract building meshes
    logger.info("Extracting building meshes from STL...")
    building_meshes = extract_building_meshes(
        mesh, buildings_gdf, terrain.bounds, z_threshold=None
    )
    logger.info(f"  Extracted {len(building_meshes)} building meshes")
    
    # Compute building-level exceedance
    logger.info("Computing building-level exceedance...")
    building_exceedance_results = {}
    
    for building_idx in tqdm(building_meshes.keys(), desc="Computing building exceedances"):
        if building_idx not in buildings_gdf.index:
            continue
        
        building_data = building_meshes[building_idx]
        building_mesh = building_data['mesh']
        footprint = building_data['footprint']
        
        building_row = buildings_gdf.loc[building_idx]
        base_height = float(building_row['base_height'])
        top_height = float(building_row['top_height'])
        
        exceedance_metrics = compute_building_exceedance(
            building_idx, building_mesh, footprint, base_height, top_height, args.ruleset
        )
        building_exceedance_results[building_idx] = exceedance_metrics
    
    # Create building-level exceedance map
    logger.info("Generating building-level exceedance map...")
    building_map_path = output_dir / f"building_exceedance_map_{args.ruleset}.png"
    create_building_exceedance_map(
        buildings_gdf, building_exceedance_results, args.ruleset, building_map_path
    )
    
    # Save building exceedance results
    building_exceedance_df = pd.DataFrame([
        {
            'building_idx': idx,
            **metrics
        }
        for idx, metrics in building_exceedance_results.items()
    ])
    building_csv_path = output_dir / f"building_exceedance_statistics_{args.ruleset}.csv"
    building_exceedance_df.to_csv(building_csv_path, index=False)
    logger.info(f"  Saved building exceedance statistics to {building_csv_path}")
    
    # Street-level analysis (only if roads provided)
    points_gdf = None
    segments_gdf = None
    selected_points_gdf = None
    
    if roads_gdf is not None:
        # Sample points along streets
        points_gdf = sample_street_points(roads_gdf, args.spacing, terrain)
        
        # Remove points with invalid elevation
        valid_mask = ~points_gdf.geometry.apply(lambda p: np.isnan(p.z))
        if not valid_mask.all():
            logger.warning(f"  Removed {np.sum(~valid_mask)} points with invalid elevation")
            points_gdf = points_gdf[valid_mask].copy()
        
        # Convert points to numpy array
        street_points_3d = np.array([
            [geom.x, geom.y, geom.z] for geom in points_gdf.geometry
        ])
        
        # Compute exceedance
        exceedance_values, metadata_list = compute_street_exceedance(
            street_points_3d, roads_gdf, buildings_gdf, building_meshes,
            terrain, ruleset=args.ruleset, search_radius=args.search_radius
        )
        
        # Add exceedance values and metadata to points GeoDataFrame
        points_gdf = points_gdf.copy()
        points_gdf['exceedance'] = exceedance_values
        for key in metadata_list[0].keys():
            points_gdf[key] = [m[key] for m in metadata_list]
        
        # Aggregate to segment level
        segments_gdf = aggregate_segment_statistics(points_gdf, exceedance_values, roads_gdf)
        
        # Save street-level results
        logger.info("Saving street-level results...")
        
        # Point-level output
        points_output = output_dir / f"street_exceedance_points_{args.ruleset}.gpkg"
        points_gdf.to_file(points_output, driver='GPKG')
        logger.info(f"  Saved point-level results to {points_output}")
        
        # Segment-level output
        segments_output = output_dir / f"street_exceedance_segments_{args.ruleset}.gpkg"
        segments_gdf.to_file(segments_output, driver='GPKG')
        logger.info(f"  Saved segment-level results to {segments_output}")
        
        # CSV summary
        csv_output = output_dir / f"street_exceedance_statistics_{args.ruleset}.csv"
        segments_gdf.drop(columns=['geometry']).to_csv(csv_output, index=False)
        logger.info(f"  Saved statistics to {csv_output}")
        
        # Visualizations
        logger.info("Generating street-level visualizations...")
        
        # Street section views (select points first, then use for map)
        selected_points_gdf = create_street_section_views(
            points_gdf, mesh, buildings_gdf, building_meshes, terrain,
            args.ruleset, output_dir
        )
        
        # Street exceedance map (with selected points highlighted)
        map_path = output_dir / f"street_exceedance_map_{args.ruleset}.png"
        create_street_exceedance_map(
            segments_gdf, points_gdf, buildings_gdf, args.ruleset, map_path,
            selected_points_gdf=selected_points_gdf
        )
        
        # Statistics plots
        create_statistics_plots(segments_gdf, points_gdf, args.ruleset, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"SKY EXPOSURE PLANE EXCEEDANCE SUMMARY ({args.ruleset.upper()} RULESET)")
    print("=" * 60)
    if roads_gdf is not None:
        print(f"Total street segments: {len(segments_gdf)}")
        print(f"Total sample points: {len(points_gdf)}")
        print(f"\nPoint-level exceedance:")
        exceedance_array = points_gdf['exceedance'].values
        points_with_exceedance = (exceedance_array > 0).sum()
        print(f"  Points with exceedance: {points_with_exceedance} ({points_with_exceedance/len(exceedance_array)*100:.1f}%)")
        if points_with_exceedance > 0:
            exceedance_violations = exceedance_array[exceedance_array > 0]
            print(f"  Mean exceedance (violations only): {np.mean(exceedance_violations):.2f}m")
            print(f"  Max exceedance: {np.max(exceedance_array):.2f}m")
            print(f"  Median exceedance: {np.median(exceedance_violations):.2f}m")
        else:
            print(f"  No exceedances found (all buildings comply)")
        print(f"\nSegment-level exceedance (mean values):")
        print(f"  Mean: {segments_gdf['mean_exceedance'].mean():.2f}m")
        print(f"  Max: {segments_gdf['max_exceedance'].max():.2f}m")
        print(f"  Segments with violations: {(segments_gdf['mean_exceedance'] > 0).sum()}")
    print(f"\nBuilding-level exceedance:")
    exceedance_ratios = [m['exceedance_ratio'] for m in building_exceedance_results.values()]
    buildings_with_exceedance = sum(1 for r in exceedance_ratios if r > 0)
    total_buildings = len(exceedance_ratios)
    print(f"  Buildings with exceedance: {buildings_with_exceedance} / {total_buildings} ({buildings_with_exceedance/total_buildings*100:.1f}%)")
    if buildings_with_exceedance > 0:
        exceedance_violations = [r for r in exceedance_ratios if r > 0]
        print(f"  Mean exceedance ratio (violations only): {np.mean(exceedance_violations):.2f}%")
        print(f"  Max exceedance ratio: {np.max(exceedance_ratios):.2f}%")
        print(f"  Median exceedance ratio: {np.median(exceedance_violations):.2f}%")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

