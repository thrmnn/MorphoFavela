"""
Utilities for data alignment and road redirection.

This module handles:
- Coordinate system alignment between DTM, buildings, roads, and STL
- Road-building intersection detection
- Road redirection to avoid building collisions
"""

import numpy as np
import geopandas as gpd
import pyvista as pv
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import warnings
import logging

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)

# Alignment tolerance (meters)
ALIGNMENT_TOLERANCE = 0.1  # 10cm


def check_crs_alignment(
    datasets: dict,
    tolerance: float = ALIGNMENT_TOLERANCE
) -> tuple:
    """
    Check if all datasets have compatible CRS and are aligned.
    
    Args:
        datasets: Dict with keys 'dtm', 'buildings', 'roads', 'stl' (optional)
                 Values are GeoDataFrames, rasterio datasets, or STL bounds
        tolerance: Spatial alignment tolerance in meters
    
    Returns:
        Tuple of (is_aligned, warnings, corrections)
        - is_aligned: Boolean indicating if all datasets are aligned
        - warnings: List of warning messages
        - corrections: Dict of suggested corrections
    """
    warnings_list = []
    corrections = {}
    
    # Extract CRS from each dataset
    crs_dict = {}
    
    if 'dtm' in datasets and datasets['dtm'] is not None:
        if HAS_RASTERIO:
            with rasterio.open(datasets['dtm']) as src:
                crs_dict['dtm'] = src.crs
        else:
            warnings_list.append("rasterio not available - cannot check DTM CRS")
    
    if 'buildings' in datasets and datasets['buildings'] is not None:
        crs_dict['buildings'] = datasets['buildings'].crs
    
    if 'roads' in datasets and datasets['roads'] is not None:
        crs_dict['roads'] = datasets['roads'].crs
    
    # STL doesn't have CRS, but we can check bounds alignment
    if 'stl' in datasets and datasets['stl'] is not None:
        crs_dict['stl'] = None  # STL has no CRS
    
    # Check CRS compatibility
    crs_values = [crs for crs in crs_dict.values() if crs is not None]
    if len(set(str(crs) for crs in crs_values)) > 1:
        warnings_list.append(
            f"Multiple CRS detected: {dict(crs_dict)}. "
            "Datasets should be in the same coordinate system."
        )
        # Suggest transformation to first non-None CRS
        target_crs = crs_values[0] if crs_values else None
        if target_crs:
            corrections['target_crs'] = target_crs
    
    # Check spatial alignment (bounds)
    bounds_dict = {}
    
    if 'dtm' in datasets and datasets['dtm'] is not None:
        if HAS_RASTERIO:
            with rasterio.open(datasets['dtm']) as src:
                bounds_dict['dtm'] = src.bounds
        else:
            warnings_list.append("rasterio not available - cannot check DTM bounds")
    
    if 'buildings' in datasets and datasets['buildings'] is not None:
        bounds_dict['buildings'] = datasets['buildings'].total_bounds
    
    if 'roads' in datasets and datasets['roads'] is not None:
        bounds_dict['roads'] = datasets['roads'].total_bounds
    
    if 'stl' in datasets and datasets['stl'] is not None:
        if isinstance(datasets['stl'], pv.PolyData):
            bounds_dict['stl'] = datasets['stl'].bounds
        elif isinstance(datasets['stl'], (tuple, list, np.ndarray)):
            # Assume it's bounds tuple (x_min, x_max, y_min, y_max, z_min, z_max)
            bounds_dict['stl'] = datasets['stl']
    
    # Check bounds alignment
    if len(bounds_dict) > 1:
        # Compare 2D bounds (x, y)
        centers = {}
        for name, bounds in bounds_dict.items():
            if len(bounds) >= 4:
                centers[name] = (
                    (bounds[0] + bounds[1]) / 2,  # x center
                    (bounds[2] + bounds[3]) / 2   # y center
                )
        
        if len(centers) > 1:
            center_values = list(centers.values())
            center_x = [c[0] for c in center_values]
            center_y = [c[1] for c in center_values]
            
            x_diff = max(center_x) - min(center_x)
            y_diff = max(center_y) - min(center_y)
            
            if x_diff > tolerance or y_diff > tolerance:
                warnings_list.append(
                    f"Datasets are misaligned: "
                    f"X difference: {x_diff:.2f}m, Y difference: {y_diff:.2f}m "
                    f"(tolerance: {tolerance}m)"
                )
                
                # Calculate correction (translate to first dataset's center)
                target_name = list(centers.keys())[0]
                target_center = centers[target_name]
                corrections['translation'] = {}
                
                for name, center in centers.items():
                    if name != target_name:
                        dx = target_center[0] - center[0]
                        dy = target_center[1] - center[1]
                        if abs(dx) > tolerance or abs(dy) > tolerance:
                            corrections['translation'][name] = (dx, dy)
    
    is_aligned = len(warnings_list) == 0
    
    return is_aligned, warnings_list, corrections


def auto_correct_alignment(
    datasets: dict,
    corrections: dict,
    inplace: bool = False
) -> dict:
    """
    Automatically correct dataset alignment based on corrections dict.
    
    Args:
        datasets: Dict of datasets to correct
        corrections: Dict from check_crs_alignment with correction suggestions
        inplace: If True, modify datasets in place
    
    Returns:
        Dict of corrected datasets
    """
    corrected = datasets.copy() if not inplace else datasets
    
    # Apply CRS transformation
    if 'target_crs' in corrections:
        target_crs = corrections['target_crs']
        logger.warning(f"Transforming datasets to CRS: {target_crs}")
        
        if 'buildings' in corrected and corrected['buildings'] is not None:
            if corrected['buildings'].crs != target_crs:
                logger.info(f"  Transforming buildings to {target_crs}")
                corrected['buildings'] = corrected['buildings'].to_crs(target_crs)
        
        if 'roads' in corrected and corrected['roads'] is not None:
            if corrected['roads'].crs != target_crs:
                logger.info(f"  Transforming roads to {target_crs}")
                corrected['roads'] = corrected['roads'].to_crs(target_crs)
    
    # Apply translation
    if 'translation' in corrections:
        translations = corrections['translation']
        logger.warning("Applying translations to align datasets:")
        
        for name, (dx, dy) in translations.items():
            logger.info(f"  Translating {name} by dx={dx:.2f}m, dy={dy:.2f}m")
            
            if name == 'buildings' and 'buildings' in corrected:
                corrected['buildings'].geometry = corrected['buildings'].geometry.translate(
                    xoff=dx, yoff=dy
                )
            elif name == 'roads' and 'roads' in corrected:
                corrected['roads'].geometry = corrected['roads'].geometry.translate(
                    xoff=dx, yoff=dy
                )
            elif name == 'stl' and 'stl' in corrected:
                # STL translation would need to be applied to mesh points
                # This is handled in svf_utils.load_building_footprints
                logger.info(f"  STL translation should be handled during mesh loading")
    
    return corrected


def detect_road_building_intersections(
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Detect road segments that intersect with building polygons.
    
    Args:
        roads_gdf: GeoDataFrame with LineString geometries
        buildings_gdf: GeoDataFrame with Polygon geometries
    
    Returns:
        GeoDataFrame with intersecting road segments and intersection info
    """
    logger.info("Detecting road-building intersections...")
    
    # Ensure same CRS
    if roads_gdf.crs != buildings_gdf.crs:
        logger.warning(f"CRS mismatch: roads={roads_gdf.crs}, buildings={buildings_gdf.crs}")
        buildings_gdf = buildings_gdf.to_crs(roads_gdf.crs)
    
    # Find intersections
    intersecting_roads = []
    
    for idx, road in roads_gdf.iterrows():
        road_geom = road.geometry
        
        # Check if road intersects any building
        intersects = buildings_gdf.geometry.intersects(road_geom)
        
        if intersects.any():
            # Get intersecting buildings
            intersecting_buildings = buildings_gdf[intersects]
            
            # Calculate intersection geometry
            intersection_geom = road_geom.intersection(
                unary_union(intersecting_buildings.geometry.values)
            )
            
            intersecting_roads.append({
                'road_idx': idx,
                'n_buildings': len(intersecting_buildings),
                'intersection_length': intersection_geom.length if hasattr(intersection_geom, 'length') else 0,
                'geometry': intersection_geom  # Use 'geometry' as column name for GeoDataFrame
            })
    
    if intersecting_roads:
        logger.warning(f"Found {len(intersecting_roads)} road segments intersecting buildings")
        # Create GeoDataFrame with geometry column
        result_df = gpd.GeoDataFrame(intersecting_roads, crs=roads_gdf.crs)
        # Add road_geometry as regular column (not geometry)
        result_df['road_geometry'] = [roads_gdf.loc[r['road_idx']].geometry for r in intersecting_roads]
        return result_df
    else:
        logger.info("No road-building intersections detected")
        # Create empty GeoDataFrame with geometry column
        return gpd.GeoDataFrame(
            geometry=[],
            columns=['road_idx', 'n_buildings', 'intersection_length', 'road_geometry'],
            crs=roads_gdf.crs
        )


def redirect_road_parallel_offset(
    road: LineString,
    buildings: gpd.GeoDataFrame,
    offset_distance: float = 2.0,
    max_attempts: int = 4
) -> LineString:
    """
    Redirect a road by parallel offset to avoid buildings.
    
    Tries offsetting in both directions (left/right) and returns the first
    valid offset that doesn't intersect buildings.
    
    Args:
        road: LineString geometry of the road
        buildings: GeoDataFrame with building polygons
        offset_distance: Distance to offset (meters)
        max_attempts: Maximum number of offset attempts (tries both directions)
    
    Returns:
        Redirected LineString, or original if redirection fails
    """
    # Try offsetting in both directions
    for direction in [1, -1]:
        offset = offset_distance * direction
        
        try:
            # Create parallel offset
            offset_road = road.parallel_offset(offset, 'left')
            
            # Handle MultiLineString (can happen with complex geometries)
            if hasattr(offset_road, 'geoms'):
                # Use the longest segment
                offset_road = max(offset_road.geoms, key=lambda g: g.length)
            
            # Check if offset road intersects buildings
            if not buildings.geometry.intersects(offset_road).any():
                logger.debug(f"  Successfully offset road by {offset:.1f}m")
                return offset_road
            
        except Exception as e:
            logger.debug(f"  Offset failed (direction={direction}): {e}")
            continue
    
    # If all offsets fail, return original
    logger.warning(f"  Could not redirect road - using original geometry")
    return road


def redirect_road_simple_reroute(
    road: LineString,
    buildings: gpd.GeoDataFrame,
    buffer_distance: float = 2.0
) -> LineString:
    """
    Redirect a road by simple rerouting around buildings.
    
    Creates a buffer around buildings and clips the road, then connects
    the remaining segments with a simple path around the building.
    
    Args:
        road: LineString geometry of the road
        buildings: GeoDataFrame with building polygons
        buffer_distance: Buffer distance around buildings (meters)
    
    Returns:
        Redirected LineString, or original if redirection fails
    """
    try:
        # Create buffer around buildings
        building_buffers = buildings.geometry.buffer(buffer_distance)
        combined_buffer = unary_union(building_buffers.values)
        
        # Clip road to remove intersecting parts
        road_clipped = road.difference(combined_buffer)
        
        # If road was completely removed, return original
        if road_clipped.is_empty:
            return road
        
        # Handle MultiLineString
        if hasattr(road_clipped, 'geoms'):
            segments = list(road_clipped.geoms)
        else:
            segments = [road_clipped]
        
        # If we have multiple segments, try to connect them
        if len(segments) > 1:
            # Simple approach: connect segments with straight lines
            # This is a basic implementation - could be improved
            connected_segments = []
            for i, seg in enumerate(segments):
                connected_segments.append(seg)
                if i < len(segments) - 1:
                    # Connect to next segment
                    end_point = Point(seg.coords[-1])
                    next_start = Point(segments[i+1].coords[0])
                    connection = LineString([end_point, next_start])
                    connected_segments.append(connection)
            
            # Combine all segments
            from shapely.ops import linemerge
            merged = linemerge(connected_segments)
            
            if hasattr(merged, 'geoms'):
                # Use longest segment
                merged = max(merged.geoms, key=lambda g: g.length)
            
            return merged
        
        return segments[0] if segments else road
        
    except Exception as e:
        logger.warning(f"  Simple reroute failed: {e}")
        return road


def redirect_roads(
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    method: str = 'parallel_offset',
    offset_distance: float = 2.0,
    buffer_distance: float = 2.0
) -> tuple:
    """
    Redirect roads to avoid building intersections.
    
    Args:
        roads_gdf: GeoDataFrame with road LineStrings
        buildings_gdf: GeoDataFrame with building Polygons
        method: Redirection method ('parallel_offset' or 'simple_reroute')
        offset_distance: Distance for parallel offset (meters)
        buffer_distance: Buffer distance for simple reroute (meters)
    
    Returns:
        Tuple of (redirected_roads_gdf, intersection_info)
        - redirected_roads_gdf: GeoDataFrame with redirected roads
        - intersection_info: GeoDataFrame with intersection details
    """
    logger.info(f"Redirecting roads using method: {method}")
    
    # Detect intersections
    intersection_info = detect_road_building_intersections(roads_gdf, buildings_gdf)
    
    if len(intersection_info) == 0:
        logger.info("No intersections to redirect")
        return roads_gdf.copy(), intersection_info
    
    # Create copy of roads
    redirected_roads = roads_gdf.copy()
    redirected_count = 0
    
    # Redirect each intersecting road
    for idx, intersection in intersection_info.iterrows():
        road_idx = intersection['road_idx']
        original_road = redirected_roads.loc[road_idx, 'geometry']
        
        # Apply redirection method
        if method == 'parallel_offset':
            redirected_road = redirect_road_parallel_offset(
                original_road, buildings_gdf, offset_distance
            )
        elif method == 'simple_reroute':
            redirected_road = redirect_road_simple_reroute(
                original_road, buildings_gdf, buffer_distance
            )
        else:
            logger.warning(f"Unknown method: {method}, using parallel_offset")
            redirected_road = redirect_road_parallel_offset(
                original_road, buildings_gdf, offset_distance
            )
        
        # Update road geometry
        redirected_roads.loc[road_idx, 'geometry'] = redirected_road
        
        # Check if redirection was successful
        if redirected_road != original_road:
            # Verify no intersection
            still_intersects = buildings_gdf.geometry.intersects(redirected_road).any()
            if not still_intersects:
                redirected_count += 1
                logger.debug(f"  Road {road_idx} successfully redirected")
            else:
                logger.warning(f"  Road {road_idx} still intersects after redirection")
    
    logger.info(f"Redirected {redirected_count}/{len(intersection_info)} roads")
    
    return redirected_roads, intersection_info
