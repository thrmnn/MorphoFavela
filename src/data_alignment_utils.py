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
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import logging

try:
    import rasterio
    from rasterio.crs import CRS  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)

# Alignment tolerance (meters)
ALIGNMENT_TOLERANCE = 0.1  # 10cm


def georeference_stl_using_dtm(
    stl_bounds: tuple, dtm_path: Path, tolerance: float = ALIGNMENT_TOLERANCE
) -> tuple:
    """
    Georeference ungeoreferenced STL using DTM bounds as reference.

    Simple translation: align STL center to DTM center (X, Y, Z).
    Transforms STL FROM local coordinates TO world coordinates.

    Args:
        stl_bounds: STL mesh bounds tuple (x_min, x_max, y_min, y_max, z_min, z_max)
        dtm_path: Path to DTM raster (required)
        tolerance: Alignment tolerance in meters

    Returns:
        Tuple of (dx, dy, dz, dtm_bounds):
        - dx, dy, dz: Translation offsets to apply to STL mesh vertices
        - dtm_bounds: DTM bounds for validation
    """
    if not HAS_RASTERIO:
        raise ValueError("rasterio is required for DTM-based georeferencing")

    if dtm_path is None or not dtm_path.exists():
        raise ValueError(f"DTM file not found: {dtm_path}")

    # Get STL center (in local coordinates)
    stl_x_min, stl_x_max, stl_y_min, stl_y_max, stl_z_min, stl_z_max = (
        stl_bounds[0],
        stl_bounds[1],
        stl_bounds[2],
        stl_bounds[3],
        stl_bounds[4],
        stl_bounds[5],
    )
    stl_center_x = (stl_x_min + stl_x_max) / 2
    stl_center_y = (stl_y_min + stl_y_max) / 2
    stl_center_z = (stl_z_min + stl_z_max) / 2

    # Get DTM center and elevation range (in world coordinates)
    with rasterio.open(dtm_path) as src:
        dtm_bounds = src.bounds  # (left, bottom, right, top)
        dtm_center_x = (dtm_bounds.left + dtm_bounds.right) / 2
        dtm_center_y = (dtm_bounds.bottom + dtm_bounds.top) / 2

        # Read DTM to get elevation range
        dtm_data = src.read(1)
        # Filter out NaN and invalid values (extremely large numbers, nodata values)
        valid_mask = ~np.isnan(dtm_data)
        if src.nodata is not None:
            valid_mask = valid_mask & (dtm_data != src.nodata)
        # Filter out unrealistic values (e.g., > 10000m elevation)
        valid_mask = valid_mask & (np.abs(dtm_data) < 10000)

        dtm_valid = dtm_data[valid_mask]
        if len(dtm_valid) > 0:
            dtm_z_min = float(np.nanmin(dtm_valid))
            dtm_z_max = float(np.nanmax(dtm_valid))
            # Use minimum elevation to align STL base
        else:
            logger.warning("DTM has no valid elevation data, using STL Z as-is")
            dtm_z_min = stl_z_min
            dtm_z_max = stl_z_max

    # Calculate translation: to transform STL FROM local TO world coordinates
    # X, Y: align centers
    dx = dtm_center_x - stl_center_x
    dy = dtm_center_y - stl_center_y
    # Z: align STL base (z_min) to DTM minimum elevation
    # This ensures terrain aligns properly
    dz = dtm_z_min - stl_z_min

    logger.info("Georeferencing STL using DTM bounds")
    logger.info(f"  DTM bounds: {dtm_bounds}")
    logger.info(f"  DTM center: ({dtm_center_x:.2f}, {dtm_center_y:.2f})")
    logger.info(f"  DTM elevation range: {dtm_z_min:.2f}m to {dtm_z_max:.2f}m")
    logger.info(
        f"  STL bounds (local): ({stl_x_min:.2f}, {stl_y_min:.2f}, {stl_z_min:.2f}) to ({stl_x_max:.2f}, {stl_y_max:.2f}, {stl_z_max:.2f})"
    )
    logger.info(
        f"  STL center (local): ({stl_center_x:.2f}, {stl_center_y:.2f}, {stl_center_z:.2f})"
    )
    logger.info(f"  Translation: dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m")

    return dx, dy, dz, dtm_bounds


def validate_alignment(
    dtm_path: Path = None,
    buildings_gdf: gpd.GeoDataFrame = None,
    roads_gdf: gpd.GeoDataFrame = None,
    stl_bounds: tuple = None,
    transform: tuple = None,
    tolerance: float = ALIGNMENT_TOLERANCE,
) -> tuple:
    """
    Validate that alignment is correct after transformation.

    Args:
        dtm_path: Optional DTM raster path
        buildings_gdf: Optional building footprints
        roads_gdf: Optional roads
        stl_bounds: STL bounds after transformation
        transform: (dx, dy) transformation applied
        tolerance: Alignment tolerance in meters

    Returns:
        Tuple of (is_valid, warnings):
        - is_valid: Boolean indicating if alignment is acceptable
        - warnings: List of warning messages
    """
    warnings_list = []
    dx, dy = transform if transform else (0.0, 0.0)

    # Get reference bounds
    reference_bounds = None
    if dtm_path and dtm_path.exists() and HAS_RASTERIO:
        try:
            with rasterio.open(dtm_path) as src:
                reference_bounds = src.bounds
        except Exception:
            pass

    if (
        reference_bounds is None
        and buildings_gdf is not None
        and len(buildings_gdf) > 0
    ):
        bld_bounds = buildings_gdf.total_bounds
        reference_bounds = (bld_bounds[0], bld_bounds[1], bld_bounds[2], bld_bounds[3])

    if reference_bounds is None and roads_gdf is not None and len(roads_gdf) > 0:
        road_bounds = roads_gdf.total_bounds
        reference_bounds = (
            road_bounds[0],
            road_bounds[1],
            road_bounds[2],
            road_bounds[3],
        )

    if reference_bounds is None or stl_bounds is None:
        warnings_list.append(
            "Cannot validate alignment: missing reference or STL bounds"
        )
        return False, warnings_list

    # Check if transformed STL bounds overlap with reference
    stl_x_min, stl_x_max, stl_y_min, stl_y_max = (
        stl_bounds[0],
        stl_bounds[1],
        stl_bounds[2],
        stl_bounds[3],
    )
    ref_x_min, ref_y_min, ref_x_max, ref_y_max = (
        reference_bounds[0],
        reference_bounds[1],
        reference_bounds[2],
        reference_bounds[3],
    )

    # Calculate overlap
    overlap_x = min(stl_x_max, ref_x_max) - max(stl_x_min, ref_x_min)
    overlap_y = min(stl_y_max, ref_y_max) - max(stl_y_min, ref_y_min)

    if overlap_x < 0 or overlap_y < 0:
        warnings_list.append(
            f"STL and reference bounds do not overlap after transformation. "
            f"STL: ({stl_x_min:.2f}, {stl_y_min:.2f}) to ({stl_x_max:.2f}, {stl_y_max:.2f}), "
            f"Reference: ({ref_x_min:.2f}, {ref_y_min:.2f}) to ({ref_x_max:.2f}, {ref_y_max:.2f})"
        )
        return False, warnings_list

    # Check center alignment
    stl_center_x = (stl_x_min + stl_x_max) / 2
    stl_center_y = (stl_y_min + stl_y_max) / 2
    ref_center_x = (ref_x_min + ref_x_max) / 2
    ref_center_y = (ref_y_min + ref_y_max) / 2

    center_diff_x = abs(stl_center_x - ref_center_x)
    center_diff_y = abs(stl_center_y - ref_center_y)

    if center_diff_x > tolerance or center_diff_y > tolerance:
        warnings_list.append(
            f"Center misalignment: X={center_diff_x:.2f}m, Y={center_diff_y:.2f}m "
            f"(tolerance: {tolerance}m)"
        )
        return False, warnings_list

    logger.info(
        f"Alignment validated: center difference X={center_diff_x:.3f}m, Y={center_diff_y:.3f}m"
    )
    return True, warnings_list


def check_crs_alignment(
    datasets: dict, tolerance: float = ALIGNMENT_TOLERANCE
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

    if "dtm" in datasets and datasets["dtm"] is not None:
        if HAS_RASTERIO:
            with rasterio.open(datasets["dtm"]) as src:
                crs_dict["dtm"] = src.crs
        else:
            warnings_list.append("rasterio not available - cannot check DTM CRS")

    if "buildings" in datasets and datasets["buildings"] is not None:
        crs_dict["buildings"] = datasets["buildings"].crs

    if "roads" in datasets and datasets["roads"] is not None:
        crs_dict["roads"] = datasets["roads"].crs

    # STL doesn't have CRS, but we can check bounds alignment
    if "stl" in datasets and datasets["stl"] is not None:
        crs_dict["stl"] = None  # STL has no CRS

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
            corrections["target_crs"] = target_crs

    # Check spatial alignment (bounds)
    bounds_dict = {}

    if "dtm" in datasets and datasets["dtm"] is not None:
        if HAS_RASTERIO:
            with rasterio.open(datasets["dtm"]) as src:
                bounds_dict["dtm"] = src.bounds
        else:
            warnings_list.append("rasterio not available - cannot check DTM bounds")

    if "buildings" in datasets and datasets["buildings"] is not None:
        bounds_dict["buildings"] = datasets["buildings"].total_bounds

    if "roads" in datasets and datasets["roads"] is not None:
        bounds_dict["roads"] = datasets["roads"].total_bounds

    if "stl" in datasets and datasets["stl"] is not None:
        if isinstance(datasets["stl"], pv.PolyData):
            bounds_dict["stl"] = datasets["stl"].bounds
        elif isinstance(datasets["stl"], (tuple, list, np.ndarray)):
            # Assume it's bounds tuple (x_min, x_max, y_min, y_max, z_min, z_max)
            bounds_dict["stl"] = datasets["stl"]

    # Check bounds alignment
    if len(bounds_dict) > 1:
        # Compare 2D bounds (x, y)
        centers = {}
        for name, bounds in bounds_dict.items():
            if len(bounds) >= 4:
                centers[name] = (
                    (bounds[0] + bounds[1]) / 2,  # x center
                    (bounds[2] + bounds[3]) / 2,  # y center
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
                corrections["translation"] = {}

                for name, center in centers.items():
                    if name != target_name:
                        dx = target_center[0] - center[0]
                        dy = target_center[1] - center[1]
                        if abs(dx) > tolerance or abs(dy) > tolerance:
                            corrections["translation"][name] = (dx, dy)

    is_aligned = len(warnings_list) == 0

    return is_aligned, warnings_list, corrections


def auto_correct_alignment(
    datasets: dict, corrections: dict, inplace: bool = False
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
    if "target_crs" in corrections:
        target_crs = corrections["target_crs"]
        logger.warning(f"Transforming datasets to CRS: {target_crs}")

        if "buildings" in corrected and corrected["buildings"] is not None:
            if corrected["buildings"].crs != target_crs:
                logger.info(f"  Transforming buildings to {target_crs}")
                corrected["buildings"] = corrected["buildings"].to_crs(target_crs)

        if "roads" in corrected and corrected["roads"] is not None:
            if corrected["roads"].crs != target_crs:
                logger.info(f"  Transforming roads to {target_crs}")
                corrected["roads"] = corrected["roads"].to_crs(target_crs)

    # Apply translation
    if "translation" in corrections:
        translations = corrections["translation"]
        logger.warning("Applying translations to align datasets:")

        for name, (dx, dy) in translations.items():
            logger.info(f"  Translating {name} by dx={dx:.2f}m, dy={dy:.2f}m")

            if name == "buildings" and "buildings" in corrected:
                corrected["buildings"].geometry = corrected[
                    "buildings"
                ].geometry.translate(xoff=dx, yoff=dy)
            elif name == "roads" and "roads" in corrected:
                corrected["roads"].geometry = corrected["roads"].geometry.translate(
                    xoff=dx, yoff=dy
                )
            elif name == "stl" and "stl" in corrected:
                # STL translation would need to be applied to mesh points
                # This is handled in svf_utils.load_building_footprints
                logger.info("  STL translation should be handled during mesh loading")

    return corrected


def detect_road_building_intersections(
    roads_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame
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
        logger.warning(
            f"CRS mismatch: roads={roads_gdf.crs}, buildings={buildings_gdf.crs}"
        )
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

            intersecting_roads.append(
                {
                    "road_idx": idx,
                    "n_buildings": len(intersecting_buildings),
                    "intersection_length": intersection_geom.length
                    if hasattr(intersection_geom, "length")
                    else 0,
                    "geometry": intersection_geom,  # Use 'geometry' as column name for GeoDataFrame
                }
            )

    if intersecting_roads:
        logger.warning(
            f"Found {len(intersecting_roads)} road segments intersecting buildings"
        )
        # Create GeoDataFrame with geometry column
        result_df = gpd.GeoDataFrame(intersecting_roads, crs=roads_gdf.crs)
        # Add road_geometry as regular column (not geometry)
        result_df["road_geometry"] = [
            roads_gdf.loc[r["road_idx"]].geometry for r in intersecting_roads
        ]
        return result_df
    else:
        logger.info("No road-building intersections detected")
        # Create empty GeoDataFrame with geometry column
        return gpd.GeoDataFrame(
            geometry=[],
            columns=["road_idx", "n_buildings", "intersection_length", "road_geometry"],
            crs=roads_gdf.crs,
        )


def redirect_road_parallel_offset(
    road: LineString,
    buildings: gpd.GeoDataFrame,
    offset_distance: float = 2.0,
    max_attempts: int = 4,
) -> LineString:
    """
    Redirect a road by parallel offset to avoid buildings.

    Tries offsetting in both directions (left/right) with multiple distances
    and returns the first valid offset that doesn't intersect buildings.

    Args:
        road: LineString geometry of the road
        buildings: GeoDataFrame with building polygons
        offset_distance: Base distance to offset (meters)
        max_attempts: Maximum number of offset attempts

    Returns:
        Redirected LineString, or original if redirection fails
    """
    # Find intersecting buildings to determine best offset direction
    intersecting_buildings = buildings[buildings.geometry.intersects(road)]
    if len(intersecting_buildings) == 0:
        return road  # No intersection, no need to redirect

    # Try multiple offset distances (more aggressive)
    offset_distances = [
        offset_distance,
        offset_distance * 1.5,
        offset_distance * 2.0,
        offset_distance * 3.0,
        offset_distance * 4.0,
        offset_distance * 5.0,
    ]

    # Try offsetting in both directions with different distances
    for offset_dist in offset_distances:
        for side in ["left", "right"]:
            try:
                # Create parallel offset
                offset_road = road.parallel_offset(offset_dist, side)

                # Handle MultiLineString (can happen with complex geometries)
                if hasattr(offset_road, "geoms"):
                    # Try to merge segments if possible
                    from shapely.ops import linemerge

                    try:
                        merged = linemerge(list(offset_road.geoms))
                        if hasattr(merged, "geoms"):
                            # Still multiple segments, use longest
                            offset_road = max(merged.geoms, key=lambda g: g.length)
                        else:
                            offset_road = merged
                    except Exception:
                        # If merge fails, use longest segment
                        offset_road = max(offset_road.geoms, key=lambda g: g.length)

                # Validate the offset road
                if offset_road.is_empty or not hasattr(offset_road, "coords"):
                    continue

                # Check if offset road intersects buildings
                if not buildings.geometry.intersects(offset_road).any():
                    # Also check if it's significantly different from original
                    min_distance = offset_road.distance(road)
                    if min_distance > offset_dist * 0.3:  # More lenient threshold
                        logger.debug(
                            f"  Successfully offset road by {offset_dist:.1f}m ({side})"
                        )
                        return offset_road

            except Exception as e:
                logger.debug(
                    f"  Offset failed (distance={offset_dist:.1f}m, side={side}): {e}"
                )
                continue

    # Try segment-based redirection: split road at intersections and redirect only intersecting segments
    try:
        from shapely.geometry import Point

        # Create buffer around intersecting buildings
        building_buffers = intersecting_buildings.geometry.buffer(offset_distance * 2.0)
        combined_buffer = (
            unary_union(building_buffers.values) if len(building_buffers) > 0 else None
        )

        if combined_buffer is not None and not combined_buffer.is_empty:
            # Find intersection points
            intersection_points = []
            for building in intersecting_buildings.geometry:
                intersection = road.intersection(building)
                if hasattr(intersection, "geoms"):
                    for geom in intersection.geoms:
                        if isinstance(geom, Point):
                            intersection_points.append(geom)
                elif isinstance(intersection, Point):
                    intersection_points.append(intersection)

            if len(intersection_points) > 0:
                # Try to split and redirect
                # For now, fall through to next method
                pass
    except Exception as e:
        logger.debug(f"  Segment-based redirection failed: {e}")

    # If all offsets fail, return original
    logger.warning("  Could not redirect road - using original geometry")
    return road


def redirect_road_simple_reroute(
    road: LineString, buildings: gpd.GeoDataFrame, buffer_distance: float = 2.0
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
    # Try multiple buffer distances (more aggressive)
    buffer_distances = [
        buffer_distance,
        buffer_distance * 1.5,
        buffer_distance * 2.0,
        buffer_distance * 3.0,
        buffer_distance * 4.0,
    ]

    for buf_dist in buffer_distances:
        try:
            # Find intersecting buildings
            intersecting_buildings = buildings[buildings.geometry.intersects(road)]
            if len(intersecting_buildings) == 0:
                return road  # No intersection, no need to redirect

            # Create buffer around intersecting buildings only
            building_buffers = intersecting_buildings.geometry.buffer(buf_dist)
            combined_buffer = (
                unary_union(building_buffers.values)
                if len(building_buffers) > 0
                else None
            )

            if combined_buffer is None or combined_buffer.is_empty:
                continue

            # Clip road to remove intersecting parts
            road_clipped = road.difference(combined_buffer)

            # If road was completely removed, try next buffer distance
            if road_clipped.is_empty:
                continue

            # Handle MultiLineString
            if hasattr(road_clipped, "geoms"):
                segments = list(road_clipped.geoms)
            else:
                segments = [road_clipped]

            # Filter out very short segments
            segments = [s for s in segments if s.length > 0.1]

            if len(segments) == 0:
                continue

            # If we have multiple segments, try to connect them
            if len(segments) > 1:
                # Try to connect segments by finding shortest path around buffer
                from shapely.ops import linemerge

                try:
                    # Sort segments by distance from road start
                    road_start = Point(road.coords[0])
                    segments_sorted = sorted(
                        segments, key=lambda s: Point(s.coords[0]).distance(road_start)
                    )

                    connected_segments = []
                    for i, seg in enumerate(segments_sorted):
                        connected_segments.append(seg)
                        if i < len(segments_sorted) - 1:
                            # Connect to next segment - try to go around buffer
                            end_point = Point(seg.coords[-1])
                            next_start = Point(segments_sorted[i + 1].coords[0])

                            # Try direct connection first
                            connection = LineString([end_point, next_start])
                            if not combined_buffer.intersects(connection):
                                connected_segments.append(connection)
                            else:
                                # Try going around the buffer
                                # Simple approach: offset the connection line
                                try:
                                    offset_conn = connection.parallel_offset(
                                        buf_dist, "left"
                                    )
                                    if hasattr(offset_conn, "geoms"):
                                        offset_conn = max(
                                            offset_conn.geoms, key=lambda g: g.length
                                        )
                                    if not combined_buffer.intersects(offset_conn):
                                        connected_segments.append(offset_conn)
                                    else:
                                        # Fallback to direct connection
                                        connected_segments.append(connection)
                                except Exception:
                                    connected_segments.append(connection)

                    # Merge all segments
                    merged = linemerge(connected_segments)

                    if hasattr(merged, "geoms"):
                        # Use longest segment
                        merged = max(merged.geoms, key=lambda g: g.length)

                    # Verify it doesn't intersect buildings
                    if not buildings.geometry.intersects(merged).any():
                        logger.debug(
                            f"  Successfully rerouted road with buffer {buf_dist:.1f}m"
                        )
                        return merged

                except Exception as e:
                    logger.debug(f"  Reroute merge failed: {e}")
                    # Fallback to longest segment
                    longest_seg = max(segments, key=lambda g: g.length)
                    if not buildings.geometry.intersects(longest_seg).any():
                        return longest_seg
            else:
                # Single segment - check if it's valid
                seg = segments[0]
                if not buildings.geometry.intersects(seg).any():
                    logger.debug(
                        f"  Successfully rerouted road with buffer {buf_dist:.1f}m"
                    )
                    return seg

        except Exception as e:
            logger.debug(f"  Simple reroute failed (buffer={buf_dist:.1f}m): {e}")
            continue

    # If all attempts fail, return original
    logger.warning("  Could not reroute road - using original geometry")
    return road


def redirect_roads(
    roads_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    method: str = "parallel_offset",
    offset_distance: float = 2.0,
    buffer_distance: float = 2.0,
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
    successfully_redirected_indices = []

    # Redirect each intersecting road
    for idx, intersection in intersection_info.iterrows():
        road_idx = intersection["road_idx"]
        original_road = redirected_roads.loc[road_idx, "geometry"]
        redirected_road = original_road

        # Try primary method first, then fallback to other methods
        redirected_road = original_road

        # Strategy: Try parallel offset first (usually works better), then simple reroute
        # If both fail, try with larger distances
        methods_to_try = [
            ("parallel_offset", offset_distance),
            ("simple_reroute", buffer_distance),
            ("parallel_offset", offset_distance * 2.0),  # Try larger offset
            ("simple_reroute", buffer_distance * 2.0),  # Try larger buffer
            ("parallel_offset", offset_distance * 3.0),  # Even larger
        ]

        for method_name, dist in methods_to_try:
            if redirected_road != original_road:
                break  # Already successfully redirected

            if method_name == "parallel_offset":
                redirected_road = redirect_road_parallel_offset(
                    original_road, buildings_gdf, dist
                )
            elif method_name == "simple_reroute":
                redirected_road = redirect_road_simple_reroute(
                    original_road, buildings_gdf, dist
                )

            # Check if successful
            if redirected_road != original_road:
                still_intersects = buildings_gdf.geometry.intersects(
                    redirected_road
                ).any()
                if still_intersects:
                    # Still intersects, continue trying
                    redirected_road = original_road
                else:
                    # Success!
                    break

        # Update road geometry
        redirected_roads.loc[road_idx, "geometry"] = redirected_road

        # Check if redirection was successful
        if redirected_road != original_road:
            # Verify no intersection
            still_intersects = buildings_gdf.geometry.intersects(redirected_road).any()
            if not still_intersects:
                redirected_count += 1
                successfully_redirected_indices.append(idx)
                logger.debug(f"  Road {road_idx} successfully redirected")
            else:
                logger.debug(f"  Road {road_idx} still intersects after redirection")

    logger.info(f"Redirected {redirected_count}/{len(intersection_info)} roads")

    # Return only successfully redirected roads in intersection_info
    if successfully_redirected_indices:
        redirected_info = intersection_info.loc[successfully_redirected_indices].copy()
    else:
        redirected_info = gpd.GeoDataFrame(
            columns=intersection_info.columns, crs=intersection_info.crs
        )

    return redirected_roads, redirected_info
