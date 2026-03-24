"""
Point generation for SVF computation: grid, street, and facade sampling.

All coordinates are in world CRS (EPSG:31983).
"""

import numpy as np
import geopandas as gpd
import rasterio
from pathlib import Path
from typing import Optional
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely import STRtree
from shapely.ops import nearest_points
from tqdm import tqdm
import logging

from src.svf_v2.scene import sample_dtm_at_points

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTM coverage validation
# ---------------------------------------------------------------------------


def _check_dtm_coverage(
    dtm_path: Path,
    data_gdf: gpd.GeoDataFrame,
    label: str = "data",
) -> float:
    """Check what percentage of a GeoDataFrame's extent is covered by the DTM.

    Logs a warning when coverage is below 95%, an error below 50%.
    Returns the coverage percentage.
    """
    with rasterio.open(dtm_path) as src:
        db = src.bounds

    tb = data_gdf.total_bounds  # [minx, miny, maxx, maxy]
    data_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
    if data_area <= 0:
        return 100.0

    ix_w = max(0, min(db.right, tb[2]) - max(db.left, tb[0]))
    ix_h = max(0, min(db.top, tb[3]) - max(db.bottom, tb[1]))
    pct = (ix_w * ix_h) / data_area * 100

    if pct < 50:
        logger.error(
            f"DTM covers only {pct:.0f}% of {label} extent — "
            f"most data will be lost. Re-export DTM to cover full area."
        )
    elif pct < 95:
        logger.warning(
            f"DTM covers {pct:.0f}% of {label} extent — "
            f"some data may be lost near edges."
        )
    else:
        logger.info(f"DTM covers {pct:.0f}% of {label} extent.")

    return pct


# ---------------------------------------------------------------------------
# Grid sampling
# ---------------------------------------------------------------------------


def sample_grid_points(
    dtm_path: Path,
    footprints_gdf: gpd.GeoDataFrame,
    grid_spacing: float = 2.0,
    pedestrian_height: float = 1.5,
    buffer_around_buildings: Optional[float] = None,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
) -> np.ndarray:
    """
    Generate a regular grid of observer points excluding building interiors.

    Args:
        dtm_path: Path to DTM raster.
        footprints_gdf: Building footprints GeoDataFrame (world CRS).
        grid_spacing: Grid cell size in metres.
        pedestrian_height: Height above ground for observer.
        buffer_around_buildings: If set, only keep points within this distance
            of at least one building (reduces computation in open areas).
        boundary_gdf: If set, clip grid domain to this boundary polygon.

    Returns:
        Nx3 array of observer positions (x, y, z + pedestrian_height).
    """
    _check_dtm_coverage(dtm_path, footprints_gdf, label="building footprints")

    with rasterio.open(dtm_path) as src:
        bounds = src.bounds  # (left, bottom, right, top)

    xs = np.arange(bounds.left, bounds.right, grid_spacing)
    ys = np.arange(bounds.bottom, bounds.top, grid_spacing)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    logger.info(f"Grid: {len(flat_x)} candidate points ({len(xs)}x{len(ys)})")

    # Optional boundary clip (before expensive spatial join)
    if boundary_gdf is not None and len(boundary_gdf) > 0 and len(flat_x) > 0:
        from shapely.ops import unary_union

        boundary_poly = unary_union(boundary_gdf.geometry.values)
        pts_arr = gpd.points_from_xy(flat_x, flat_y)
        inside_boundary = np.array([boundary_poly.contains(p) for p in pts_arr])
        flat_x = flat_x[inside_boundary]
        flat_y = flat_y[inside_boundary]
        logger.info(f"  After boundary clip: {len(flat_x)} points")

    # Exclude points inside building footprints via spatial join
    pts_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(flat_x, flat_y)],
        crs=footprints_gdf.crs,
    )
    joined = gpd.sjoin(
        pts_gdf, footprints_gdf[["geometry"]], how="left", predicate="within"
    )
    # Deduplicate (point may touch multiple buildings)
    inside_mask = ~joined["index_right"].isna()
    if joined.index.duplicated().any():
        inside_mask = inside_mask.groupby(inside_mask.index).any()
    ground_mask = ~inside_mask.values[: len(flat_x)]

    flat_x = flat_x[ground_mask]
    flat_y = flat_y[ground_mask]
    logger.info(f"  After building mask: {len(flat_x)} ground points")

    # Proximity filter: keep only points within buffer distance of a building
    if buffer_around_buildings is not None and len(flat_x) > 0:
        import shapely

        tree = STRtree(footprints_gdf.geometry.values)
        pts = gpd.points_from_xy(flat_x, flat_y)
        nearest_idx = tree.nearest(pts)
        geom_arr = np.asarray(footprints_gdf.geometry.values)
        distances = shapely.distance(pts, geom_arr[nearest_idx])
        near_mask = distances <= buffer_around_buildings
        n_before = len(flat_x)
        flat_x = flat_x[near_mask]
        flat_y = flat_y[near_mask]
        logger.info(
            f"  After proximity filter ({buffer_around_buildings}m): "
            f"{len(flat_x)} points ({n_before - len(flat_x)} removed)"
        )

    # Sample Z from DTM
    zs = sample_dtm_at_points(dtm_path, flat_x, flat_y)
    valid = np.isfinite(zs)
    flat_x = flat_x[valid]
    flat_y = flat_y[valid]
    zs = zs[valid]

    observers = np.column_stack([flat_x, flat_y, zs + pedestrian_height])
    logger.info(f"  Final grid observers: {len(observers)}")
    return observers


# ---------------------------------------------------------------------------
# Street sampling
# ---------------------------------------------------------------------------


def _sample_points_along_line(line: LineString, spacing: float) -> list:
    """Return [(Point, distance_along), ...] at regular intervals."""
    length = line.length
    if length < spacing:
        pt = line.interpolate(length / 2)
        return [(pt, length / 2)]

    pts = []
    d = 0.0
    while d <= length:
        pts.append((line.interpolate(d), d))
        d += spacing
    # Include endpoint if not already close
    if pts and pts[-1][0].distance(line.interpolate(length)) > spacing / 2:
        pts.append((line.interpolate(length), length))
    return pts


def _nearest_edge_outward_normal(point: Point, polygon: Polygon) -> np.ndarray:
    """
    Outward normal at the nearest edge of *polygon* to *point*.

    Fallback for when point sits exactly on the boundary (zero distance to
    nearest_points), so the direction vector is degenerate.
    """
    coords = np.array(polygon.exterior.coords)
    n_verts = len(coords) - 1
    best_dist = np.inf
    best_normal = np.array([1.0, 0.0])
    px, py = point.x, point.y

    for i in range(n_verts):
        ax, ay = coords[i, 0], coords[i, 1]
        bx, by = coords[i + 1, 0], coords[i + 1, 1]
        edge_pt = LineString([(ax, ay), (bx, by)]).interpolate(
            LineString([(ax, ay), (bx, by)]).project(point)
        )
        d = point.distance(edge_pt)
        if d < best_dist:
            best_dist = d
            best_normal = _outward_normal_2d(ax, ay, bx, by)

    norm = np.linalg.norm(best_normal)
    if norm < 1e-12:
        return np.array([1.0, 0.0])
    return best_normal / norm


def _offset_points_outside_buildings(
    gdf_pts: gpd.GeoDataFrame,
    footprints_gdf: gpd.GeoDataFrame,
    safety_margin: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Move street sample points that fall inside building footprints to the
    nearest location outside, plus a safety margin.

    Adds columns: ``was_offset``, ``offset_distance``, ``original_x``, ``original_y``.
    """
    gdf_pts = gdf_pts.copy()
    gdf_pts["was_offset"] = False
    gdf_pts["offset_distance"] = 0.0
    gdf_pts["original_x"] = np.array([p.x for p in gdf_pts.geometry])
    gdf_pts["original_y"] = np.array([p.y for p in gdf_pts.geometry])

    if footprints_gdf is None or len(footprints_gdf) == 0:
        return gdf_pts

    # Explode MultiPolygons for STRtree
    exploded = footprints_gdf.explode(index_parts=False)
    building_polys = list(exploded.geometry.values)
    tree = STRtree(building_polys)

    def _collides(point):
        """Check if point is inside any building polygon."""
        idxs = tree.query(point)
        return any(building_polys[i].contains(point) for i in idxs)

    # Identify trapped points via spatial join
    joined = gpd.sjoin(
        gdf_pts[["geometry"]], footprints_gdf[["geometry"]],
        how="inner", predicate="within",
    )
    trapped_indices = joined.index.unique()

    if len(trapped_indices) == 0:
        logger.info("  No street points inside buildings — no offsets needed")
        return gdf_pts

    logger.info(
        f"  Offsetting {len(trapped_indices)}/{len(gdf_pts)} points "
        f"inside buildings (margin={safety_margin}m)"
    )

    n_unresolvable = 0
    for idx in trapped_indices:
        pt = gdf_pts.at[idx, "geometry"]
        px, py = pt.x, pt.y

        # Find containing building(s) via STRtree
        containing = [building_polys[i] for i in tree.query(pt)
                       if building_polys[i].contains(pt)]
        if not containing:
            continue
        bldg = containing[0]

        # Nearest point on boundary
        _, boundary_pt = nearest_points(pt, bldg.exterior)
        dx = boundary_pt.x - px
        dy = boundary_pt.y - py
        dist = np.hypot(dx, dy)

        if dist < 1e-9:
            # Point on boundary — use outward normal of nearest edge
            direction = _nearest_edge_outward_normal(pt, bldg)
        else:
            direction = np.array([dx, dy]) / dist

        # Offset to boundary + safety margin
        new_x = boundary_pt.x + safety_margin * direction[0]
        new_y = boundary_pt.y + safety_margin * direction[1]
        new_pt = Point(new_x, new_y)

        # Collision check
        collides = _collides(new_pt)

        if collides:
            # Try 8 cardinal/intercardinal directions at increasing radii
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            resolved = False
            for radius in [safety_margin, safety_margin * 2, safety_margin * 4]:
                best_disp = np.inf
                best_candidate = None
                for angle in angles:
                    cx = boundary_pt.x + radius * np.cos(angle)
                    cy = boundary_pt.y + radius * np.sin(angle)
                    candidate = Point(cx, cy)
                    if not _collides(candidate):
                        disp = pt.distance(candidate)
                        if disp < best_disp:
                            best_disp = disp
                            best_candidate = candidate
                if best_candidate is not None:
                    new_pt = best_candidate
                    new_x, new_y = new_pt.x, new_pt.y
                    resolved = True
                    break

            if not resolved:
                logger.warning(
                    f"  Point {idx} at ({px:.1f}, {py:.1f}) — all directions blocked, keeping original"
                )
                n_unresolvable += 1
                continue

        gdf_pts.at[idx, "geometry"] = new_pt
        gdf_pts.at[idx, "was_offset"] = True
        gdf_pts.at[idx, "offset_distance"] = pt.distance(new_pt)

    n_offset = int(gdf_pts["was_offset"].sum())
    logger.info(
        f"  Offset complete: {n_offset} moved, {n_unresolvable} unresolvable"
    )
    return gdf_pts


def sample_street_points(
    roads_path: Path,
    dtm_path: Path,
    spacing: float = 1.5,
    pedestrian_height: float = 1.5,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
    building_safety_margin: float = 0.5,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """
    Sample observer points along road centre-lines.

    When a *boundary_gdf* is provided, road geometries are clipped to the
    boundary **before** sampling.  This prevents observation points on
    highways or avenues outside the community from biasing SVF statistics.

    Args:
        roads_path: Path to roads shapefile.
        dtm_path: Path to DTM raster.
        spacing: Distance between samples along each line (metres).
        pedestrian_height: Height above ground for observer.
        footprints_gdf: Building footprints GeoDataFrame.  If provided,
            points that fall inside buildings are offset to the nearest
            exterior location plus *building_safety_margin*.
        building_safety_margin: Buffer distance (m) beyond the building
            boundary when offsetting trapped points.
        boundary_gdf: If set, clip road geometries to this boundary polygon
            before sampling, eliminating edge-effect bias from external roads.

    Returns:
        GeoDataFrame with columns:
            geometry, z, z_observer, street_id, distance_along
            (and was_offset, offset_distance, original_x, original_y
             when footprints_gdf is supplied)
    """
    roads_gdf = gpd.read_file(roads_path)
    logger.info(f"Loaded {len(roads_gdf)} road features")

    # Reproject if needed
    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
    if roads_gdf.crs is not None and roads_gdf.crs != dtm_crs:
        logger.info(f"  Reprojecting roads from {roads_gdf.crs} to {dtm_crs}")
        roads_gdf = roads_gdf.to_crs(dtm_crs)

    # Clip roads to boundary — prevents sampling on external highways/avenues
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        from shapely.ops import unary_union

        boundary_poly = unary_union(boundary_gdf.geometry.values)
        n_before = len(roads_gdf)
        roads_gdf = gpd.clip(roads_gdf, boundary_poly)
        # clip can produce MultiLineString fragments; explode to LineStrings
        roads_gdf = roads_gdf.explode(index_parts=False).reset_index(drop=True)
        # Drop tiny fragments from clipping artifacts
        roads_gdf = roads_gdf[roads_gdf.geometry.length > spacing]
        logger.info(
            f"  Clipped roads to boundary: {n_before} -> {len(roads_gdf)} segments"
        )

    rows = []
    for idx, row in tqdm(
        roads_gdf.iterrows(), total=len(roads_gdf), desc="Sampling streets"
    ):
        geom = row.geometry
        if not isinstance(geom, LineString):
            continue
        for pt, dist in _sample_points_along_line(geom, spacing):
            rows.append(
                {
                    "geometry": pt,
                    "street_id": idx,
                    "distance_along": dist,
                }
            )

    if not rows:
        logger.warning("No street sample points generated!")
        return gpd.GeoDataFrame(
            columns=["geometry", "z", "z_observer", "street_id", "distance_along"]
        )

    gdf_pts = gpd.GeoDataFrame(rows, crs=roads_gdf.crs)

    _check_dtm_coverage(dtm_path, gdf_pts, label="road network")

    # Offset points trapped inside buildings (before Z sampling)
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        gdf_pts = _offset_points_outside_buildings(
            gdf_pts, footprints_gdf, safety_margin=building_safety_margin
        )

    # Sample Z from DTM
    xs = np.array([p.x for p in gdf_pts.geometry])
    ys = np.array([p.y for p in gdf_pts.geometry])
    zs = sample_dtm_at_points(dtm_path, xs, ys)

    gdf_pts["z"] = zs
    gdf_pts["z_observer"] = zs + pedestrian_height

    # Drop points with invalid elevation
    valid = np.isfinite(zs)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        logger.warning(
            f"  Dropped {n_dropped}/{len(zs)} street points outside DTM bounds"
        )
    gdf_pts = gdf_pts[valid].reset_index(drop=True)

    logger.info(f"  Street sample points: {len(gdf_pts)}")
    return gdf_pts


# ---------------------------------------------------------------------------
# Facade sampling
# ---------------------------------------------------------------------------


def _outward_normal_2d(ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    """
    Outward normal for edge A->B of a CCW-wound exterior ring.

    For CCW winding (Shapely default), rotating edge vector 90 deg clockwise
    gives the outward normal:  (dy, -dx) normalised.
    """
    dx = bx - ax
    dy = by - ay
    length = np.hypot(dx, dy)
    if length < 1e-12:
        return np.array([0.0, 0.0])
    return np.array([dy, -dx]) / length


def sample_facade_points(
    footprints_gdf: gpd.GeoDataFrame,
    dtm_path: Path,
    height_field: str = "altura",
    base_field: str = "base",
    vertical_spacing: float = 1.0,
    horizontal_spacing: float = 1.0,
    inset: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Sample points on building facades (exterior walls).

    For each building edge, points are generated in a grid across the wall
    surface, offset slightly outward from the wall face.

    Args:
        footprints_gdf: Building footprints with ``base`` and ``altura`` fields.
        dtm_path: DTM raster for fallback base elevation.
        height_field: Column with extrusion height (default ``"altura"``).
        base_field: Column with base elevation (default ``"base"``).
        vertical_spacing: Vertical distance between sample rows (m).
        horizontal_spacing: Horizontal distance along each edge (m).
        inset: Outward offset from wall surface (m).

    Returns:
        GeoDataFrame with columns:
            geometry (Point), x, y, z, normal_x, normal_y, normal_z,
            building_id, facade_azimuth, height_above_ground
    """
    from src.svf_v2.scene import sample_dtm_at_points as _batch_z

    rows = []

    for bldg_idx, brow in tqdm(
        footprints_gdf.iterrows(), total=len(footprints_gdf), desc="Facade sampling"
    ):
        geom = brow.geometry
        if geom is None or geom.is_empty:
            continue

        h = brow.get(height_field, np.nan)
        if not np.isfinite(h) or h <= 0:
            continue

        base = brow.get(base_field, np.nan)
        if not np.isfinite(base):
            # Fallback: sample from DTM at centroid
            cx, cy = geom.centroid.x, geom.centroid.y
            base_arr = _batch_z(dtm_path, np.array([cx]), np.array([cy]))
            base = float(base_arr[0]) if np.isfinite(base_arr[0]) else np.nan
        if not np.isfinite(base):
            continue

        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

        for poly in polys:
            if not isinstance(poly, Polygon):
                continue
            coords = np.array(poly.exterior.coords)
            n_verts = len(coords) - 1  # last == first

            for i in range(n_verts):
                ax, ay = coords[i, 0], coords[i, 1]
                bx, by = coords[i + 1, 0], coords[i + 1, 1]

                edge_len = np.hypot(bx - ax, by - ay)
                if edge_len < 0.1:
                    continue

                normal_2d = _outward_normal_2d(ax, ay, bx, by)
                nx, ny = normal_2d
                # Facade azimuth: angle from north (CW positive)
                azimuth = (np.degrees(np.arctan2(nx, ny)) + 360) % 360

                # Horizontal samples along edge
                n_horiz = max(1, int(np.ceil(edge_len / horizontal_spacing)))
                for hi in range(n_horiz):
                    t = (hi + 0.5) / n_horiz
                    ex = ax + t * (bx - ax) + inset * nx
                    ey = ay + t * (by - ay) + inset * ny

                    # Vertical samples
                    n_vert = max(1, int(np.ceil(h / vertical_spacing)))
                    for vi in range(n_vert):
                        z_offset = (vi + 0.5) * (h / n_vert)
                        ez = base + z_offset

                        rows.append(
                            {
                                "geometry": Point(ex, ey),
                                "x": ex,
                                "y": ey,
                                "z": ez,
                                "normal_x": nx,
                                "normal_y": ny,
                                "normal_z": 0.0,
                                "building_id": bldg_idx,
                                "facade_azimuth": azimuth,
                                "height_above_ground": z_offset,
                            }
                        )

    if not rows:
        logger.warning("No facade sample points generated!")
        cols = [
            "geometry",
            "x",
            "y",
            "z",
            "normal_x",
            "normal_y",
            "normal_z",
            "building_id",
            "facade_azimuth",
            "height_above_ground",
        ]
        return gpd.GeoDataFrame(columns=cols, crs=footprints_gdf.crs)

    gdf = gpd.GeoDataFrame(rows, crs=footprints_gdf.crs)
    logger.info(f"  Facade sample points: {len(gdf)}")
    return gdf
