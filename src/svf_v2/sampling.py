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
from tqdm import tqdm
import logging

from src.svf_v2.scene import sample_dtm_at_points

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid sampling
# ---------------------------------------------------------------------------


def sample_grid_points(
    dtm_path: Path,
    footprints_gdf: gpd.GeoDataFrame,
    grid_spacing: float = 2.0,
    pedestrian_height: float = 1.5,
    buffer_around_buildings: Optional[float] = None,
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

    Returns:
        Nx3 array of observer positions (x, y, z + pedestrian_height).
    """
    with rasterio.open(dtm_path) as src:
        bounds = src.bounds  # (left, bottom, right, top)

    xs = np.arange(bounds.left, bounds.right, grid_spacing)
    ys = np.arange(bounds.bottom, bounds.top, grid_spacing)
    xx, yy = np.meshgrid(xs, ys)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    logger.info(f"Grid: {len(flat_x)} candidate points ({len(xs)}x{len(ys)})")

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

    # Optional proximity filter
    if buffer_around_buildings is not None and len(flat_x) > 0:
        from shapely.ops import unary_union

        buf = unary_union(footprints_gdf.geometry.buffer(buffer_around_buildings))
        pts_gdf2 = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in zip(flat_x, flat_y)],
            crs=footprints_gdf.crs,
        )
        near_mask = pts_gdf2.geometry.within(buf).values
        flat_x = flat_x[near_mask]
        flat_y = flat_y[near_mask]
        logger.info(
            f"  After proximity filter ({buffer_around_buildings}m): {len(flat_x)} points"
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


def sample_street_points(
    roads_path: Path,
    dtm_path: Path,
    spacing: float = 1.5,
    pedestrian_height: float = 1.5,
) -> gpd.GeoDataFrame:
    """
    Sample observer points along road centre-lines.

    Args:
        roads_path: Path to roads shapefile.
        dtm_path: Path to DTM raster.
        spacing: Distance between samples along each line (metres).
        pedestrian_height: Height above ground for observer.

    Returns:
        GeoDataFrame with columns:
            geometry, z, z_observer, street_id, distance_along
    """
    roads_gdf = gpd.read_file(roads_path)
    logger.info(f"Loaded {len(roads_gdf)} road features")

    # Reproject if needed
    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
    if roads_gdf.crs is not None and roads_gdf.crs != dtm_crs:
        logger.info(f"  Reprojecting roads from {roads_gdf.crs} to {dtm_crs}")
        roads_gdf = roads_gdf.to_crs(dtm_crs)

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

    # Sample Z from DTM
    xs = np.array([p.x for p in gdf_pts.geometry])
    ys = np.array([p.y for p in gdf_pts.geometry])
    zs = sample_dtm_at_points(dtm_path, xs, ys)

    gdf_pts["z"] = zs
    gdf_pts["z_observer"] = zs + pedestrian_height

    # Drop points with invalid elevation
    valid = np.isfinite(zs)
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
