"""Morphology metrics beyond the basic 6 indicators."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import unary_union, voronoi_diagram
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


def _get_polygon_parts(geom: Polygon | MultiPolygon) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    return []


def _largest_polygon(geom: Polygon | MultiPolygon) -> Polygon | None:
    parts = _get_polygon_parts(geom)
    if not parts:
        return None
    return max(parts, key=lambda p: p.area)


def _safe_length(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return geom.length


def _safe_area(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return geom.area


def _minimum_rotated_rectangle_axes(
    geom: Polygon | MultiPolygon,
) -> tuple[float, float, float]:
    """
    Return (L_major, L_minor, angle_rad) from the minimum rotated rectangle.
    Angle is the orientation of the major axis.
    """
    poly = _largest_polygon(geom)
    if poly is None:
        return (np.nan, np.nan, np.nan)
    mrr = poly.minimum_rotated_rectangle
    if mrr is None or mrr.is_empty:
        return (np.nan, np.nan, np.nan)

    # Degenerate polygons may produce a Point or LineString instead of a Polygon
    if not isinstance(mrr, Polygon):
        return (0.0, 0.0, 0.0)

    coords = list(mrr.exterior.coords)
    if len(coords) < 4:
        return (np.nan, np.nan, np.nan)

    # Remove closing coordinate
    coords = coords[:-1]
    edges = []
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        edges.append((length, dx, dy))

    # Four edges: two pairs of equal lengths
    edges_sorted = sorted(edges, key=lambda e: e[0], reverse=True)
    L_major = edges_sorted[0][0]
    L_minor = edges_sorted[-1][0]

    # Orientation from the longest edge
    dx, dy = edges_sorted[0][1], edges_sorted[0][2]
    angle = math.atan2(dy, dx)
    return (L_major, L_minor, angle)


def _count_corners(geom: Polygon | MultiPolygon) -> int:
    parts = _get_polygon_parts(geom)
    if not parts:
        return 0
    count = 0
    for poly in parts:
        coords = list(poly.exterior.coords)
        if len(coords) > 1:
            count += max(len(coords) - 1, 0)  # exclude closing coordinate
    return count


def _interior_angles(geom: Polygon | MultiPolygon) -> list[float]:
    """Return interior angles in degrees for polygon exterior rings."""
    parts = _get_polygon_parts(geom)
    angles = []
    for poly in parts:
        coords = list(poly.exterior.coords)
        if len(coords) < 4:
            continue
        coords = coords[:-1]  # remove closing
        n = len(coords)
        for i in range(n):
            prev_pt = coords[(i - 1) % n]
            curr_pt = coords[i]
            next_pt = coords[(i + 1) % n]
            v1 = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
            v2 = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
            norm1 = math.hypot(v1[0], v1[1])
            norm2 = math.hypot(v2[0], v2[1])
            if norm1 == 0 or norm2 == 0:
                continue
            cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
            cos_theta = max(min(cos_theta, 1.0), -1.0)
            angle = math.degrees(math.acos(cos_theta))
            angles.append(angle)
    return angles


def _shared_wall_length(geom_a: Polygon | MultiPolygon, geom_b: Polygon | MultiPolygon) -> float:
    if geom_a is None or geom_b is None:
        return 0.0
    if geom_a.is_empty or geom_b.is_empty:
        return 0.0
    inter = geom_a.boundary.intersection(geom_b.boundary)
    return _safe_length(inter)


def _build_neighbor_indices(
    geoms: Iterable[Polygon | MultiPolygon],
) -> tuple[list[list[int]], np.ndarray]:
    """
    Build neighbor indices and shared wall lengths.

    Returns:
        neighbors: list of neighbor index lists for each geometry
        shared_lengths: array of shared wall lengths per geometry
    """
    geoms_list = list(geoms)
    tree = STRtree(geoms_list)
    neighbors = [[] for _ in geoms_list]
    shared_lengths = np.zeros(len(geoms_list), dtype=float)

    for idx, geom in enumerate(geoms_list):
        candidates = tree.query(geom)
        for cand_idx in candidates:
            if cand_idx == idx:
                continue
            cand_geom = geoms_list[cand_idx]
            if not geom.intersects(cand_geom):
                continue
            neighbors[idx].append(cand_idx)
            shared_lengths[idx] += _shared_wall_length(geom, cand_geom)

    # Deduplicate neighbors
    neighbors = [sorted(set(n)) for n in neighbors]
    return neighbors, shared_lengths


def _mean_distance_between_buildings(centroids: np.ndarray) -> float:
    n = len(centroids)
    if n < 2:
        return np.nan
    try:
        from scipy.spatial.distance import pdist

        dists = pdist(centroids)
        return (2.0 * dists.sum()) / (n * n)
    except Exception:
        # Fallback O(n^2)
        total = 0.0
        for i in range(n):
            for j in range(n):
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                total += math.hypot(dx, dy)
        return total / (n * n)


def _build_voronoi_cells(
    centroids: np.ndarray, envelope: Polygon | None, buffer_distance: float
) -> list[Polygon]:
    if len(centroids) == 0:
        return []
    points = [Point(pt[0], pt[1]) for pt in centroids]
    if envelope is None:
        coords = np.array([[p.x, p.y] for p in points])
        minx, miny = coords.min(axis=0)
        maxx, maxy = coords.max(axis=0)
        envelope = Polygon(
            [
                (minx - buffer_distance, miny - buffer_distance),
                (maxx + buffer_distance, miny - buffer_distance),
                (maxx + buffer_distance, maxy + buffer_distance),
                (minx - buffer_distance, maxy + buffer_distance),
            ]
        )

    vor = voronoi_diagram(MultiPoint(points), envelope=envelope, edges=False)
    if isinstance(vor, (Polygon, MultiPolygon)):
        cells = _get_polygon_parts(vor)
    elif isinstance(vor, GeometryCollection):
        cells = [g for g in vor.geoms if isinstance(g, Polygon)]
    else:
        cells = []
    return cells


def _assign_voronoi_cells_to_points(
    cells: list[Polygon], centroids: np.ndarray
) -> list[Polygon | None]:
    if not cells:
        return [None] * len(centroids)
    tree = STRtree(cells)
    assignments: list[Polygon | None] = [None] * len(centroids)
    for idx, (x, y) in enumerate(centroids):
        point = Point(x, y)
        candidates = tree.query(point)
        for cand_idx in candidates:
            cell = cells[cand_idx]
            if cell.contains(point):
                assignments[idx] = cell
                break
        if assignments[idx] is None:
            # fallback: nearest cell
            assignments[idx] = min(cells, key=lambda c: c.distance(point))
    return assignments


def calculate_morphology_metrics(
    buildings: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame | None = None,
    street_buffer: float = 1.0,
    site_boundary: Polygon | None = None,
    voronoi_buffer: float = 50.0,
    weight_mode: str = "adjacent",
) -> gpd.GeoDataFrame:
    """Calculate morphology metrics for building footprints.

    Computes 25+ morphometric indicators covering shape, compactness,
    adjacency, tessellation, and spatial distribution for each building.

    Args:
        buildings: GeoDataFrame with building footprint geometries.
            Geometries must be Polygon or MultiPolygon; degenerate,
            empty, or invalid geometries are filtered out automatically.
        streets: Optional GeoDataFrame of street centre-lines used
            to compute the facade ratio metric. If *None*, the
            ``facade_ratio`` column is filled with NaN.
        street_buffer: Buffer distance (m) applied to street geometries
            when computing facade ratio.
        site_boundary: Optional Polygon defining the study-area
            boundary. Used for covered-area ratio and as the Voronoi
            diagram envelope. If *None*, the convex hull of all
            building footprints is used.
        voronoi_buffer: Buffer distance (m) added around the site
            boundary when constructing Voronoi tessellation cells.
        weight_mode: Neighbour weighting strategy for the
            average-weighted-distance metric. Currently only
            ``'adjacent'`` is supported.

    Returns:
        GeoDataFrame with the original columns plus the following
        metric columns (NaN where a metric cannot be computed):

        *Shape / axes*:
            area, perimeter, L_major, L_minor, orientation,
            longest_axis_length

        *Compactness*:
            shape_index, compactness_weighted_axis, convexity,
            equivalent_rectangular_index, rectangularity,
            square_compactness, elongation, squareness

        *Adjacency*:
            shared_walls, perimeter_wall, building_adjacency, num_corners

        *Tessellation*:
            tessellation_area, cell_alignment, tessellation_neighbors, cwt

        *Spatial distribution*:
            mean_distance_between_buildings, covered_area_ratio,
            fractal_dimension, average_weighted_distance, facade_ratio
    """
    gdf = buildings.copy()
    if gdf.empty:
        return gdf

    # --- Filter degenerate / invalid geometries --------------------------
    _n_before = len(gdf)
    valid_mask = ~gdf.geometry.is_empty & gdf.geometry.is_valid & (gdf.geometry.area >= 1e-6)
    n_filtered = int((~valid_mask).sum())
    if n_filtered > 0:
        logger.warning(
            "Filtered %d degenerate/invalid geometries (empty, invalid, "
            "or area < 1e-6 m²); %d buildings remain",
            n_filtered,
            int(valid_mask.sum()),
        )
        gdf = gdf.loc[valid_mask].copy()
        gdf = gdf.reset_index(drop=True)

    if gdf.empty:
        return gdf

    geoms = list(gdf.geometry)

    # Base geometry properties (ensure available)
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length

    # Major/minor axes + orientation
    axes = [_minimum_rotated_rectangle_axes(geom) for geom in geoms]
    gdf["L_major"] = [a[0] for a in axes]
    gdf["L_minor"] = [a[1] for a in axes]
    gdf["orientation"] = [a[2] for a in axes]
    gdf["longest_axis_length"] = gdf[["L_major", "L_minor"]].max(axis=1)

    # Shape/compactness metrics
    gdf["shape_index"] = np.where(gdf["area"] > 0, gdf["perimeter"] / np.sqrt(gdf["area"]), np.nan)
    gdf["compactness_weighted_axis"] = np.where(
        (gdf["L_minor"] > 0) & (gdf["perimeter"] > 0),
        (gdf["L_major"] / gdf["L_minor"]) * (4 * math.pi * gdf["area"] / (gdf["perimeter"] ** 2)),
        np.nan,
    )
    gdf["convexity"] = gdf.geometry.apply(
        lambda geom: (
            _safe_area(geom) / _safe_area(geom.convex_hull)
            if _safe_area(geom.convex_hull) > 0
            else np.nan
        )
    )

    # Shared walls + adjacency
    neighbors, shared_lengths = _build_neighbor_indices(geoms)
    gdf["shared_walls"] = shared_lengths
    gdf["perimeter_wall"] = gdf["perimeter"] - gdf["shared_walls"]
    gdf["building_adjacency"] = [len(n) for n in neighbors]

    # Number of corners
    gdf["num_corners"] = gdf.geometry.apply(_count_corners)

    # Equivalent rectangular index (min bounding rectangle area)
    gdf["equivalent_rectangular_index"] = gdf.geometry.apply(
        lambda geom: (
            _safe_area(geom) / _safe_area(_largest_polygon(geom).minimum_rotated_rectangle)
            if _largest_polygon(geom) is not None
            else np.nan
        )
    )

    # Rectangularity
    gdf["rectangularity"] = np.where(
        (gdf["L_major"] > 0) & (gdf["L_minor"] > 0),
        gdf["area"] / (gdf["L_major"] * gdf["L_minor"]),
        np.nan,
    )

    # Squareness
    def _squareness(geom):
        angles = _interior_angles(geom)
        if not angles:
            return np.nan
        return 1.0 - (np.mean([abs(a - 90.0) for a in angles]) / 90.0)

    gdf["squareness"] = gdf.geometry.apply(_squareness)

    # Square compactness + elongation
    gdf["square_compactness"] = np.where(
        gdf["longest_axis_length"] > 0,
        gdf["area"] / (gdf["longest_axis_length"] ** 2),
        np.nan,
    )
    gdf["elongation"] = np.where(gdf["L_minor"] > 0, gdf["L_major"] / gdf["L_minor"], np.nan)

    # Spatial distribution metrics
    centroids = np.array([[pt.x, pt.y] for pt in gdf.geometry.centroid])
    gdf["mean_distance_between_buildings"] = _mean_distance_between_buildings(centroids)

    # Covered area ratio (CAR)
    if site_boundary is None:
        site_boundary = unary_union(gdf.geometry).convex_hull
    site_area = _safe_area(site_boundary)
    if site_area > 0:
        gdf["covered_area_ratio"] = gdf["area"].sum() / site_area
    else:
        gdf["covered_area_ratio"] = np.nan

    # Tessellation metrics (Voronoi)
    cells = _build_voronoi_cells(centroids, site_boundary, voronoi_buffer)
    assignments = _assign_voronoi_cells_to_points(cells, centroids)
    gdf["tessellation_area"] = [
        _safe_area(cell) if cell is not None else np.nan for cell in assignments
    ]

    # Cell alignment (CA)
    cell_angles = []
    for cell in assignments:
        if cell is None:
            cell_angles.append(np.nan)
        else:
            _, _, angle = _minimum_rotated_rectangle_axes(cell)
            cell_angles.append(angle)
    cell_angles_arr = np.array([a for a in cell_angles if not np.isnan(a)])
    if len(cell_angles_arr) > 0:
        mean_angle = math.atan2(np.sin(cell_angles_arr).mean(), np.cos(cell_angles_arr).mean())
        ca = np.mean(np.cos(cell_angles_arr - mean_angle))
    else:
        ca = np.nan
    gdf["cell_alignment"] = ca

    # Tessellation neighbors (TN)
    if cells:
        cell_tree = STRtree(cells)
        cell_neighbors = []
        for cell in assignments:
            if cell is None:
                cell_neighbors.append(0)
                continue
            candidates = cell_tree.query(cell)
            count = 0
            for cand_idx in candidates:
                cand = cells[cand_idx]
                if cand.equals(cell):
                    continue
                if cell.touches(cand):
                    inter = cell.boundary.intersection(cand.boundary)
                    if _safe_length(inter) > 0:
                        count += 1
            cell_neighbors.append(count)
        gdf["tessellation_neighbors"] = cell_neighbors
    else:
        gdf["tessellation_neighbors"] = np.nan

    # Compactness-weighted axis of each tessellation (CWT)
    gdf["cwt"] = [
        (_minimum_rotated_rectangle_axes(cell)[0] / math.sqrt(_safe_area(cell)))
        if cell is not None and _safe_area(cell) > 0
        else np.nan
        for cell in assignments
    ]

    # Topological features
    gdf["fractal_dimension"] = np.where(
        (gdf["perimeter"] > 0) & (gdf["area"] > 0),
        (2 * np.log(gdf["perimeter"])) / np.log(gdf["area"]),
        np.nan,
    )

    # Average weighted distance (d_w)
    if weight_mode == "adjacent":
        d_w = np.full(len(gdf), np.nan)
        for idx, neigh in enumerate(neighbors):
            if not neigh:
                continue
            distances = []
            for j in neigh:
                dx = centroids[idx][0] - centroids[j][0]
                dy = centroids[idx][1] - centroids[j][1]
                distances.append(math.hypot(dx, dy))
            d_w[idx] = np.sum(distances) / len(neigh)
        gdf["average_weighted_distance"] = d_w
    else:
        gdf["average_weighted_distance"] = np.nan

    # Facade ratio (optional, requires streets)
    if streets is not None and not streets.empty:
        street_buffer_geom = streets.geometry.buffer(street_buffer).unary_union

        def _facade_ratio(geom):
            if geom is None or geom.is_empty:
                return np.nan
            boundary = geom.boundary
            facade_len = boundary.intersection(street_buffer_geom).length
            perim = boundary.length
            if perim == 0:
                return np.nan
            return facade_len / perim

        gdf["facade_ratio"] = gdf.geometry.apply(_facade_ratio)
    else:
        gdf["facade_ratio"] = np.nan

    return gdf
