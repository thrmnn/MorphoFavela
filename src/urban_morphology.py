"""Zone-level urban morphology metrics.

Unlike ``morphology_metrics.py`` which computes per-building descriptors,
this module operates at the **analysis-zone** scale (grid cells, hexagons,
or arbitrary polygons) and returns aggregate density / variability indicators
commonly used in urban climate and urban form studies.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, box

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Analysis-zone generation
# ---------------------------------------------------------------------------


def _hexagon(cx: float, cy: float, size: float) -> Polygon:
    """Return a flat-topped regular hexagon centred on (*cx*, *cy*).

    *size* is the circumradius (centre to vertex).
    """
    angles_deg = [0, 60, 120, 180, 240, 300]
    coords = [
        (cx + size * math.cos(math.radians(a)), cy + size * math.sin(math.radians(a)))
        for a in angles_deg
    ]
    return Polygon(coords)


def create_analysis_zones(
    buildings: gpd.GeoDataFrame,
    method: str = "grid",
    cell_size: float = 50.0,
) -> gpd.GeoDataFrame:
    """Generate analysis zones covering the buildings' extent.

    Parameters
    ----------
    buildings : GeoDataFrame
        Building footprints used to derive the spatial extent.
    method : ``"grid"`` | ``"hexgrid"``
        ``"grid"`` creates square cells; ``"hexgrid"`` creates hexagons via a
        row-offset pattern.
    cell_size : float
        Side length (metres) for square cells or circumradius for hexagons.

    Returns
    -------
    GeoDataFrame
        Columns: ``zone_id`` (int), ``geometry`` (Polygon), ``zone_area`` (float).
    """
    minx, miny, maxx, maxy = buildings.total_bounds
    # Expand extent by one cell_size on each side
    minx -= cell_size
    miny -= cell_size
    maxx += cell_size
    maxy += cell_size

    polygons: list[Polygon] = []

    if method == "grid":
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                polygons.append(box(x, y, x + cell_size, y + cell_size))
                y += cell_size
            x += cell_size

    elif method == "hexgrid":
        # Flat-topped hexagonal grid with row-offset pattern
        size = cell_size
        horiz = size * math.sqrt(3)
        vert = size * 1.5
        row = 0
        cy = miny
        while cy < maxy:
            offset = horiz / 2 if row % 2 == 1 else 0.0
            cx = minx + offset
            while cx < maxx:
                polygons.append(_hexagon(cx, cy, size))
                cx += horiz
            cy += vert
            row += 1
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'grid' or 'hexgrid'.")

    zones = gpd.GeoDataFrame(
        {
            "zone_id": range(len(polygons)),
            "geometry": polygons,
        },
        crs=buildings.crs,
    )
    zones["zone_area"] = zones.geometry.area
    logger.info("Created %d analysis zones (method=%s).", len(zones), method)
    return zones


# ---------------------------------------------------------------------------
# 2. Building Coverage Ratio (BCR)
# ---------------------------------------------------------------------------


def compute_bcr(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Building Coverage Ratio per zone.

    BCR = sum(intersection area of building footprints with zone) / zone_area.
    """
    zones = zones.copy()

    if buildings.empty:
        zones["bcr"] = 0.0
        return zones

    joined = gpd.sjoin(buildings, zones, how="inner", predicate="intersects")

    bcr_values: dict[int, float] = {}
    for zone_id, group in joined.groupby("zone_id"):
        zone_geom = zones.loc[zones["zone_id"] == zone_id, "geometry"].iloc[0]
        total_intersection = 0.0
        for _, row in group.iterrows():
            intersection = row.geometry.intersection(zone_geom)
            total_intersection += intersection.area
        zone_area = zones.loc[zones["zone_id"] == zone_id, "zone_area"].iloc[0]
        bcr_values[zone_id] = total_intersection / zone_area if zone_area > 0 else 0.0

    zones["bcr"] = zones["zone_id"].map(bcr_values).fillna(0.0)
    return zones


def compute_plan_area_density(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Plan area density (lambda_p) per zone. Alias for compute_bcr."""
    return compute_bcr(buildings, zones).rename(columns={"bcr": "lambda_p"})


# ---------------------------------------------------------------------------
# 3. Floor Area Ratio (FAR)
# ---------------------------------------------------------------------------


def compute_far(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    floor_height: float = 3.0,
) -> gpd.GeoDataFrame:
    """Floor Area Ratio per zone.

    n_floors = max(1, floor(building_height / floor_height)).
    FAR = sum(footprint_area * n_floors) / zone_area.
    Requires a ``height`` column in *buildings*.
    """
    zones = zones.copy()

    if buildings.empty or "height" not in buildings.columns:
        zones["far"] = 0.0
        return zones

    bld = buildings.copy()
    bld["_n_floors"] = np.maximum(1, np.floor(bld["height"] / floor_height)).astype(int)

    joined = gpd.sjoin(bld, zones, how="inner", predicate="intersects")

    far_values: dict[int, float] = {}
    for zone_id, group in joined.groupby("zone_id"):
        zone_geom = zones.loc[zones["zone_id"] == zone_id, "geometry"].iloc[0]
        total_floor_area = 0.0
        for _, row in group.iterrows():
            intersection = row.geometry.intersection(zone_geom)
            total_floor_area += intersection.area * row["_n_floors"]
        zone_area = zones.loc[zones["zone_id"] == zone_id, "zone_area"].iloc[0]
        far_values[zone_id] = total_floor_area / zone_area if zone_area > 0 else 0.0

    zones["far"] = zones["zone_id"].map(far_values).fillna(0.0)
    return zones


# ---------------------------------------------------------------------------
# 4. Height variability (sigma_h)
# ---------------------------------------------------------------------------


def compute_height_variability(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Standard deviation of building heights per zone.

    Requires a ``height`` column in *buildings*.
    Zones with fewer than two buildings receive ``NaN``.
    """
    zones = zones.copy()

    if buildings.empty or "height" not in buildings.columns:
        zones["sigma_h"] = np.nan
        return zones

    joined = gpd.sjoin(buildings, zones, how="inner", predicate="intersects")

    sigma_values: dict[int, float] = {}
    for zone_id, group in joined.groupby("zone_id"):
        heights = group["height"].values
        if len(heights) < 2:
            sigma_values[zone_id] = np.nan
        else:
            sigma_values[zone_id] = float(np.std(heights, ddof=0))

    zones["sigma_h"] = zones["zone_id"].map(sigma_values)
    # Zones with no buildings remain NaN (map returns NaN for missing keys)
    return zones


# ---------------------------------------------------------------------------
# 5. Frontal Area Ratio (lambda_f)
# ---------------------------------------------------------------------------


def _projected_width(
    geom: Polygon | MultiPolygon,
    wind_dir_rad: float,
) -> float:
    """Projected width of *geom* perpendicular to the wind direction.

    The projection axis is perpendicular to the wind vector:
    wind vector = (sin(wind_dir_rad), cos(wind_dir_rad))  (bearing from north)
    perpendicular = (cos(wind_dir_rad), -sin(wind_dir_rad))
    """
    if geom is None or geom.is_empty:
        return 0.0

    # Use the minimum bounding rectangle
    mbr = geom.minimum_rotated_rectangle
    if mbr is None or mbr.is_empty:
        return 0.0

    coords = np.array(mbr.exterior.coords)[:-1]  # drop closing vertex

    # Perpendicular to wind direction (projected axis)
    perp_x = math.cos(wind_dir_rad)
    perp_y = -math.sin(wind_dir_rad)

    projections = coords[:, 0] * perp_x + coords[:, 1] * perp_y
    return float(projections.max() - projections.min())


def compute_frontal_area_ratio(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    wind_dir: float = 0.0,
) -> gpd.GeoDataFrame:
    """Projected frontal area density per zone.

    Parameters
    ----------
    wind_dir : float
        Wind bearing in degrees from north (0 = N, 90 = E).

    lambda_f = sum(projected_width * height) / zone_area.
    Requires a ``height`` column in *buildings*.
    """
    zones = zones.copy()

    if buildings.empty or "height" not in buildings.columns:
        zones["lambda_f"] = 0.0
        return zones

    wind_dir_rad = math.radians(wind_dir)

    joined = gpd.sjoin(buildings, zones, how="inner", predicate="intersects")

    lf_values: dict[int, float] = {}
    for zone_id, group in joined.groupby("zone_id"):
        total_frontal = 0.0
        for _, row in group.iterrows():
            w = _projected_width(row.geometry, wind_dir_rad)
            total_frontal += w * row["height"]
        zone_area = zones.loc[zones["zone_id"] == zone_id, "zone_area"].iloc[0]
        lf_values[zone_id] = total_frontal / zone_area if zone_area > 0 else 0.0

    zones["lambda_f"] = zones["zone_id"].map(lf_values).fillna(0.0)
    return zones


# ---------------------------------------------------------------------------
# 6. Street orientation entropy
# ---------------------------------------------------------------------------


def compute_street_orientation_entropy(
    streets: gpd.GeoDataFrame,
    n_bins: int = 36,
) -> float:
    """Shannon entropy of the street-bearing distribution, normalised to [0, 1].

    Each segment's bearing is folded into [0, 180) because streets are
    bidirectional.  Result is 0 when all segments share one direction and 1
    when bearings are perfectly uniform across *n_bins* bins.
    """
    if streets is None or streets.empty:
        return 0.0

    bearings: list[float] = []
    for geom in streets.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            lines = [geom]
        elif hasattr(geom, "geoms"):
            lines = [g for g in geom.geoms if isinstance(g, LineString)]
        else:
            continue

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            bearing = math.degrees(math.atan2(dx, dy)) % 360
            # Fold to [0, 180)
            if bearing >= 180.0:
                bearing -= 180.0
            bearings.append(bearing)

    if not bearings:
        return 0.0

    bearings_arr = np.array(bearings)
    bin_edges = np.linspace(0, 180, n_bins + 1)
    counts, _ = np.histogram(bearings_arr, bins=bin_edges)

    # Probabilities (ignore zero bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]

    entropy = -np.sum(nonzero * np.log2(nonzero))
    max_entropy = math.log2(n_bins)
    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


# ---------------------------------------------------------------------------
# 6b. Per-zone street orientation entropy
# ---------------------------------------------------------------------------


def _compute_zone_entropy(
    streets: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    n_bins: int = 36,
) -> gpd.GeoDataFrame:
    """Compute street orientation entropy per zone by clipping streets to each zone."""
    zones = zones.copy()

    if streets is None or streets.empty:
        zones["street_orientation_entropy"] = np.nan
        return zones

    # Spatial index for efficiency
    sindex = streets.sindex

    entropy_values: dict[int, float] = {}
    for _, zone_row in zones.iterrows():
        zone_id = zone_row["zone_id"]
        zone_geom = zone_row.geometry

        # Find candidate streets via spatial index
        candidates = list(sindex.intersection(zone_geom.bounds))
        if not candidates:
            entropy_values[zone_id] = np.nan
            continue

        # Clip streets to zone
        zone_streets = streets.iloc[candidates].copy()
        zone_streets = zone_streets[zone_streets.intersects(zone_geom)]
        if zone_streets.empty:
            entropy_values[zone_id] = np.nan
            continue

        clipped = zone_streets.clip(zone_geom)
        clipped = clipped[~clipped.is_empty]
        if clipped.empty:
            entropy_values[zone_id] = np.nan
            continue

        entropy_values[zone_id] = compute_street_orientation_entropy(clipped, n_bins)

    zones["street_orientation_entropy"] = zones["zone_id"].map(entropy_values)
    return zones


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------


def compute_zone_metrics(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    streets: Optional[gpd.GeoDataFrame] = None,
    floor_height: float = 3.0,
    wind_dir: float = 0.0,
    n_bins: int = 36,
) -> gpd.GeoDataFrame:
    """Compute all zone-level urban morphology metrics.

    Returns a copy of *zones* augmented with columns:
    ``bcr``, ``far``, ``sigma_h``, ``lambda_f``, and (optionally)
    ``street_orientation_entropy``.
    """
    # Repair invalid geometries (e.g. self-intersections in Rocinha/Maré)
    buildings = buildings.copy()
    invalid_mask = ~buildings.geometry.is_valid
    if invalid_mask.any():
        logger.warning(
            "Repairing %d invalid building geometries with buffer(0).",
            invalid_mask.sum(),
        )
        buildings.loc[invalid_mask, "geometry"] = buildings.loc[invalid_mask, "geometry"].buffer(0)

    result = compute_bcr(buildings, zones)
    result = compute_far(buildings, result, floor_height=floor_height)
    result = compute_height_variability(buildings, result)
    result = compute_frontal_area_ratio(buildings, result, wind_dir=wind_dir)

    if streets is not None:
        result = _compute_zone_entropy(streets, result, n_bins=n_bins)
    else:
        result["street_orientation_entropy"] = np.nan

    logger.info("Zone metrics computed for %d zones.", len(result))
    return result
