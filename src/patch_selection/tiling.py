"""Tile generation, classification, and buffering for CFD patch selection.

Generates a regular grid of analysis tiles over the study area, classifies
them by overlap with the community boundary, and computes buffered extents
for data extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TILE_SIZE = 200.0  # metres
DEFAULT_INTERIOR_THRESHOLD = 0.50
DEFAULT_BUFFER_CLAMP = (50.0, 500.0)  # min / max buffer distance (m)
DEFAULT_BUFFER_MULTIPLIER = 5.0  # buffer = multiplier × H_mean
DEFAULT_MIN_BUILDING_COUNT = 10  # tiles with fewer buildings are excluded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_tiles(
    boundary_gdf: gpd.GeoDataFrame,
    tile_size: float = DEFAULT_TILE_SIZE,
    overlap: float = 0.0,
) -> gpd.GeoDataFrame:
    """Generate rectangular grid tiles over the bounding box of *boundary_gdf*.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Community boundary polygon(s).
    tile_size : float
        Side length of each tile in metres.
    overlap : float
        Overlap between adjacent tiles in metres (0 = no overlap).

    Returns
    -------
    GeoDataFrame
        Columns: ``tile_id`` (int), ``geometry`` (Polygon).
        CRS inherited from *boundary_gdf*.
    """
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    step = tile_size - overlap

    tiles = []
    row = 0
    y = miny
    while y < maxy:
        col = 0
        x = minx
        while x < maxx:
            tiles.append(
                {
                    "tile_id": f"{row:03d}_{col:03d}",
                    "geometry": box(x, y, x + tile_size, y + tile_size),
                }
            )
            col += 1
            x += step
        row += 1
        y += step

    gdf = gpd.GeoDataFrame(tiles, crs=boundary_gdf.crs)
    logger.info("Generated %d tiles (%.0fm, overlap=%.0fm).", len(gdf), tile_size, overlap)
    return gdf


def classify_tiles(
    tiles: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    interior_threshold: float = DEFAULT_INTERIOR_THRESHOLD,
    clip_to_boundary: bool = True,
) -> gpd.GeoDataFrame:
    """Classify tiles as *interior*, *edge*, or drop *exterior* tiles.

    - **interior**: boundary overlap fraction >= *interior_threshold*
    - **edge**: 0 < overlap < *interior_threshold*
    - **exterior**: no intersection with boundary (dropped)

    When *clip_to_boundary* is True (default), edge tile geometries are
    clipped to the settlement boundary so that features are computed only
    within the actual settlement footprint.  The original unclipped
    geometry is preserved in ``geometry_unclipped``.

    Returns
    -------
    GeoDataFrame
        Copy of *tiles* with added columns: ``classification`` (str),
        ``boundary_overlap_frac`` (float).  Exterior tiles are excluded.
        If *clip_to_boundary*, edge tiles also get ``geometry_unclipped``.
    """
    boundary_poly = unary_union(boundary_gdf.geometry.values)
    tiles = tiles.copy()

    fracs = []
    for geom in tiles.geometry:
        inter = geom.intersection(boundary_poly)
        fracs.append(inter.area / geom.area if geom.area > 0 else 0.0)

    tiles["boundary_overlap_frac"] = fracs

    # Drop exterior
    tiles = tiles[tiles["boundary_overlap_frac"] > 0].copy()

    tiles["classification"] = np.where(
        tiles["boundary_overlap_frac"] >= interior_threshold,
        "interior",
        "edge",
    )

    # Clip edge tiles to boundary
    if clip_to_boundary:
        tiles["geometry_unclipped"] = tiles.geometry.copy()
        edge_mask = tiles["classification"] == "edge"
        for idx in tiles.index[edge_mask]:
            clipped = tiles.loc[idx, "geometry"].intersection(boundary_poly)
            if not clipped.is_empty:
                tiles.loc[idx, "geometry"] = clipped

    n_int = (tiles["classification"] == "interior").sum()
    n_edge = (tiles["classification"] == "edge").sum()
    logger.info(
        "Classified %d tiles: %d interior, %d edge (dropped %d exterior)%s.",
        len(tiles), n_int, n_edge,
        len(fracs) - len(tiles),
        ", edge tiles clipped to boundary" if clip_to_boundary else "",
    )
    return tiles.reset_index(drop=True)


def suggest_tile_size(
    boundary_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
) -> float:
    """Suggest a tile size (metres) based on area extent and building density.

    Heuristic for informal settlements:

    * Dense (>5000 buildings/km²): 100-150 m tiles
    * Moderate (2000-5000): 150-200 m
    * Sparse (<2000): 200-300 m

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Community boundary polygon(s).
    buildings_gdf : GeoDataFrame
        Building footprints.

    Returns
    -------
    float
        Suggested tile side length in metres.
    """
    area_m2 = boundary_gdf.geometry.area.sum()
    area_km2 = area_m2 / 1e6
    n_buildings = len(buildings_gdf)

    if area_km2 <= 0:
        logger.warning("Boundary area is zero — defaulting to %.0fm.", DEFAULT_TILE_SIZE)
        return DEFAULT_TILE_SIZE

    density = n_buildings / area_km2  # buildings per km²

    if density > 5000:
        tile_size = 100.0
    elif density > 3500:
        tile_size = 125.0
    elif density > 2000:
        tile_size = 150.0
    elif density > 1000:
        tile_size = 200.0
    else:
        tile_size = 250.0

    logger.info(
        "suggest_tile_size: %.1f km², %d buildings, density=%.0f/km² → %.0fm tiles.",
        area_km2, n_buildings, density, tile_size,
    )
    return tile_size


def filter_tiles_by_building_count(
    tiles: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    min_building_count: int = DEFAULT_MIN_BUILDING_COUNT,
) -> gpd.GeoDataFrame:
    """Exclude tiles with fewer than *min_building_count* buildings.

    Uses a spatial index for fast intersection counting.  Tiles that do not
    meet the threshold are dropped.

    Parameters
    ----------
    tiles : GeoDataFrame
        Tiles with ``tile_id`` and ``geometry``.
    buildings_gdf : GeoDataFrame
        Building footprints.
    min_building_count : int
        Minimum number of intersecting buildings required to keep a tile.

    Returns
    -------
    GeoDataFrame
        Filtered copy of *tiles* with an added ``n_buildings`` column.
    """
    tiles = tiles.copy()

    bldg_sindex = buildings_gdf.sindex
    counts = []
    for _, tile in tiles.iterrows():
        candidates = list(bldg_sindex.intersection(tile.geometry.bounds))
        if candidates:
            nearby = buildings_gdf.iloc[candidates]
            n = int(nearby.geometry.intersects(tile.geometry).sum())
        else:
            n = 0
        counts.append(n)

    tiles["n_buildings"] = counts
    n_before = len(tiles)
    tiles = tiles[tiles["n_buildings"] >= min_building_count].copy()
    n_dropped = n_before - len(tiles)

    if n_dropped > 0:
        logger.info(
            "Building count filter: dropped %d / %d tiles (< %d buildings).",
            n_dropped, n_before, min_building_count,
        )
    else:
        logger.info(
            "Building count filter: all %d tiles have >= %d buildings.",
            n_before, min_building_count,
        )

    return tiles.reset_index(drop=True)


def compute_buffered_extents(
    tiles: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    clamp: tuple[float, float] = DEFAULT_BUFFER_CLAMP,
    height_field: str = "altura",
    global_h_mean: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Compute per-tile buffered geometry for CFD data extraction.

    Buffer distance = *multiplier* × mean building height of buildings
    intersecting the tile, clamped to *clamp*.  Falls back to *global_h_mean*
    (or the dataset-wide mean) for tiles with no buildings.

    Adds columns: ``geometry_buffered`` (Polygon), ``buffer_distance_m`` (float).
    """
    tiles = tiles.copy()

    if global_h_mean is None:
        h = buildings_gdf[height_field].dropna()
        global_h_mean = float(h[h > 0].mean()) if len(h[h > 0]) > 0 else 7.6

    # Spatial index for fast intersection lookup
    bldg_sindex = buildings_gdf.sindex

    buf_dists = []
    buf_geoms = []

    for _, tile in tiles.iterrows():
        # Find candidate buildings
        candidates = list(bldg_sindex.intersection(tile.geometry.bounds))
        if candidates:
            nearby = buildings_gdf.iloc[candidates]
            intersecting = nearby[nearby.geometry.intersects(tile.geometry)]
            heights = intersecting[height_field].dropna()
            heights = heights[heights > 0]
            h_mean = float(heights.mean()) if len(heights) > 0 else global_h_mean
        else:
            h_mean = global_h_mean

        dist = np.clip(multiplier * h_mean, clamp[0], clamp[1])
        buf_dists.append(dist)
        buf_geoms.append(tile.geometry.buffer(dist))

    tiles["buffer_distance_m"] = buf_dists
    tiles["geometry_buffered"] = buf_geoms

    if buf_dists:
        logger.info(
            "Buffer distances: min=%.0fm, max=%.0fm, mean=%.0fm.",
            min(buf_dists), max(buf_dists), np.mean(buf_dists),
        )
    else:
        logger.warning("No tiles to buffer.")
    return tiles


def enrich_tiles(
    tiles: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    dtm_path: Path,
) -> gpd.GeoDataFrame:
    """Add per-tile building count and DTM coverage flag.

    Adds columns: ``n_buildings`` (int), ``has_dtm_coverage`` (bool — True if
    ≥50% of tile pixels have valid DTM data).
    """
    tiles = tiles.copy()

    # Building counts via spatial join
    bldg_sindex = buildings_gdf.sindex
    counts = []
    for _, tile in tiles.iterrows():
        candidates = list(bldg_sindex.intersection(tile.geometry.bounds))
        if candidates:
            nearby = buildings_gdf.iloc[candidates]
            n = nearby.geometry.intersects(tile.geometry).sum()
        else:
            n = 0
        counts.append(int(n))
    tiles["n_buildings"] = counts

    # DTM coverage check
    has_coverage = []
    with rasterio.open(dtm_path) as src:
        for _, tile in tiles.iterrows():
            try:
                from rasterio.mask import mask as rio_mask
                out_image, _ = rio_mask(
                    src,
                    [tile.geometry.__geo_interface__],
                    crop=True,
                    nodata=np.nan,
                )
                data = out_image[0]
                total = data.size
                valid = np.isfinite(data).sum()
                has_coverage.append(valid / total >= 0.50 if total > 0 else False)
            except Exception:
                has_coverage.append(False)

    tiles["has_dtm_coverage"] = has_coverage

    n_with_bldg = (tiles["n_buildings"] > 0).sum()
    n_with_dtm = sum(has_coverage)
    logger.info(
        "Enriched %d tiles: %d with buildings, %d with DTM coverage.",
        len(tiles), n_with_bldg, n_with_dtm,
    )
    return tiles


def build_tile_grid(
    boundary_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    dtm_path: Path,
    tile_size: float = DEFAULT_TILE_SIZE,
    overlap: float = 0.0,
    interior_threshold: float = DEFAULT_INTERIOR_THRESHOLD,
    min_building_count: int = DEFAULT_MIN_BUILDING_COUNT,
    clip_to_boundary: bool = True,
) -> gpd.GeoDataFrame:
    """Convenience: generate, classify, filter, buffer, and enrich tiles in one call.

    Parameters
    ----------
    boundary_gdf : GeoDataFrame
        Community boundary polygon(s).
    buildings_gdf : GeoDataFrame
        Building footprints.
    dtm_path : Path
        Path to the DTM raster.
    tile_size : float
        Side length of each tile in metres.
    overlap : float
        Overlap between adjacent tiles in metres.
    interior_threshold : float
        Minimum overlap fraction to classify a tile as interior.
    min_building_count : int
        Tiles with fewer buildings are excluded.  Set to 0 to disable.
    clip_to_boundary : bool
        If True, edge tile geometries are clipped to the settlement boundary.
    """
    tiles = generate_tiles(boundary_gdf, tile_size=tile_size, overlap=overlap)
    tiles = classify_tiles(
        tiles, boundary_gdf,
        interior_threshold=interior_threshold,
        clip_to_boundary=clip_to_boundary,
    )
    if min_building_count > 0:
        tiles = filter_tiles_by_building_count(
            tiles, buildings_gdf, min_building_count=min_building_count,
        )
    tiles = compute_buffered_extents(tiles, buildings_gdf)
    tiles = enrich_tiles(tiles, buildings_gdf, dtm_path)
    return tiles
