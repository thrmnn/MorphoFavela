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
) -> gpd.GeoDataFrame:
    """Classify tiles as *interior*, *edge*, or drop *exterior* tiles.

    - **interior**: boundary overlap fraction ≥ *interior_threshold*
    - **edge**: 0 < overlap < *interior_threshold*
    - **exterior**: no intersection with boundary (dropped)

    Returns
    -------
    GeoDataFrame
        Copy of *tiles* with added columns: ``classification`` (str),
        ``boundary_overlap_frac`` (float).  Exterior tiles are excluded.
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

    n_int = (tiles["classification"] == "interior").sum()
    n_edge = (tiles["classification"] == "edge").sum()
    logger.info(
        "Classified %d tiles: %d interior, %d edge (dropped %d exterior).",
        len(tiles), n_int, n_edge,
        len(fracs) - len(tiles),
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

    logger.info(
        "Buffer distances: min=%.0fm, max=%.0fm, mean=%.0fm.",
        min(buf_dists), max(buf_dists), np.mean(buf_dists),
    )
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
) -> gpd.GeoDataFrame:
    """Convenience: generate, classify, buffer, and enrich tiles in one call."""
    tiles = generate_tiles(boundary_gdf, tile_size=tile_size, overlap=overlap)
    tiles = classify_tiles(tiles, boundary_gdf, interior_threshold=interior_threshold)
    tiles = compute_buffered_extents(tiles, buildings_gdf)
    tiles = enrich_tiles(tiles, buildings_gdf, dtm_path)
    return tiles
