"""Tile-level feature computation for CFD patch selection.

Computes a rich set of morphometric, height-distribution, and contextual
features for each analysis tile.  Feature groups are computed independently
and merged into a single DataFrame keyed by ``tile_id``.

Feature groups
--------------
1. Morphometric features — BCR, FAR, sigma_h, lambda_f (reuses
   :mod:`src.urban_morphology`).
2. Height distribution — H_mean, H_median, H_std, H_max, H_skewness,
   H_kurtosis, H_iqr.
3. Network features — road_density, intersection_density,
   street_orientation_entropy.
4. SVF aggregation — svf_mean, svf_std from pre-computed SVF points.
5. Topography — elev_mean, elev_range, slope_mean, dtm_coverage_frac
   from a DTM raster.
6. Spatial texture — lacunarity, fractal_dim_box, gini_building_area.
7. Porosity and orientation — porosity_N/E/S/W, orientation_entropy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from scipy import stats
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from src.urban_morphology import (
    compute_bcr,
    compute_far,
    compute_frontal_area_ratio,
    compute_height_variability,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter: tiles → zones
# ---------------------------------------------------------------------------


def _tiles_to_zones(tiles: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert a tiles GeoDataFrame to the zones schema expected by
    :mod:`src.urban_morphology`.

    Renames ``tile_id`` → ``zone_id`` and adds ``zone_area`` from geometry.
    Returns a copy; the original is not mutated.
    """
    zones = tiles.copy()
    zones = zones.rename(columns={"tile_id": "zone_id"})
    zones["zone_area"] = zones.geometry.area
    return zones


def _zones_to_tile_df(
    zones: gpd.GeoDataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extract feature columns from a zones GeoDataFrame, renaming
    ``zone_id`` back to ``tile_id``.

    Returns a plain DataFrame (no geometry) with ``tile_id`` + *feature_cols*.
    """
    df = pd.DataFrame(zones[["zone_id"] + feature_cols])
    df = df.rename(columns={"zone_id": "tile_id"})
    return df


# ---------------------------------------------------------------------------
# Feature Group 1: Morphometric features
# ---------------------------------------------------------------------------


def _compute_morphometric_features(
    tiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    floor_height: float = 3.0,
    wind_dir: float = 0.0,
) -> pd.DataFrame:
    """Compute zone-level urban morphology metrics per tile.

    Wraps the four indicators from :mod:`src.urban_morphology`:

    * **bcr** — Building Coverage Ratio
    * **far** — Floor Area Ratio
    * **sigma_h** — height variability (std dev)
    * **lambda_f** — frontal area ratio

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    buildings : GeoDataFrame
        Building footprints with a ``height`` column.
    floor_height : float
        Assumed storey height for FAR calculation (metres).
    wind_dir : float
        Wind bearing in degrees from north for lambda_f.

    Returns
    -------
    DataFrame
        Columns: ``tile_id``, ``bcr``, ``far``, ``sigma_h``, ``lambda_f``.
    """
    zones = _tiles_to_zones(tiles)

    # Ensure buildings have a 'height' column (urban_morphology expects it)
    bldg = buildings.copy()
    if "height" not in bldg.columns:
        for col in ("altura", "h"):
            if col in bldg.columns:
                bldg["height"] = bldg[col]
                break
    if "height" not in bldg.columns:
        bldg["height"] = np.nan

    zones = compute_bcr(bldg, zones)
    zones = compute_far(bldg, zones, floor_height=floor_height)
    zones = compute_height_variability(bldg, zones)

    # Frontal area ratio for 4 cardinal directions
    feature_cols = ["bcr", "far", "sigma_h"]
    for direction, bearing in [("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)]:
        zones = compute_frontal_area_ratio(bldg, zones, wind_dir=bearing)
        zones = zones.rename(columns={"lambda_f": f"lambda_f_{direction}"})
        feature_cols.append(f"lambda_f_{direction}")

    df = _zones_to_tile_df(zones, feature_cols)

    logger.info("Morphometric features computed for %d tiles.", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature Group 2: Height distribution features
# ---------------------------------------------------------------------------


def _compute_height_features(
    tiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    height_col: str = "height",
) -> pd.DataFrame:
    """Compute building-height distribution statistics per tile.

    For each tile the buildings are selected via a spatial join, and the
    following statistics are derived from their *height_col* values:

    * **H_mean** — arithmetic mean
    * **H_median** — median
    * **H_std** — standard deviation (population, ddof=0)
    * **H_max** — maximum
    * **H_skewness** — Fisher skewness (``scipy.stats.skew``)
    * **H_kurtosis** — excess kurtosis (``scipy.stats.kurtosis``)
    * **H_iqr** — inter-quartile range (Q3 − Q1)

    Tiles with no buildings receive ``NaN`` for all statistics.

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    buildings : GeoDataFrame
        Building footprints; must contain *height_col*.
    height_col : str
        Column name holding building heights.

    Returns
    -------
    DataFrame
        Columns: ``tile_id`` + the seven height features listed above.
    """
    feature_names = [
        "H_mean",
        "H_median",
        "H_std",
        "H_max",
        "H_skewness",
        "H_kurtosis",
        "H_iqr",
    ]

    # Fast path: no usable height data
    if buildings.empty or height_col not in buildings.columns:
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        logger.info("Height features: no building data — all NaN.")
        return result

    # Spatial join: attach tile_id to each building
    joined = gpd.sjoin(
        buildings[[height_col, "geometry"]],
        tiles[["tile_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    records: list[dict] = []
    for tile_id, group in joined.groupby("tile_id"):
        heights = group[height_col].dropna().values
        if len(heights) == 0:
            row = {feat: np.nan for feat in feature_names}
        else:
            q1, q3 = np.percentile(heights, [25, 75])
            row = {
                "H_mean": float(np.mean(heights)),
                "H_median": float(np.median(heights)),
                "H_std": float(np.std(heights, ddof=0)),
                "H_max": float(np.max(heights)),
                "H_skewness": float(stats.skew(heights, bias=True)),
                "H_kurtosis": float(stats.kurtosis(heights, bias=True)),
                "H_iqr": float(q3 - q1),
            }
        row["tile_id"] = tile_id
        records.append(row)

    if records:
        feat_df = pd.DataFrame(records)
    else:
        feat_df = pd.DataFrame(columns=["tile_id"] + feature_names)

    # Ensure every tile is represented (left join with all tile_ids)
    all_tiles = pd.DataFrame({"tile_id": tiles["tile_id"]})
    result = all_tiles.merge(feat_df, on="tile_id", how="left")

    logger.info("Height features computed for %d tiles.", len(result))
    return result


# ---------------------------------------------------------------------------
# Feature Group 3: Street-network features
# ---------------------------------------------------------------------------


def _compute_street_features(
    tiles: gpd.GeoDataFrame,
    streets: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """Street-network features per tile.

    Features
    --------
    * **road_density** — total road length / tile area (km/km²)
    * **intersection_density** — nodes with degree >= 3 / tile area (per km²)
    * **street_orientation_entropy** — Shannon entropy of segment bearings
      binned into 36 bins of 10°, normalised to [0, 1].

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    streets : GeoDataFrame, optional
        Street centreline geometries (LineString / MultiLineString).

    Returns
    -------
    DataFrame
        Columns: ``tile_id``, ``road_density``, ``intersection_density``,
        ``street_orientation_entropy``.
    """
    feature_names = ["road_density", "intersection_density",
                     "street_orientation_entropy"]

    # Fast path: no street data
    if streets is None or streets.empty:
        logger.info("Street features: no street data — all NaN.")
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        return result

    max_entropy = np.log(36)  # max Shannon entropy for 36 bins
    records: list[dict] = []

    for _, tile in tiles.iterrows():
        tile_id = tile["tile_id"]
        tile_area_m2 = tile.geometry.area
        tile_area_km2 = tile_area_m2 / 1e6

        clipped = gpd.clip(streets, tile.geometry)

        if clipped.empty:
            records.append({
                "tile_id": tile_id,
                "road_density": 0.0,
                "intersection_density": 0.0,
                "street_orientation_entropy": np.nan,
            })
            continue

        # --- road_density: total length (km) / tile area (km²) ---
        total_length_m = clipped.geometry.length.sum()
        road_density = (total_length_m / 1000) / tile_area_km2

        # --- intersection_density: endpoints with degree >= 3 ---
        start_pts = []
        end_pts = []
        bearings = []
        for geom in clipped.geometry:
            if geom is None or geom.is_empty:
                continue
            # Handle both LineString and MultiLineString
            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                sx, sy = coords[0][0], coords[0][1]
                ex, ey = coords[-1][0], coords[-1][1]
                # Round to 0.1 m for node matching
                start_pts.append((round(sx, 1), round(sy, 1)))
                end_pts.append((round(ex, 1), round(ey, 1)))
                # Bearing from start to end
                dx = ex - sx
                dy = ey - sy
                bearing = np.degrees(np.arctan2(dx, dy)) % 360
                bearings.append(bearing)

        all_pts = start_pts + end_pts
        if all_pts:
            pt_counts = pd.Series(all_pts).value_counts()
            n_intersections = int((pt_counts >= 3).sum())
        else:
            n_intersections = 0
        intersection_density = n_intersections / tile_area_km2

        # --- street_orientation_entropy ---
        if bearings:
            bearings_arr = np.array(bearings) % 180  # fold to [0, 180)
            bins = np.linspace(0, 180, 37)  # 36 bins of 5° (half-circle)
            counts, _ = np.histogram(bearings_arr, bins=bins)
            counts = counts[counts > 0]
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs))
            # Normalise by max entropy for 36 bins
            orientation_entropy = float(entropy / max_entropy)
        else:
            orientation_entropy = np.nan

        records.append({
            "tile_id": tile_id,
            "road_density": float(road_density),
            "intersection_density": float(intersection_density),
            "street_orientation_entropy": orientation_entropy,
        })

    feat_df = pd.DataFrame(records)
    all_tiles = pd.DataFrame({"tile_id": tiles["tile_id"]})
    result = all_tiles.merge(feat_df, on="tile_id", how="left")

    logger.info("Street features computed for %d tiles.", len(result))
    return result


# ---------------------------------------------------------------------------
# Feature Group 4: SVF aggregation
# ---------------------------------------------------------------------------


def _compute_svf_features(
    tiles: gpd.GeoDataFrame,
    svf_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Sky View Factor statistics per tile.

    Loads pre-computed SVF point data from a GeoPackage and aggregates
    per tile via spatial join.

    Features
    --------
    * **svf_mean** — mean SVF of street-level points within tile
    * **svf_std** — standard deviation of SVF within tile

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    svf_path : str or Path, optional
        Path to a GeoPackage with Point geometry and an ``svf`` column.

    Returns
    -------
    DataFrame
        Columns: ``tile_id``, ``svf_mean``, ``svf_std``.
    """
    feature_names = ["svf_mean", "svf_std"]

    # Fast path: no SVF data
    if svf_path is None or not Path(svf_path).exists():
        logger.info("SVF features: no SVF data — all NaN.")
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        return result

    svf_gdf = gpd.read_file(svf_path)
    if svf_gdf.empty or "svf" not in svf_gdf.columns:
        logger.info("SVF features: SVF file empty or missing 'svf' column — all NaN.")
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        return result

    # Ensure matching CRS
    if svf_gdf.crs != tiles.crs:
        svf_gdf = svf_gdf.to_crs(tiles.crs)

    # Spatial join: attach tile_id to each SVF point
    joined = gpd.sjoin(
        svf_gdf[["svf", "geometry"]],
        tiles[["tile_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    records: list[dict] = []
    for tile_id, group in joined.groupby("tile_id"):
        svf_vals = group["svf"].dropna().values
        if len(svf_vals) == 0:
            records.append({"tile_id": tile_id, "svf_mean": np.nan, "svf_std": np.nan})
        else:
            records.append({
                "tile_id": tile_id,
                "svf_mean": float(np.mean(svf_vals)),
                "svf_std": float(np.std(svf_vals, ddof=0)),
            })

    if records:
        feat_df = pd.DataFrame(records)
    else:
        feat_df = pd.DataFrame(columns=["tile_id"] + feature_names)

    all_tiles = pd.DataFrame({"tile_id": tiles["tile_id"]})
    result = all_tiles.merge(feat_df, on="tile_id", how="left")

    logger.info("SVF features computed for %d tiles.", len(result))
    return result


# ---------------------------------------------------------------------------
# Feature Group 5: Topography (DTM-derived)
# ---------------------------------------------------------------------------


def _compute_topography_features(
    tiles: gpd.GeoDataFrame,
    dtm_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """DTM-derived topography features per tile.

    Features
    --------
    * **elev_mean** — mean elevation from DTM within tile
    * **elev_range** — max − min elevation within tile
    * **slope_mean** — mean slope magnitude (degrees) via ``np.gradient``
    * **dtm_coverage_frac** — fraction of DTM pixels that are valid (not NaN)

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    dtm_path : str or Path, optional
        Path to a DTM raster file readable by rasterio.

    Returns
    -------
    DataFrame
        Columns: ``tile_id``, ``elev_mean``, ``elev_range``, ``slope_mean``,
        ``dtm_coverage_frac``.
    """
    feature_names = ["elev_mean", "elev_range", "slope_mean", "dtm_coverage_frac"]

    # Fast path: no DTM
    if dtm_path is None or not Path(dtm_path).exists():
        logger.info("Topography features: no DTM data — all NaN.")
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        return result

    records: list[dict] = []

    with rasterio.open(dtm_path) as src:
        nodata = src.nodata
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)

        for _, tile in tiles.iterrows():
            tile_id = tile["tile_id"]
            tile_geom = tile.geometry.__geo_interface__

            try:
                out_image, _ = rasterio.mask.mask(
                    src, [tile_geom], crop=True, filled=True,
                )
            except Exception:
                records.append({
                    "tile_id": tile_id,
                    **{feat: np.nan for feat in feature_names},
                })
                continue

            data = out_image[0].astype(float)

            # Mark nodata as NaN
            if nodata is not None:
                data[data == nodata] = np.nan
            valid = np.isfinite(data)
            n_total = data.size
            n_valid = int(valid.sum())

            if n_valid == 0:
                records.append({
                    "tile_id": tile_id,
                    **{feat: np.nan for feat in feature_names},
                })
                continue

            valid_vals = data[valid]
            elev_mean = float(np.mean(valid_vals))
            elev_range = float(np.max(valid_vals) - np.min(valid_vals))
            dtm_coverage_frac = n_valid / n_total

            # Slope: gradient magnitude in degrees
            # Replace NaN with local mean for gradient computation
            data_filled = np.where(valid, data, np.nanmean(data))
            dy, dx = np.gradient(data_filled, res_y, res_x)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            # Only average over valid pixels
            slope_mean = float(np.mean(slope_deg[valid]))

            records.append({
                "tile_id": tile_id,
                "elev_mean": elev_mean,
                "elev_range": elev_range,
                "slope_mean": slope_mean,
                "dtm_coverage_frac": dtm_coverage_frac,
            })

    if records:
        feat_df = pd.DataFrame(records)
    else:
        feat_df = pd.DataFrame(columns=["tile_id"] + feature_names)

    all_tiles = pd.DataFrame({"tile_id": tiles["tile_id"]})
    result = all_tiles.merge(feat_df, on="tile_id", how="left")

    logger.info("Topography features computed for %d tiles.", len(result))
    return result


# ---------------------------------------------------------------------------
# Feature Group 6: Spatial texture features
# ---------------------------------------------------------------------------


def _rasterize_buildings_in_tile(
    tile_geom,
    tile_buildings: gpd.GeoDataFrame,
    resolution: float = 5.0,
    grid_size: int = 40,
) -> np.ndarray:
    """Rasterize building footprints to a binary 2D array within a tile.

    Parameters
    ----------
    tile_geom : shapely geometry
        The tile polygon.
    tile_buildings : GeoDataFrame
        Building footprints already clipped/filtered to this tile.
    resolution : float
        Pixel size in metres.
    grid_size : int
        Number of pixels along each axis.

    Returns
    -------
    ndarray of shape (grid_size, grid_size)
        Binary array: 1 = building present, 0 = empty.
    """
    raster = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if tile_buildings.empty:
        return raster

    minx, miny, _, maxy = tile_geom.bounds

    for geom in tile_buildings.geometry:
        if geom is None or geom.is_empty:
            continue
        gminx, gminy, gmaxx, gmaxy = geom.bounds
        # Pixel range that this building could touch
        col_start = max(0, int((gminx - minx) / resolution))
        col_end = min(grid_size, int(np.ceil((gmaxx - minx) / resolution)))
        row_start = max(0, int((maxy - gmaxy) / resolution))
        row_end = min(grid_size, int(np.ceil((maxy - gminy) / resolution)))

        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                # Centre of pixel
                px = minx + (c + 0.5) * resolution
                # Row 0 = top (max y)
                py = maxy - (r + 0.5) * resolution
                if geom.contains(Point(px, py)):
                    raster[r, c] = 1
    return raster


def _lacunarity(binary_raster: np.ndarray, box_size: int) -> float:
    """Compute gliding-box lacunarity for a given box size (in pixels).

    Slides the box over the binary raster with stride=1.
    Lacunarity = var(sums) / mean(sums)^2 + 1.
    Returns NaN if the mean is zero (empty raster region).
    """
    rows, cols = binary_raster.shape
    if box_size > rows or box_size > cols:
        return np.nan

    # Use a 2D prefix sum for efficient box-sum computation
    cum = binary_raster.astype(np.float64)
    cum = np.cumsum(np.cumsum(cum, axis=0), axis=1)

    # Pad with zeros for easier indexing
    padded = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    padded[1:, 1:] = cum

    n_rows = rows - box_size + 1
    n_cols = cols - box_size + 1
    sums = (
        padded[box_size:, box_size:]
        - padded[:n_rows, box_size:]
        - padded[box_size:, :n_cols]
        + padded[:n_rows, :n_cols]
    )

    mean_val = sums.mean()
    if mean_val == 0:
        return np.nan

    var_val = sums.var()
    return float(var_val / (mean_val ** 2) + 1.0)


def _fractal_dimension_box_counting(binary_raster: np.ndarray) -> float:
    """Box-counting fractal dimension of a binary raster.

    Uses box sizes of [1, 2, 4, 8] pixels.  Fits log(N) vs log(1/size).
    Returns NaN if fewer than 2 valid data points.
    """
    rows, cols = binary_raster.shape
    box_sizes = [1, 2, 4, 8]

    log_inv_sizes: list[float] = []
    log_counts: list[float] = []

    for bs in box_sizes:
        if bs > rows or bs > cols:
            continue
        # Count boxes that contain at least one occupied pixel
        n_boxes_r = rows // bs
        n_boxes_c = cols // bs
        count = 0
        for r in range(n_boxes_r):
            for c in range(n_boxes_c):
                block = binary_raster[r * bs:(r + 1) * bs, c * bs:(c + 1) * bs]
                if block.any():
                    count += 1
        if count > 0:
            log_inv_sizes.append(np.log(1.0 / bs))
            log_counts.append(np.log(count))

    if len(log_inv_sizes) < 2:
        return np.nan

    # Linear regression: slope = fractal dimension
    coeffs = np.polyfit(log_inv_sizes, log_counts, 1)
    return float(coeffs[0])


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values.

    If 0 or 1 values, returns 0.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    x_sorted = np.sort(values)
    total = x_sorted.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x_sorted)) / (n * total) - (n + 1.0) / n)


def _compute_texture_features(
    tiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Spatial texture features per tile.

    Features
    --------
    * **lacunarity** — gliding-box lacunarity on a binary building footprint
      raster (5 m resolution, box size = 25 m = 5 pixels).
    * **fractal_dim_box** — box-counting fractal dimension of the binary
      building mask.
    * **gini_building_area** — Gini coefficient of individual building
      footprint areas within the tile.

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    buildings : GeoDataFrame
        Building footprints.

    Returns
    -------
    DataFrame
        Columns: ``tile_id``, ``lacunarity``, ``fractal_dim_box``,
        ``gini_building_area``.
    """
    feature_names = ["lacunarity", "fractal_dim_box", "gini_building_area"]
    resolution = 5.0
    grid_size = 40
    box_size_px = 5  # 25 m / 5 m

    if buildings.empty:
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        logger.info("Texture features: no building data — all NaN.")
        return result

    # Spatial join: find buildings per tile
    joined = gpd.sjoin(
        buildings[["geometry"]],
        tiles[["tile_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    records: list[dict] = []
    for _, tile in tiles.iterrows():
        tile_id = tile["tile_id"]
        tile_geom = tile.geometry

        # Buildings in this tile
        mask = joined["tile_id"] == tile_id
        tile_bldg_idx = joined.loc[mask].index
        tile_buildings = buildings.loc[buildings.index.isin(tile_bldg_idx)]

        # Rasterize
        raster = _rasterize_buildings_in_tile(
            tile_geom, tile_buildings,
            resolution=resolution, grid_size=grid_size,
        )

        # Lacunarity
        lac = _lacunarity(raster, box_size_px)

        # Fractal dimension
        fd = _fractal_dimension_box_counting(raster)

        # Gini of building areas
        if len(tile_buildings) > 0:
            areas = tile_buildings.geometry.area.values
            gini = _gini_coefficient(areas)
        else:
            gini = 0.0

        records.append({
            "tile_id": tile_id,
            "lacunarity": lac,
            "fractal_dim_box": fd,
            "gini_building_area": gini,
        })

    feat_df = pd.DataFrame(records)
    all_tiles = pd.DataFrame({"tile_id": tiles["tile_id"]})
    result = all_tiles.merge(feat_df, on="tile_id", how="left")

    logger.info("Texture features computed for %d tiles.", len(result))
    return result


# ---------------------------------------------------------------------------
# Feature Group 7: Porosity and orientation features
# ---------------------------------------------------------------------------


def _compute_porosity_orientation_features(
    tiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Porosity and orientation features per tile.

    Features
    --------
    * **porosity_N**, **porosity_E**, **porosity_S**, **porosity_W** —
      connected porosity in four cardinal directions.  For each direction,
      rays are cast across the tile at 5 m intervals perpendicular to the
      travel direction.  Porosity = fraction of rays NOT blocked by any
      building footprint.
    * **orientation_entropy** — Shannon entropy of building footprint
      major-axis orientations (from minimum rotated rectangle), binned
      into 18 bins of 10 degrees, normalised by max entropy ``log2(18)``.

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry``.
    buildings : GeoDataFrame
        Building footprints.

    Returns
    -------
    DataFrame
        Columns: ``tile_id``, ``porosity_N``, ``porosity_E``, ``porosity_S``,
        ``porosity_W``, ``orientation_entropy``.
    """
    feature_names = [
        "porosity_N", "porosity_E", "porosity_S", "porosity_W",
        "orientation_entropy",
    ]
    ray_spacing = 5.0
    max_entropy = np.log2(18)

    if buildings.empty:
        result = pd.DataFrame({"tile_id": tiles["tile_id"]})
        for feat in feature_names:
            result[feat] = np.nan
        logger.info("Porosity/orientation features: no building data — all NaN.")
        return result

    # Spatial join: find buildings per tile
    joined = gpd.sjoin(
        buildings[["geometry"]],
        tiles[["tile_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    records: list[dict] = []
    for _, tile in tiles.iterrows():
        tile_id = tile["tile_id"]
        tile_geom = tile.geometry
        minx, miny, maxx, maxy = tile_geom.bounds

        # Buildings in this tile
        mask = joined["tile_id"] == tile_id
        tile_bldg_idx = joined.loc[mask].index
        tile_buildings = buildings.loc[buildings.index.isin(tile_bldg_idx)]

        row: dict = {"tile_id": tile_id}

        # --- Porosity ---
        if tile_buildings.empty:
            row["porosity_N"] = 1.0
            row["porosity_E"] = 1.0
            row["porosity_S"] = 1.0
            row["porosity_W"] = 1.0
        else:
            tree = STRtree(tile_buildings.geometry.values)

            # North/South: rays travel vertically (along y).
            # Cast vertical lines at x intervals across the tile width.
            x_positions = np.arange(minx + ray_spacing / 2, maxx, ray_spacing)
            n_blocked_ns = 0
            for x in x_positions:
                ray = LineString([(x, miny), (x, maxy)])
                hits = tree.query(ray, predicate="intersects")
                if len(hits) > 0:
                    n_blocked_ns += 1
            n_rays_ns = len(x_positions) if len(x_positions) > 0 else 1
            # N and S porosity are symmetric for vertical rays
            porosity_ns = float(1.0 - n_blocked_ns / n_rays_ns)
            row["porosity_N"] = porosity_ns
            row["porosity_S"] = porosity_ns

            # East/West: rays travel horizontally (along x).
            # Cast horizontal lines at y intervals across the tile height.
            y_positions = np.arange(miny + ray_spacing / 2, maxy, ray_spacing)
            n_blocked_ew = 0
            for y in y_positions:
                ray = LineString([(minx, y), (maxx, y)])
                hits = tree.query(ray, predicate="intersects")
                if len(hits) > 0:
                    n_blocked_ew += 1
            n_rays_ew = len(y_positions) if len(y_positions) > 0 else 1
            porosity_ew = float(1.0 - n_blocked_ew / n_rays_ew)
            row["porosity_E"] = porosity_ew
            row["porosity_W"] = porosity_ew

        # --- Orientation entropy ---
        if tile_buildings.empty:
            row["orientation_entropy"] = np.nan
        else:
            orientations: list[float] = []
            for geom in tile_buildings.geometry:
                if geom is None or geom.is_empty:
                    continue
                mrr = geom.minimum_rotated_rectangle
                coords = list(mrr.exterior.coords)
                # Find the longest edge to determine orientation
                max_len = 0.0
                angle = 0.0
                for i in range(len(coords) - 1):
                    dx = coords[i + 1][0] - coords[i][0]
                    dy = coords[i + 1][1] - coords[i][1]
                    edge_len = np.sqrt(dx ** 2 + dy ** 2)
                    if edge_len > max_len:
                        max_len = edge_len
                        angle = np.degrees(np.arctan2(dy, dx)) % 180.0
                orientations.append(angle)

            if len(orientations) == 0:
                row["orientation_entropy"] = np.nan
            else:
                orientations_arr = np.array(orientations)
                bins = np.linspace(0, 180, 19)  # 18 bins of 10 degrees
                counts, _ = np.histogram(orientations_arr, bins=bins)
                counts = counts[counts > 0]
                if len(counts) == 0:
                    row["orientation_entropy"] = 0.0
                else:
                    probs = counts / counts.sum()
                    entropy = -np.sum(probs * np.log2(probs))
                    row["orientation_entropy"] = float(entropy / max_entropy)

        records.append(row)

    feat_df = pd.DataFrame(records)
    all_tiles = pd.DataFrame({"tile_id": tiles["tile_id"]})
    result = all_tiles.merge(feat_df, on="tile_id", how="left")

    logger.info(
        "Porosity/orientation features computed for %d tiles.", len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_tile_features(
    tiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    streets: Optional[gpd.GeoDataFrame] = None,
    svf_path: Optional[Union[str, Path]] = None,
    dtm_path: Optional[Union[str, Path]] = None,
    floor_height: float = 3.0,
    wind_dir: float = 0.0,
    height_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute all tile-level features and merge into a single DataFrame.

    Each feature group function is called independently and produces a
    ``pd.DataFrame`` keyed by ``tile_id``.  Results are successively merged
    so that the final output contains one row per tile with all available
    feature columns.

    Parameters
    ----------
    tiles : GeoDataFrame
        Analysis tiles with ``tile_id`` and ``geometry`` columns.
    buildings : GeoDataFrame
        Building footprints; must include a height column.
    streets : GeoDataFrame, optional
        Street centrelines for network features.
    svf_path : str or Path, optional
        Path to a GeoPackage with SVF point data (``svf`` column).
    dtm_path : str or Path, optional
        Path to a DTM raster for topography features.
    floor_height : float
        Assumed storey height for FAR (metres).
    wind_dir : float
        Wind bearing in degrees from north for frontal area ratio.
    height_col : str
        Column in *buildings* that holds building height values.

    Returns
    -------
    DataFrame
        One row per tile.  Columns: ``tile_id`` + all feature columns from
        every group.
    """
    logger.info("Computing tile features for %d tiles ...", len(tiles))

    # Auto-detect height column
    if height_col is None:
        for candidate in ("altura", "height", "h"):
            if candidate in buildings.columns:
                height_col = candidate
                break
        if height_col is None:
            height_col = "altura"  # fallback even if missing
    logger.info("Using height column: %s", height_col)

    # --- Group 1: morphometric features ---
    morpho_df = _compute_morphometric_features(
        tiles, buildings, floor_height=floor_height, wind_dir=wind_dir,
    )

    # --- Group 2: height distribution ---
    height_df = _compute_height_features(
        tiles, buildings, height_col=height_col,
    )

    # --- Group 3: street network ---
    street_df = _compute_street_features(tiles, streets=streets)

    # --- Group 4: SVF aggregation ---
    svf_df = _compute_svf_features(tiles, svf_path=svf_path)

    # --- Group 5: topography (DTM-derived) ---
    topo_df = _compute_topography_features(tiles, dtm_path=dtm_path)

    # --- Group 6: spatial texture ---
    texture_df = _compute_texture_features(tiles, buildings)

    # --- Group 7: porosity and orientation ---
    porosity_df = _compute_porosity_orientation_features(tiles, buildings)

    # --- Merge all groups on tile_id ---
    group_dfs = [
        morpho_df,
        height_df,
        street_df,
        svf_df,
        topo_df,
        texture_df,
        porosity_df,
    ]

    result = group_dfs[0]
    for df in group_dfs[1:]:
        result = pd.merge(result, df, on="tile_id", how="outer")

    logger.info(
        "Tile features complete: %d tiles, %d feature columns.",
        len(result),
        len(result.columns) - 1,  # exclude tile_id
    )
    return result
