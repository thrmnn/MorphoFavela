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
3–7. (Stubs) Planned groups for street-network, SVF, land-cover, terrain,
   and spatial-context features.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

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

    zones = compute_bcr(buildings, zones)
    zones = compute_far(buildings, zones, floor_height=floor_height)
    zones = compute_height_variability(buildings, zones)
    zones = compute_frontal_area_ratio(buildings, zones, wind_dir=wind_dir)

    feature_cols = ["bcr", "far", "sigma_h", "lambda_f"]
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
# Feature Group 3: Street-network features (stub)
# ---------------------------------------------------------------------------


def _compute_street_features(
    tiles: gpd.GeoDataFrame,
    streets: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """Street-network features per tile.

    # TODO: implement — street density, mean width, orientation entropy, etc.
    """
    logger.debug("Street features: stub — returning empty DataFrame.")
    return pd.DataFrame({"tile_id": tiles["tile_id"]})


# ---------------------------------------------------------------------------
# Feature Group 4: SVF features (stub)
# ---------------------------------------------------------------------------


def _compute_svf_features(
    tiles: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Sky View Factor statistics per tile.

    # TODO: implement — SVF mean, std, min from pre-computed rasters.
    """
    logger.debug("SVF features: stub — returning empty DataFrame.")
    return pd.DataFrame({"tile_id": tiles["tile_id"]})


# ---------------------------------------------------------------------------
# Feature Group 5: Land-cover features (stub)
# ---------------------------------------------------------------------------


def _compute_landcover_features(
    tiles: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Land-cover composition per tile.

    # TODO: implement — fraction impervious, vegetation, water, etc.
    """
    logger.debug("Land-cover features: stub — returning empty DataFrame.")
    return pd.DataFrame({"tile_id": tiles["tile_id"]})


# ---------------------------------------------------------------------------
# Feature Group 6: Terrain features (stub)
# ---------------------------------------------------------------------------


def _compute_terrain_features(
    tiles: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Terrain / DTM-derived features per tile.

    # TODO: implement — mean slope, aspect variance, elevation range, etc.
    """
    logger.debug("Terrain features: stub — returning empty DataFrame.")
    return pd.DataFrame({"tile_id": tiles["tile_id"]})


# ---------------------------------------------------------------------------
# Feature Group 7: Spatial-context features (stub)
# ---------------------------------------------------------------------------


def _compute_spatial_context_features(
    tiles: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Spatial-context features per tile.

    # TODO: implement — distance to coast, centrality, neighbour similarity, etc.
    """
    logger.debug("Spatial-context features: stub — returning empty DataFrame.")
    return pd.DataFrame({"tile_id": tiles["tile_id"]})


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_tile_features(
    tiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    streets: Optional[gpd.GeoDataFrame] = None,
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
        Street centrelines (used by group 3 when implemented).
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

    # --- Group 3: street network (stub) ---
    street_df = _compute_street_features(tiles, streets=streets)

    # --- Group 4: SVF (stub) ---
    svf_df = _compute_svf_features(tiles)

    # --- Group 5: land cover (stub) ---
    landcover_df = _compute_landcover_features(tiles)

    # --- Group 6: terrain (stub) ---
    terrain_df = _compute_terrain_features(tiles)

    # --- Group 7: spatial context (stub) ---
    context_df = _compute_spatial_context_features(tiles)

    # --- Merge all groups on tile_id ---
    group_dfs = [
        morpho_df,
        height_df,
        street_df,
        svf_df,
        landcover_df,
        terrain_df,
        context_df,
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
