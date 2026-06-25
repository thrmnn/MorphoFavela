"""New morphometric indicators not covered by src.urban_morphology.

- Volumetric porosity
- 8-direction frontal area density (lambda_f)
- Terrain slope and aspect from DTM
- Zone mean building height
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from src.urban_morphology import compute_frontal_area_ratio

logger = logging.getLogger(__name__)

# 8 cardinal + intercardinal wind directions (degrees from north)
WIND_DIRECTIONS = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}


def compute_lambda_f_directional(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    clip_to_zone: bool = True,
    dissolve: bool = False,
) -> gpd.GeoDataFrame:
    """Compute frontal area density for 8 wind directions.

    Adds columns: lambda_f_N, lambda_f_NE, ..., lambda_f_NW,
    lambda_f_mean, lambda_f_max.

    ``clip_to_zone=True`` (default) clips footprints to the cell before
    projecting — required for small grid cells, see compute_frontal_area_ratio.
    ``dissolve=True`` unions party-walled footprints before projecting (the
    aerodynamically correct λf for fused favela fabric; see
    compute_frontal_area_ratio).
    """
    zones = zones.copy()

    if buildings.empty or "height" not in buildings.columns:
        for label in WIND_DIRECTIONS:
            zones[f"lambda_f_{label}"] = 0.0
        zones["lambda_f_mean"] = 0.0
        zones["lambda_f_max"] = 0.0
        return zones

    direction_cols = []
    for label, bearing in WIND_DIRECTIONS.items():
        col = f"lambda_f_{label}"
        result = compute_frontal_area_ratio(
            buildings, zones, wind_dir=bearing, clip_to_zone=clip_to_zone, dissolve=dissolve
        )
        zones[col] = result["lambda_f"]
        direction_cols.append(col)

    zones["lambda_f_mean"] = zones[direction_cols].mean(axis=1)
    zones["lambda_f_max"] = zones[direction_cols].max(axis=1)
    return zones


def compute_zone_mean_height(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compute mean building height per zone (H_mean).

    Also adds building_count per zone.
    """
    zones = zones.copy()

    if buildings.empty or "height" not in buildings.columns:
        zones["H_mean"] = np.nan
        zones["building_count"] = 0
        return zones

    joined = gpd.sjoin(buildings, zones, how="inner", predicate="intersects")

    h_mean = joined.groupby("zone_id")["height"].mean()
    b_count = joined.groupby("zone_id")["height"].count()

    zones["H_mean"] = zones["zone_id"].map(h_mean)
    zones["building_count"] = zones["zone_id"].map(b_count).fillna(0).astype(int)
    return zones


def compute_porosity(
    buildings: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Volumetric porosity per zone.

    porosity = 1 - sum(building_volume_in_zone) / (zone_area * H_mean)

    Where H_mean is the mean building height within the zone (defining
    the urban canopy layer height).  Zones with no buildings get porosity = 1.
    """
    zones = zones.copy()

    if buildings.empty or "height" not in buildings.columns:
        zones["porosity"] = 1.0
        return zones

    joined = gpd.sjoin(buildings, zones, how="inner", predicate="intersects")

    porosity_values: dict[int, float] = {}
    for zone_id, group in joined.groupby("zone_id"):
        zone_geom = zones.loc[zones["zone_id"] == zone_id, "geometry"].iloc[0]
        zone_area = zones.loc[zones["zone_id"] == zone_id, "zone_area"].iloc[0]
        heights = group["height"].values

        if len(heights) == 0 or zone_area <= 0:
            porosity_values[zone_id] = 1.0
            continue

        h_mean = float(np.mean(heights))
        if h_mean <= 0:
            porosity_values[zone_id] = 1.0
            continue

        # Sum building volume clipped to zone
        total_volume = 0.0
        for _, row in group.iterrows():
            intersection = row.geometry.intersection(zone_geom)
            total_volume += intersection.area * row["height"]

        canopy_volume = zone_area * h_mean
        porosity_values[zone_id] = max(0.0, 1.0 - total_volume / canopy_volume)

    zones["porosity"] = zones["zone_id"].map(porosity_values).fillna(1.0)
    return zones


def compute_slope_aspect(
    zones: gpd.GeoDataFrame,
    dtm_path: Path,
) -> gpd.GeoDataFrame:
    """Compute mean terrain slope (degrees) and aspect per zone from DTM.

    Uses numpy gradient on the DTM raster and samples within each zone.
    """
    zones = zones.copy()

    if not Path(dtm_path).exists():
        logger.warning("DTM not found at %s; slope/aspect will be NaN.", dtm_path)
        zones["slope_deg"] = np.nan
        zones["aspect_deg"] = np.nan
        return zones

    with rasterio.open(dtm_path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
        res_x, res_y = src.res

        # Mask nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan

        # Compute gradient
        dy, dx = np.gradient(dem, res_y, res_x)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = np.degrees(aspect_rad) % 360

    slope_values: dict[int, float] = {}
    aspect_values: dict[int, float] = {}

    for _, zone_row in zones.iterrows():
        zone_id = zone_row["zone_id"]
        zone_geom = zone_row.geometry

        try:
            # Get pixel mask for this zone
            with rasterio.open(dtm_path) as src:
                mask = geometry_mask(
                    [zone_geom],
                    out_shape=src.shape,
                    transform=src.transform,
                    invert=True,
                )
            zone_slope = slope_deg[mask]
            zone_aspect = aspect_deg[mask]

            # Remove NaN pixels
            valid = ~np.isnan(zone_slope)
            zone_slope = zone_slope[valid]
            zone_aspect = zone_aspect[valid]

            if len(zone_slope) > 0:
                slope_values[zone_id] = float(np.mean(zone_slope))
                # Circular mean for aspect
                sin_sum = np.sum(np.sin(np.radians(zone_aspect)))
                cos_sum = np.sum(np.cos(np.radians(zone_aspect)))
                aspect_values[zone_id] = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360)
            else:
                slope_values[zone_id] = np.nan
                aspect_values[zone_id] = np.nan
        except Exception:
            slope_values[zone_id] = np.nan
            aspect_values[zone_id] = np.nan

    zones["slope_deg"] = zones["zone_id"].map(slope_values)
    zones["aspect_deg"] = zones["zone_id"].map(aspect_values)
    return zones
