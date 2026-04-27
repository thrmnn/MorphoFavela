"""Grid-based morphometric computation at arbitrary resolution.

Builds a regular grid over a study area and computes per-cell morphometric
indicators by composing functions from ``src.urban_morphology`` and the
new indicator functions in ``src.morphometry.indicators``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np

from src.morphometry.indicators import (
    compute_lambda_f_directional,
    compute_porosity,
    compute_slope_aspect,
    compute_zone_mean_height,
)
from src.urban_morphology import (
    compute_bcr,
    compute_far,
    compute_height_variability,
    create_analysis_zones,
)

logger = logging.getLogger(__name__)


def _aggregate_svf_to_grid(
    svf_points: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    svf_col: str = "svf",
) -> gpd.GeoDataFrame:
    """Aggregate point-level SVF values to grid cells.

    For each zone, computes the mean SVF of all sample points falling
    within the cell.  Cells with no SVF points receive NaN.
    """
    zones = zones.copy()

    if svf_points is None or svf_points.empty:
        zones["svf"] = np.nan
        zones["svf_count"] = 0
        return zones

    joined = gpd.sjoin(
        svf_points[[svf_col, "geometry"]], zones, how="inner", predicate="within"
    )

    stats = joined.groupby("zone_id")[svf_col].agg(["mean", "count"])
    zones["svf"] = zones["zone_id"].map(stats["mean"])
    zones["svf_count"] = zones["zone_id"].map(stats["count"]).fillna(0).astype(int)
    return zones


def _clip_streets_to_zones(
    streets: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    n_bins: int = 36,
) -> gpd.GeoDataFrame:
    """Compute per-zone street orientation entropy."""
    from src.urban_morphology import _compute_zone_entropy

    return _compute_zone_entropy(streets, zones, n_bins=n_bins)


def compute_grid_morphometrics(
    buildings: gpd.GeoDataFrame,
    dtm_path: Path,
    cell_size: float = 10.0,
    svf_points: Optional[gpd.GeoDataFrame] = None,
    streets: Optional[gpd.GeoDataFrame] = None,
    boundary: Optional[gpd.GeoDataFrame] = None,
    floor_height: float = 3.0,
) -> gpd.GeoDataFrame:
    """Compute all morphometric indicators on a regular grid.

    Parameters
    ----------
    buildings : GeoDataFrame
        Building footprints with ``height`` column.
    dtm_path : Path
        Path to DTM raster (GeoTIFF).
    cell_size : float
        Grid cell size in metres (default 10).
    svf_points : GeoDataFrame, optional
        Point-level SVF results (with ``svf`` column) to aggregate.
    streets : GeoDataFrame, optional
        Street centerlines for orientation entropy.
    boundary : GeoDataFrame, optional
        Study area boundary for clipping.
    floor_height : float
        Assumed floor height for FAR computation.

    Returns
    -------
    GeoDataFrame
        Grid cells with all morphometric columns.
    """
    logger.info("Computing grid morphometrics at %dm resolution...", cell_size)

    # Repair invalid geometries (critical for areas like Rocinha with self-intersections)
    bld = buildings.copy()
    invalid_mask = ~bld.geometry.is_valid
    if invalid_mask.any():
        logger.warning("Repairing %d invalid building geometries.", invalid_mask.sum())
        bld.loc[invalid_mask, "geometry"] = bld.loc[invalid_mask, "geometry"].buffer(0)

    # Ensure buildings have a height column
    if "height" not in bld.columns:
        if "altura" in bld.columns and "base" in bld.columns:
            bld["height"] = bld["altura"]
            bld["base_height"] = bld["base"]
            bld["top_height"] = bld["base"] + bld["altura"]
        elif "altura" in bld.columns:
            bld["height"] = bld["altura"]
        else:
            logger.warning(
                "No height column found; height-dependent metrics will be NaN."
            )

    # Create grid
    zones = create_analysis_zones(bld, method="grid", cell_size=cell_size)

    # Clip to boundary if provided
    if boundary is not None:
        boundary_union = boundary.geometry.union_all()
        zones = zones[zones.intersects(boundary_union)].copy()
        zones = zones.reset_index(drop=True)
        zones["zone_id"] = range(len(zones))
        logger.info("Clipped grid to boundary: %d cells", len(zones))

    # 1. BCR (lambda_p) — capped at 1.0 (overlapping footprints can exceed)
    logger.info("  Computing BCR (lambda_p)...")
    zones = compute_bcr(bld, zones)
    zones = zones.rename(columns={"bcr": "lambda_p"})
    zones["lambda_p"] = zones["lambda_p"].clip(upper=1.0)

    # 2. FAR
    logger.info("  Computing FAR...")
    zones = compute_far(bld, zones, floor_height=floor_height)

    # 3. Height variability (sigma_h) + mean height
    logger.info("  Computing height variability...")
    zones = compute_height_variability(bld, zones)
    zones = compute_zone_mean_height(bld, zones)

    # 4. Frontal area density — 8 directions
    logger.info("  Computing 8-direction frontal area density...")
    zones = compute_lambda_f_directional(bld, zones)

    # 5. Porosity
    logger.info("  Computing porosity...")
    zones = compute_porosity(bld, zones)

    # 6. Terrain slope and aspect
    logger.info("  Computing terrain slope and aspect...")
    zones = compute_slope_aspect(zones, dtm_path)

    # 7. SVF aggregation
    if svf_points is not None:
        logger.info("  Aggregating SVF to grid cells...")
        zones = _aggregate_svf_to_grid(svf_points, zones)
    else:
        zones["svf"] = np.nan
        zones["svf_count"] = 0

    # 8. Street orientation entropy (per zone)
    if streets is not None:
        logger.info("  Computing per-zone street orientation entropy...")
        zones = _clip_streets_to_zones(streets, zones)
    else:
        zones["street_orientation_entropy"] = np.nan

    # Add centroid coordinates
    centroids = zones.geometry.centroid
    zones["centroid_x"] = centroids.x
    zones["centroid_y"] = centroids.y

    logger.info(
        "Grid morphometrics complete: %d cells, %d columns.",
        len(zones),
        len(zones.columns),
    )
    return zones
