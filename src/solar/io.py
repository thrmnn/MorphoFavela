"""Solar access result export following ``src/svf_v2/io.py`` pattern.

Saves GeoPackage, CSV summary statistics, and all publication-quality
plots to a ``solar/`` subdirectory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from src.solar.sun import DEFAULT_LATITUDE, DEFAULT_LONGITUDE
from src.solar.visualize import (
    plot_solar_access,
    plot_solar_dashboard,
    plot_solar_deprivation_hero,
    plot_solar_distribution,
    plot_solar_irradiance_map,
    plot_solar_seasonal_comparison_hero,
    plot_solar_seasonal_panel,
    plot_solar_sun_path,
    plot_solar_vs_svf,
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_solar_results(
    gdf: gpd.GeoDataFrame,
    output_dir: Path,
    footprints_gdf: gpd.GeoDataFrame | None = None,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    roads_gdf: gpd.GeoDataFrame | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> None:
    """Save all solar outputs to ``output_dir/solar/``.

    Outputs
    -------
    * ``solar_access.gpkg`` -- GeoPackage with all columns.
    * ``solar_access_stats.csv`` -- Descriptive summary statistics.
    * ``solar_access.png`` -- Single-panel solar hours map.
    * ``solar_seasonal_panel.png`` -- 2x2 map per reference date
      (only if date columns present).
    * ``solar_distribution.png`` -- Histogram + KDE distribution.
    * ``solar_dashboard.png`` -- 2x2 dashboard.
    * ``solar_irradiance.png`` -- Irradiance map (only if irradiance
      columns present).
    * ``solar_sun_path.png`` -- Polar stereographic sun-path diagram.

    Parameters
    ----------
    gdf : GeoDataFrame
        Solar results with columns such as ``solar_hours_mean``,
        ``solar_hours_min``, ``solar_hours_max``, ``irradiance_mean``,
        and per-date columns ``solar_hours_{date}``.
    output_dir : Path
        Base output directory. A ``solar/`` subdirectory is created.
    footprints_gdf : GeoDataFrame, optional
        Building footprint polygons for map context.
    boundary_gdf : GeoDataFrame, optional
        Settlement boundary overlay for maps.
    roads_gdf : GeoDataFrame, optional
        Road centrelines for the dashboard map panel.
    latitude, longitude : float
        Site coordinates for the sun-path diagram.
    """
    out = _ensure_dir(Path(output_dir) / "solar")

    # ------------------------------------------------------------------
    # GeoPackage
    # ------------------------------------------------------------------
    gpkg_path = out / "solar_access.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info("Saved %s (%d points)", gpkg_path, len(gdf))

    # ------------------------------------------------------------------
    # Summary statistics CSV
    # ------------------------------------------------------------------
    stat_cols = [
        c for c in gdf.columns
        if c.startswith("solar_hours") or c.startswith("irradiance")
        or c in ("svf", "shadow_frequency", "sunshine_ratio_mean", "seasonal_range")
    ]
    if stat_cols:
        stats = gdf[stat_cols].describe().T
        csv_path = out / "solar_access_stats.csv"
        stats.to_csv(csv_path)
        logger.info("Saved %s", csv_path)

    # ------------------------------------------------------------------
    # Determine primary solar column
    # ------------------------------------------------------------------
    primary_col = "solar_hours_mean"
    if primary_col not in gdf.columns:
        primary_col = "solar_hours"

    # ------------------------------------------------------------------
    # Single-panel solar access map
    # ------------------------------------------------------------------
    if primary_col in gdf.columns:
        plot_solar_access(
            gdf, out / "solar_access.png",
            footprints_gdf=footprints_gdf,
            column=primary_col,
            boundary_gdf=boundary_gdf,
        )

    # ------------------------------------------------------------------
    # Seasonal panel (if per-date columns exist)
    # ------------------------------------------------------------------
    date_cols = sorted([c for c in gdf.columns if c.startswith("solar_hours_20")])
    if date_cols:
        plot_solar_seasonal_panel(
            gdf, out / "solar_seasonal_panel.png",
            footprints_gdf=footprints_gdf,
            boundary_gdf=boundary_gdf,
            date_columns=date_cols,
        )

    # ------------------------------------------------------------------
    # Distribution plot
    # ------------------------------------------------------------------
    if primary_col in gdf.columns:
        plot_solar_distribution(
            gdf, out / "solar_distribution.png",
            column=primary_col,
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    plot_solar_dashboard(
        gdf, out / "solar_dashboard.png",
        footprints_gdf=footprints_gdf,
        roads_gdf=roads_gdf,
        boundary_gdf=boundary_gdf,
    )

    # ------------------------------------------------------------------
    # Irradiance map (if available)
    # ------------------------------------------------------------------
    if "irradiance_mean" in gdf.columns:
        plot_solar_irradiance_map(
            gdf, out / "solar_irradiance.png",
            footprints_gdf=footprints_gdf,
            boundary_gdf=boundary_gdf,
        )

    # ------------------------------------------------------------------
    # SVF vs Solar scatter (if SVF column present)
    # ------------------------------------------------------------------
    if "svf" in gdf.columns and primary_col in gdf.columns:
        plot_solar_vs_svf(
            gdf, out / "solar_vs_svf.png",
            solar_col=primary_col,
        )

    # ------------------------------------------------------------------
    # Sun-path diagram
    # ------------------------------------------------------------------
    plot_solar_sun_path(
        latitude, longitude,
        out / "solar_sun_path.png",
    )

    logger.info("Solar results saved to %s", out)
