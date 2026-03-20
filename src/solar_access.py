"""
Solar access computation: direct-sun exposure hours via ray-casting.

Computes the number of hours each ground/street point receives direct
sunlight on a given date (typically winter solstice for worst-case analysis).
Sun positions are derived from pvlib when available, with a pure-analytical
fallback. Ray-casting uses the same pyvista/VTK infrastructure as SVF v2.

All coordinates must be in a projected CRS (metres), consistent with the
rest of the project (EPSG:31983 / SIRGAS 2000 UTM 23S).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from shapely.geometry import Point

from src.config import DPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default location: Rio de Janeiro, Brazil
DEFAULT_LATITUDE = -22.97
DEFAULT_LONGITUDE = -43.17

# Default date: Southern-Hemisphere winter solstice
DEFAULT_DATE = "2026-06-21"

# Visualization
SOLAR_CMAP = "YlOrRd_r"  # more sun = warmer colour

# Ray-casting
MAX_RAY_LENGTH = 10_000.0  # metres


# ---------------------------------------------------------------------------
# 1. Sun positions
# ---------------------------------------------------------------------------


def compute_sun_positions(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    date: str = DEFAULT_DATE,
    hour_start: int = 6,
    hour_end: int = 18,
    interval_minutes: int = 60,
) -> List[Tuple[float, float]]:
    """Compute sun positions above the horizon for a given day.

    Returns a list of ``(altitude_deg, azimuth_deg)`` tuples where the
    altitude is above 0 (sun above horizon).

    Parameters
    ----------
    latitude : float
        Site latitude in degrees (negative = south).
    longitude : float
        Site longitude in degrees (negative = west).
    date : str
        ISO date string, e.g. ``"2026-06-21"``.
    hour_start : int
        First hour of the day to consider (local solar time, 0-23).
    hour_end : int
        Last hour of the day to consider (local solar time, 0-23).
    interval_minutes : int
        Time step between sun position samples.

    Returns
    -------
    list[tuple[float, float]]
        ``(altitude_deg, azimuth_deg)`` for each position above the horizon.
    """
    try:
        return _sun_positions_pvlib(
            latitude, longitude, date, hour_start, hour_end, interval_minutes
        )
    except Exception as exc:
        logger.warning(
            "pvlib unavailable or failed (%s), using analytical fallback", exc
        )
        return _sun_positions_analytical(
            latitude, longitude, date, hour_start, hour_end, interval_minutes
        )


def _sun_positions_pvlib(
    latitude: float,
    longitude: float,
    date: str,
    hour_start: int,
    hour_end: int,
    interval_minutes: int,
) -> List[Tuple[float, float]]:
    """Compute sun positions using pvlib."""
    import pandas as pd
    import pvlib

    dt = datetime.fromisoformat(date)

    # Determine timezone from longitude (rough 15-degree offset)
    tz_offset = round(longitude / 15)
    tz_str = f"Etc/GMT{-tz_offset:+d}" if tz_offset != 0 else "UTC"

    times = pd.date_range(
        start=f"{dt.date()} {hour_start:02d}:00",
        end=f"{dt.date()} {hour_end:02d}:00",
        freq=f"{interval_minutes}min",
        tz=tz_str,
    )

    location = pvlib.location.Location(latitude, longitude, tz=tz_str)
    solar_pos = location.get_solarposition(times)

    positions: List[Tuple[float, float]] = []
    for _, row in solar_pos.iterrows():
        alt = float(row["elevation"])
        az = float(row["azimuth"])
        if alt > 0:
            positions.append((alt, az))

    logger.info(
        "pvlib: %d sun positions above horizon (lat=%.2f, lon=%.2f, %s)",
        len(positions),
        latitude,
        longitude,
        date,
    )
    return positions


def _sun_positions_analytical(
    latitude: float,
    longitude: float,
    date: str,
    hour_start: int,
    hour_end: int,
    interval_minutes: int,
) -> List[Tuple[float, float]]:
    """Simple analytical solar position (Spencer / NOAA approx)."""
    dt = datetime.fromisoformat(date)
    day_of_year = dt.timetuple().tm_yday

    # Solar declination (Spencer, 1971)
    B = math.radians(360.0 / 365.0 * (day_of_year - 81))
    declination = math.radians(
        23.45 * math.sin(B)
    )

    lat_rad = math.radians(latitude)

    positions: List[Tuple[float, float]] = []
    minutes = hour_start * 60
    end_minutes = hour_end * 60

    while minutes <= end_minutes:
        solar_hour = minutes / 60.0
        # Hour angle: 0 at solar noon (12h), 15 deg per hour
        hour_angle = math.radians(15.0 * (solar_hour - 12.0))

        # Solar altitude (elevation)
        sin_alt = (
            math.sin(lat_rad) * math.sin(declination)
            + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        sin_alt = max(-1.0, min(1.0, sin_alt))
        altitude_rad = math.asin(sin_alt)
        altitude_deg = math.degrees(altitude_rad)

        if altitude_deg > 0:
            # Solar azimuth (from north, clockwise)
            cos_az = (
                math.sin(declination) - math.sin(lat_rad) * sin_alt
            ) / (math.cos(lat_rad) * math.cos(altitude_rad) + 1e-12)
            cos_az = max(-1.0, min(1.0, cos_az))
            azimuth_deg = math.degrees(math.acos(cos_az))
            if hour_angle > 0:
                azimuth_deg = 360.0 - azimuth_deg
            positions.append((altitude_deg, azimuth_deg))

        minutes += interval_minutes

    logger.info(
        "Analytical: %d sun positions above horizon (lat=%.2f, %s)",
        len(positions),
        latitude,
        date,
    )
    return positions


# ---------------------------------------------------------------------------
# 2. Direction vector conversion
# ---------------------------------------------------------------------------


def sun_position_to_direction(altitude_deg: float, azimuth_deg: float) -> np.ndarray:
    """Convert solar altitude/azimuth to a unit direction vector in local ENU.

    Convention:
    - Azimuth: 0 = north, 90 = east, 180 = south, 270 = west (clockwise).
    - Altitude: 0 = horizon, 90 = zenith.
    - ENU axes: x = east, y = north, z = up.

    Parameters
    ----------
    altitude_deg : float
        Solar altitude in degrees (0 = horizon, 90 = zenith).
    azimuth_deg : float
        Solar azimuth in degrees from north, clockwise.

    Returns
    -------
    np.ndarray
        Unit direction vector ``(x, y, z)`` pointing *toward* the sun.
    """
    alt = np.radians(altitude_deg)
    az = np.radians(azimuth_deg)

    x = np.sin(az) * np.cos(alt)  # east
    y = np.cos(az) * np.cos(alt)  # north
    z = np.sin(alt)               # up

    vec = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_obb_tree(mesh: pv.PolyData):
    """Build a VTK OBBTree for fast ray intersection tests."""
    import vtk

    obb = vtk.vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()
    return obb


def _ray_hits(obb_tree, origin: np.ndarray, endpoint: np.ndarray) -> bool:
    """Return True if the ray from *origin* to *endpoint* hits the mesh."""
    import vtk

    pts = vtk.vtkPoints()
    cell_ids = vtk.vtkIdList()
    obb_tree.IntersectWithLine(origin.tolist(), endpoint.tolist(), pts, cell_ids)
    return pts.GetNumberOfPoints() > 0


def _compute_sun_directions(
    sun_positions: List[Tuple[float, float]],
) -> np.ndarray:
    """Convert a list of (altitude, azimuth) to Nx3 direction array."""
    dirs = np.array(
        [sun_position_to_direction(alt, az) for alt, az in sun_positions],
        dtype=np.float64,
    )
    return dirs


def _batch_solar_access(
    observer_points: np.ndarray,
    sun_directions: np.ndarray,
    scene_mesh: pv.PolyData,
    ray_length: float = MAX_RAY_LENGTH,
) -> np.ndarray:
    """Vectorised solar access: batch rays per sun position (all points at once).

    For each sun direction, casts rays from *all* observer points simultaneously
    and accumulates the unobstructed count per point.

    Parameters
    ----------
    observer_points : np.ndarray
        (N, 3) array of observer positions.
    sun_directions : np.ndarray
        (M, 3) array of unit direction vectors toward the sun.
    scene_mesh : pv.PolyData
        Scene mesh (terrain + buildings).
    ray_length : float
        Maximum ray length for intersection test.

    Returns
    -------
    np.ndarray
        Integer array of length N with the number of unobstructed sun positions.
    """
    n_pts = len(observer_points)
    n_sun = len(sun_directions)
    counts = np.zeros(n_pts, dtype=np.int32)

    # Build OBB tree once for all rays
    obb = _build_obb_tree(scene_mesh)

    logger.info(
        "Batch solar access: %d points x %d sun positions = %d rays",
        n_pts,
        n_sun,
        n_pts * n_sun,
    )

    for si, sun_dir in enumerate(sun_directions):
        logger.debug("  Sun position %d/%d", si + 1, n_sun)
        for pi in range(n_pts):
            origin = observer_points[pi]
            endpoint = origin + sun_dir * ray_length
            if not _ray_hits(obb, origin, endpoint):
                counts[pi] += 1

    return counts


# ---------------------------------------------------------------------------
# 3. Grid-based solar access
# ---------------------------------------------------------------------------


def compute_solar_access_grid(
    dtm_path: Path,
    scene_mesh: pv.PolyData,
    footprints_gdf: gpd.GeoDataFrame,
    grid_spacing: float = 5.0,
    evaluation_height: float = 1.5,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    date: str = DEFAULT_DATE,
    hour_start: int = 6,
    hour_end: int = 18,
    interval_minutes: int = 60,
    ray_length: float = MAX_RAY_LENGTH,
) -> gpd.GeoDataFrame:
    """Compute ground-level solar access on a regular grid.

    Generates ground points (excluding building interiors), then for each
    sun position casts rays from every point simultaneously, counting
    unobstructed positions per point.

    Parameters
    ----------
    dtm_path : Path
        Path to the DTM GeoTIFF raster.
    scene_mesh : pv.PolyData
        Combined scene mesh (terrain + buildings).
    footprints_gdf : gpd.GeoDataFrame
        Building footprint polygons (used to mask interior points).
    grid_spacing : float
        Distance between grid points in metres.
    evaluation_height : float
        Height above ground for the observer (metres).
    latitude, longitude : float
        Site coordinates for solar position calculation.
    date : str
        ISO date string for solar position calculation.
    hour_start, hour_end : int
        Hour window for sun position sampling.
    interval_minutes : int
        Time step between sun position samples.
    ray_length : float
        Maximum ray length for intersection tests.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``geometry``, ``x``, ``y``, ``z``, ``solar_hours``,
        ``n_sun_positions``.
    """
    from src.svf_v2.sampling import sample_grid_points

    logger.info(
        "Computing grid solar access (spacing=%.1fm, height=%.1fm)", grid_spacing, evaluation_height
    )

    # --- Generate observer points ---
    observers = sample_grid_points(
        dtm_path=dtm_path,
        footprints_gdf=footprints_gdf,
        grid_spacing=grid_spacing,
        pedestrian_height=evaluation_height,
    )
    logger.info("Grid observers: %d points", len(observers))

    if len(observers) == 0:
        logger.warning("No grid observers generated -- returning empty GeoDataFrame")
        return gpd.GeoDataFrame(
            columns=["geometry", "x", "y", "z", "solar_hours", "n_sun_positions"],
            crs=footprints_gdf.crs,
        )

    # --- Sun positions ---
    sun_positions = compute_sun_positions(
        latitude, longitude, date, hour_start, hour_end, interval_minutes
    )
    n_sun = len(sun_positions)
    if n_sun == 0:
        logger.warning("No sun positions above horizon -- all solar_hours will be 0")

    sun_dirs = _compute_sun_directions(sun_positions)
    timestep_hours = interval_minutes / 60.0

    # --- Ray-cast ---
    counts = _batch_solar_access(observers, sun_dirs, scene_mesh, ray_length)
    solar_hours = counts.astype(np.float64) * timestep_hours

    # --- Assemble GeoDataFrame ---
    # Observer z includes evaluation_height; record ground z
    ground_z = observers[:, 2] - evaluation_height

    gdf = gpd.GeoDataFrame(
        {
            "x": observers[:, 0],
            "y": observers[:, 1],
            "z": ground_z,
            "solar_hours": solar_hours,
            "n_sun_positions": n_sun,
        },
        geometry=[Point(x, y) for x, y in zip(observers[:, 0], observers[:, 1])],
        crs=footprints_gdf.crs,
    )

    logger.info(
        "Grid solar access complete: mean=%.2fh, min=%.2fh, max=%.2fh",
        solar_hours.mean(),
        solar_hours.min(),
        solar_hours.max(),
    )
    return gdf


# ---------------------------------------------------------------------------
# 4. Street-based solar access
# ---------------------------------------------------------------------------


def compute_solar_access_streets(
    street_gdf: gpd.GeoDataFrame,
    scene_mesh: pv.PolyData,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    date: str = DEFAULT_DATE,
    hour_start: int = 6,
    hour_end: int = 18,
    interval_minutes: int = 60,
    evaluation_height: float = 1.5,
    ray_length: float = MAX_RAY_LENGTH,
) -> gpd.GeoDataFrame:
    """Compute solar access for pre-sampled street points.

    Expects *street_gdf* to contain a ``z_observer`` column (as produced
    by ``svf_v2.sampling.sample_street_points``).  If ``z_observer`` is
    absent, falls back to ``z`` + *evaluation_height*, then to
    *evaluation_height* alone.

    Parameters
    ----------
    street_gdf : gpd.GeoDataFrame
        Street sample points with ``geometry`` (Point), ``z`` or
        ``z_observer`` columns.
    scene_mesh : pv.PolyData
        Combined scene mesh (terrain + buildings).
    latitude, longitude : float
        Site coordinates for solar position calculation.
    date : str
        ISO date string.
    hour_start, hour_end : int
        Hour window.
    interval_minutes : int
        Time step.
    evaluation_height : float
        Fallback height above ground when ``z_observer`` is unavailable.
    ray_length : float
        Maximum ray length.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of *street_gdf* with ``solar_hours`` and ``n_sun_positions``
        columns added.
    """
    street_gdf = street_gdf.copy()
    n_pts = len(street_gdf)
    logger.info("Computing street solar access for %d points", n_pts)

    if n_pts == 0:
        street_gdf["solar_hours"] = np.float64(0)
        street_gdf["n_sun_positions"] = 0
        return street_gdf

    # Build observer array
    xs = np.array([p.x for p in street_gdf.geometry], dtype=np.float64)
    ys = np.array([p.y for p in street_gdf.geometry], dtype=np.float64)

    if "z_observer" in street_gdf.columns:
        zs = street_gdf["z_observer"].values.astype(np.float64)
    elif "z" in street_gdf.columns:
        zs = street_gdf["z"].values.astype(np.float64) + evaluation_height
    else:
        logger.warning("No z/z_observer column -- using evaluation_height=%.1f", evaluation_height)
        zs = np.full(n_pts, evaluation_height, dtype=np.float64)

    observers = np.column_stack([xs, ys, zs])

    # Sun positions
    sun_positions = compute_sun_positions(
        latitude, longitude, date, hour_start, hour_end, interval_minutes
    )
    n_sun = len(sun_positions)
    sun_dirs = _compute_sun_directions(sun_positions)
    timestep_hours = interval_minutes / 60.0

    # Ray-cast
    counts = _batch_solar_access(observers, sun_dirs, scene_mesh, ray_length)
    solar_hours = counts.astype(np.float64) * timestep_hours

    street_gdf["solar_hours"] = solar_hours
    street_gdf["n_sun_positions"] = n_sun

    logger.info(
        "Street solar access complete: mean=%.2fh, min=%.2fh, max=%.2fh",
        solar_hours.mean(),
        solar_hours.min(),
        solar_hours.max(),
    )
    return street_gdf


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------


def plot_solar_access(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    footprints_gdf: Optional[gpd.GeoDataFrame] = None,
    title: str = "Solar Access",
    cmap: str = SOLAR_CMAP,
    column: str = "solar_hours",
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = DPI,
) -> None:
    """Create a publication-quality solar access map.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Solar access results with a ``solar_hours`` column.
    output_path : Path
        File path for the saved figure (e.g. ``.png`` / ``.pdf``).
    footprints_gdf : gpd.GeoDataFrame, optional
        Building footprints for context.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap name.
    column : str
        Column to visualise.
    figsize : tuple
        ``(width, height)`` in inches.
    dpi : int
        Output resolution.
    """
    from src.cartography import (
        add_north_arrow,
        add_scale_bar,
        apply_publication_style,
        format_utm_axes,
        style_buildings_by_height,
    )

    apply_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Building context
    if footprints_gdf is not None and len(footprints_gdf) > 0:
        style_buildings_by_height(ax, footprints_gdf)

    # Solar access points
    vmin = 0.0
    vmax = gdf[column].max() if len(gdf) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        markersize=4,
        legend=True,
        legend_kwds={
            "label": "Hours of Direct Sunlight",
            "shrink": 0.7,
        },
        zorder=5,
    )

    # Cartographic elements
    add_scale_bar(ax)
    add_north_arrow(ax)
    format_utm_axes(ax)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved solar access map to %s", output_path)
