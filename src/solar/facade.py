"""
Facade-level solar irradiance and WHO 2-hour sunlight threshold assessment.

Extends the horizontal-surface irradiance models in ``irradiance.py`` with
tilted-surface geometry for vertical facades.  The incident angle for a
vertical surface with outward-normal azimuth *gamma* receiving beam radiation
from a sun at altitude *alpha* and azimuth *A_s* is::

    cos(theta_i) = cos(alpha) * cos(A_s - gamma)

Only rays where the sun is in the forward hemisphere of the facade
(cos(theta_i) > 0) contribute direct beam irradiance.

References
----------
- Duffie, J.A. & Beckman, W.A. (2013). *Solar Engineering of Thermal
  Processes*. 4th ed., Wiley. Ch. 1-2.
- WHO (2018). *Housing and Health Guidelines*. Minimum 2 hours direct
  sunlight at winter solstice.
- Liu, B.Y.H. & Jordan, R.C. (1960). Isotropic diffuse model for
  tilted surfaces.
"""

from __future__ import annotations

import logging
from datetime import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
import pyvista as pv

from src.solar.irradiance import (
    diffuse_horizontal_irradiance,
    direct_normal_irradiance,
)
from src.solar.sun import (
    DEFAULT_DATE,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    compute_sun_positions,
    sun_position_to_direction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHO_SUNLIGHT_THRESHOLD_HOURS = 2.0
"""WHO minimum daily direct sunlight for healthy housing (hours)."""

MAX_RAY_LENGTH = 10_000.0
"""Default ray length for sunlit-matrix computation (metres)."""

# ---------------------------------------------------------------------------
# Section A: Tilted-surface irradiance geometry
# ---------------------------------------------------------------------------


def compute_facade_incident_angle(
    sun_altitude_deg: float,
    sun_azimuth_deg: float,
    facade_azimuth_deg: float,
) -> float:
    """Cosine of the incidence angle on a vertical facade.

    For a vertical surface (tilt = 90 deg) with outward-normal azimuth
    *gamma*, the incidence angle simplifies to::

        cos(theta_i) = cos(alpha) * cos(A_s - gamma)

    Parameters
    ----------
    sun_altitude_deg : float
        Solar altitude above the horizon (degrees, 0-90).
    sun_azimuth_deg : float
        Solar azimuth from north, clockwise (degrees, 0-360).
    facade_azimuth_deg : float
        Facade outward-normal azimuth from north (degrees, 0-360).

    Returns
    -------
    float
        cos(theta_i).  Positive when the sun is in front of the facade.
    """
    alt_rad = np.radians(sun_altitude_deg)
    az_diff_rad = np.radians(sun_azimuth_deg - facade_azimuth_deg)
    return float(np.cos(alt_rad) * np.cos(az_diff_rad))


def direct_beam_on_facade(
    dni: float,
    sun_altitude_deg: float,
    sun_azimuth_deg: float,
    facade_azimuth_deg: float,
) -> float:
    """Direct beam irradiance on a vertical facade (W/m^2).

    Returns ``DNI * max(0, cos(theta_i))``.  Zero when the sun is
    behind the facade.
    """
    cos_theta = compute_facade_incident_angle(
        sun_altitude_deg, sun_azimuth_deg, facade_azimuth_deg
    )
    return dni * max(0.0, cos_theta)


def diffuse_on_tilted_surface(
    diffuse_horizontal: float,
    tilt_deg: float = 90.0,
) -> float:
    """Isotropic diffuse irradiance on a tilted surface (Liu-Jordan).

    ``I_d_tilted = I_d_horizontal * (1 + cos(tilt)) / 2``

    For vertical surfaces (tilt = 90 deg) the factor is 0.5.
    """
    tilt_rad = np.radians(tilt_deg)
    return diffuse_horizontal * (1.0 + np.cos(tilt_rad)) / 2.0


# ---------------------------------------------------------------------------
# Section B: Vectorized facade daily irradiance
# ---------------------------------------------------------------------------


def compute_facade_daily_irradiance_array(
    sun_positions: list[tuple[float, float]],
    sunlit_matrix: np.ndarray,
    facade_azimuths: np.ndarray,
    svf_array: np.ndarray,
    interval_minutes: int,
    day_of_year: int,
    site_elevation_m: float = 100.0,
) -> np.ndarray:
    """Vectorised daily irradiance for *N* facade points (Wh/m^2/day).

    Key differences from :func:`irradiance.compute_daily_irradiance_array`:

    * Uses the incident-angle formula for vertical surfaces instead of
      ``sin(altitude)``.
    * Applies per-facade azimuth filtering — direct beam only counted when
      the sun is in front of the facade (``cos(theta_i) > 0``).
    * Diffuse tilt factor = 0.5 (Liu-Jordan for vertical surface).

    Parameters
    ----------
    sun_positions : list of (altitude_deg, azimuth_deg)
        *M* timesteps above the horizon.
    sunlit_matrix : ndarray (N, M)
        Boolean — ``True`` if facade point *i* is geometrically un-
        obstructed at timestep *j* (from ray-casting).
    facade_azimuths : ndarray (N,)
        Facade outward-normal azimuth (degrees from north).
    svf_array : ndarray (N,)
        Sky View Factor for diffuse weighting.
    interval_minutes : int
        Duration of each timestep.
    day_of_year : int
        For extraterrestrial-irradiance correction.
    site_elevation_m : float
        For Hottel transmittance coefficients.

    Returns
    -------
    ndarray (N,)
        Daily facade irradiation in Wh/m^2/day.
    """
    N = len(facade_azimuths)
    M = len(sun_positions)
    hours_per_step = interval_minutes / 60.0

    # Pre-compute per-timestep scalars
    dni_arr = np.zeros(M)
    diffuse_h_arr = np.zeros(M)
    alt_rad = np.zeros(M)
    az_rad = np.zeros(M)

    for j, (alt, az) in enumerate(sun_positions):
        dni_arr[j] = direct_normal_irradiance(alt, day_of_year, site_elevation_m)
        diffuse_h_arr[j] = diffuse_horizontal_irradiance(
            alt, day_of_year, site_elevation_m
        )
        alt_rad[j] = np.radians(alt)
        az_rad[j] = np.radians(az)

    facade_az_rad = np.radians(facade_azimuths)  # (N,)

    # Accumulate direct component
    direct_total = np.zeros(N, dtype=np.float64)
    for j in range(M):
        cos_theta = np.cos(alt_rad[j]) * np.cos(az_rad[j] - facade_az_rad)  # (N,)
        sun_in_front = cos_theta > 0  # (N,)
        effective_sunlit = sunlit_matrix[:, j] & sun_in_front  # (N,)
        direct_total += effective_sunlit * cos_theta * dni_arr[j]

    # Diffuse component: isotropic Liu-Jordan for vertical (factor = 0.5)
    diffuse_total_scalar = float(diffuse_h_arr.sum()) * 0.5
    diffuse_total = svf_array * diffuse_total_scalar

    irradiance_wh = (direct_total + diffuse_total) * hours_per_step
    return irradiance_wh


# ---------------------------------------------------------------------------
# Section C: Facade sunlit hours with normal filtering
# ---------------------------------------------------------------------------


def compute_facade_sunlit_hours(
    sunlit_matrix: np.ndarray,
    sun_positions: list[tuple[float, float]],
    facade_azimuths: np.ndarray,
    interval_minutes: int,
) -> np.ndarray:
    """Facade-specific direct-sunlight hours with normal filtering.

    A facade point is "sunlit" at timestep *j* only if:

    1. ``sunlit_matrix[i, j]`` is ``True`` (geometrically unobstructed), AND
    2. ``cos(theta_i) > 0`` (sun is in front of the facade).

    Parameters
    ----------
    sunlit_matrix : ndarray (N, M)
        From :func:`solar.compute.compute_sunlit_matrix`.
    sun_positions : list of (altitude_deg, azimuth_deg)
        *M* timesteps.
    facade_azimuths : ndarray (N,)
        Facade outward-normal azimuth (degrees from north).
    interval_minutes : int
        Duration of each timestep.

    Returns
    -------
    ndarray (N,)
        Hours of direct sunlight per facade point.
    """
    N = len(facade_azimuths)
    hours_per_step = interval_minutes / 60.0

    facade_az_rad = np.radians(facade_azimuths)  # (N,)
    total_hours = np.zeros(N, dtype=np.float64)

    for j, (alt, az) in enumerate(sun_positions):
        cos_theta = np.cos(np.radians(alt)) * np.cos(np.radians(az) - facade_az_rad)
        sun_in_front = cos_theta > 0
        effective = sunlit_matrix[:, j] & sun_in_front
        total_hours += effective * hours_per_step

    return total_hours


# ---------------------------------------------------------------------------
# Section D: WHO threshold assessment
# ---------------------------------------------------------------------------


def assess_who_threshold(
    facade_sunlit_hours: np.ndarray,
    threshold_hours: float = WHO_SUNLIGHT_THRESHOLD_HOURS,
) -> np.ndarray:
    """Binary WHO 2-hour threshold classification.

    Parameters
    ----------
    facade_sunlit_hours : ndarray (N,)
        Direct sunlight hours per facade point.
    threshold_hours : float
        Minimum required hours (default 2.0).

    Returns
    -------
    ndarray (N,) of bool
        ``True`` = compliant, ``False`` = deprived.
    """
    return facade_sunlit_hours >= threshold_hours


def compute_floor_level(
    height_above_ground: np.ndarray,
    floor_height: float = 2.5,
) -> np.ndarray:
    """Assign floor level (0-indexed) from height above ground.

    ``floor = floor(height_above_ground / floor_height)``
    """
    return np.floor(height_above_ground / floor_height).astype(int)


# ---------------------------------------------------------------------------
# Section E: Building / floor aggregation
# ---------------------------------------------------------------------------


def aggregate_by_building(
    facade_gdf: gpd.GeoDataFrame,
    sunlit_hours_col: str = "facade_sunlit_hours",
    irradiance_col: str = "facade_irradiance_wh",
    who_col: str = "who_compliant",
) -> pd.DataFrame:
    """Building-level summary from facade point data.

    Groups by ``building_id`` and computes:
      - ``sunlit_hours_mean``, ``sunlit_hours_min``, ``sunlit_hours_max``
      - ``irradiance_wh_mean``
      - ``who_compliance_rate`` (fraction of facade points meeting threshold)
      - ``n_facade_points``
    """
    grouped = facade_gdf.groupby("building_id")

    agg = grouped.agg(
        sunlit_hours_mean=(sunlit_hours_col, "mean"),
        sunlit_hours_min=(sunlit_hours_col, "min"),
        sunlit_hours_max=(sunlit_hours_col, "max"),
        irradiance_wh_mean=(irradiance_col, "mean"),
        who_compliance_rate=(who_col, "mean"),
        n_facade_points=(sunlit_hours_col, "count"),
    )

    return agg.reset_index()


def aggregate_by_building_floor(
    facade_gdf: gpd.GeoDataFrame,
    sunlit_hours_col: str = "facade_sunlit_hours",
    irradiance_col: str = "facade_irradiance_wh",
    who_col: str = "who_compliant",
) -> pd.DataFrame:
    """Per (building_id, floor_level) summary.

    Returns DataFrame with columns:
      - ``sunlit_hours_mean``, ``irradiance_wh_mean``
      - ``who_compliance_rate``
      - ``n_points``
    """
    grouped = facade_gdf.groupby(["building_id", "floor_level"])

    agg = grouped.agg(
        sunlit_hours_mean=(sunlit_hours_col, "mean"),
        irradiance_wh_mean=(irradiance_col, "mean"),
        who_compliance_rate=(who_col, "mean"),
        n_points=(sunlit_hours_col, "count"),
    )

    return agg.reset_index()


def summarize_housing_units(
    floor_agg: pd.DataFrame,
    who_col: str = "who_compliance_rate",
    threshold: float = 0.5,
) -> dict:
    """Summarize housing unit WHO compliance.

    Each (building_id, floor_level) row = one housing unit.
    A unit is considered below WHO if fewer than *threshold* fraction
    of its facade points meet the 2-hour minimum.

    Returns
    -------
    dict
        ``total_units``, ``units_below_who``, ``pct_below``.
    """
    total = len(floor_agg)
    below = int((floor_agg[who_col] < threshold).sum())
    return {
        "total_units": total,
        "units_below_who": below,
        "pct_below": 100.0 * below / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Section F: End-to-end orchestrator
# ---------------------------------------------------------------------------


def compute_facade_solar_access(
    facade_gdf: gpd.GeoDataFrame,
    scene_mesh: pv.PolyData,
    date: str = DEFAULT_DATE,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    hour_start: int = 6,
    hour_end: int = 18,
    interval_minutes: int = 30,
    ray_length: float = MAX_RAY_LENGTH,
    n_jobs: int = 1,
    site_elevation_m: float = 100.0,
    floor_height: float = 2.5,
    who_threshold_hours: float = WHO_SUNLIGHT_THRESHOLD_HOURS,
    min_hit_distance: float = 0.5,
) -> gpd.GeoDataFrame:
    """Full facade solar access pipeline.

    Steps:
      1. Compute sun positions for the given date.
      2. Build sunlit matrix via ray-casting (reuses existing infrastructure).
      3. Apply normal filtering -> facade sunlit hours.
      4. Compute tilted-surface daily irradiance.
      5. WHO threshold assessment.
      6. Floor-level assignment.

    Parameters
    ----------
    facade_gdf : GeoDataFrame
        From :func:`svf_v2.sampling.sample_facade_points`.  Required columns:
        ``x``, ``y``, ``z``, ``facade_azimuth``, ``building_id``,
        ``height_above_ground``.  Optional: ``svf`` (if pre-computed).
    scene_mesh : pv.PolyData
        Combined terrain + building mesh.
    date : str
        ISO date for analysis (default: winter solstice for Rio).
    latitude, longitude : float
        Site coordinates.
    hour_start, hour_end : int
        Analysis window (solar time).
    interval_minutes : int
        Timestep (default 30 min).
    ray_length : float
        Maximum ray length for sunlit matrix (metres).
    n_jobs : int
        Parallel workers for ray-casting (1 = sequential).
    site_elevation_m : float
        Site elevation for Hottel model.
    floor_height : float
        Assumed floor-to-floor height for floor level assignment (metres).
    who_threshold_hours : float
        WHO minimum sunlight threshold (default 2.0 hours).

    Returns
    -------
    GeoDataFrame
        Input with added columns: ``facade_sunlit_hours``,
        ``facade_irradiance_wh``, ``who_compliant``, ``floor_level``.
    """
    from src.solar.compute import compute_sunlit_matrix

    facade_gdf = facade_gdf.copy()

    # --- 1. Sun positions ---------------------------------------------------
    sun_positions = compute_sun_positions(
        latitude,
        longitude,
        date,
        hour_start=hour_start,
        hour_end=hour_end,
        interval_minutes=interval_minutes,
    )
    if not sun_positions:
        logger.warning("No sun positions above horizon for %s.", date)
        facade_gdf["facade_sunlit_hours"] = 0.0
        facade_gdf["facade_irradiance_wh"] = 0.0
        facade_gdf["who_compliant"] = False
        facade_gdf["floor_level"] = 0
        return facade_gdf

    logger.info(
        "Sun positions: %d timesteps (%s, lat=%.2f)",
        len(sun_positions),
        date,
        latitude,
    )

    # --- 2. Sun direction vectors -------------------------------------------
    sun_dirs = np.array(
        [sun_position_to_direction(alt, az) for alt, az in sun_positions],
        dtype=np.float64,
    )

    # --- 3. Observer points -------------------------------------------------
    observer_points = np.column_stack(
        [
            facade_gdf["x"].values,
            facade_gdf["y"].values,
            facade_gdf["z"].values,
        ]
    )

    # --- 4. Sunlit matrix (existing, unchanged) -----------------------------
    logger.info(
        "Computing sunlit matrix: %d facade points x %d sun positions...",
        len(observer_points),
        len(sun_dirs),
    )
    sunlit_matrix = compute_sunlit_matrix(
        observer_points,
        sun_dirs,
        scene_mesh,
        ray_length=ray_length,
        n_jobs=n_jobs,
        min_hit_distance=min_hit_distance,
    )

    # --- 5. Facade sunlit hours (with normal filtering) --------------------
    facade_azimuths = facade_gdf["facade_azimuth"].values
    facade_sunlit_hours = compute_facade_sunlit_hours(
        sunlit_matrix,
        sun_positions,
        facade_azimuths,
        interval_minutes,
    )
    facade_gdf["facade_sunlit_hours"] = facade_sunlit_hours

    # --- 6. Facade irradiance -----------------------------------------------
    # Parse day of year
    dt = datetime.strptime(date, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday

    svf_array = (
        facade_gdf["svf"].values
        if "svf" in facade_gdf.columns
        else np.ones(len(facade_gdf))
    )

    facade_irradiance = compute_facade_daily_irradiance_array(
        sun_positions,
        sunlit_matrix,
        facade_azimuths,
        svf_array,
        interval_minutes,
        doy,
        site_elevation_m,
    )
    facade_gdf["facade_irradiance_wh"] = facade_irradiance

    # --- 7. WHO threshold ---------------------------------------------------
    facade_gdf["who_compliant"] = assess_who_threshold(
        facade_sunlit_hours,
        who_threshold_hours,
    )

    # --- 8. Floor level -----------------------------------------------------
    facade_gdf["floor_level"] = compute_floor_level(
        facade_gdf["height_above_ground"].values,
        floor_height,
    )

    # --- Summary logging ----------------------------------------------------
    n_compliant = facade_gdf["who_compliant"].sum()
    n_total = len(facade_gdf)
    logger.info(
        "Facade solar access complete: %d/%d points meet WHO %.1fh threshold (%.1f%%).",
        n_compliant,
        n_total,
        who_threshold_hours,
        100.0 * n_compliant / n_total if n_total > 0 else 0.0,
    )
    logger.info(
        "  Sunlit hours: mean=%.2f, min=%.2f, max=%.2f",
        facade_sunlit_hours.mean(),
        facade_sunlit_hours.min(),
        facade_sunlit_hours.max(),
    )
    logger.info(
        "  Irradiance: mean=%.0f Wh/m2/day",
        facade_irradiance.mean(),
    )

    return facade_gdf
