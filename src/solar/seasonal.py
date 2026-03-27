"""Multi-day seasonal solar analysis orchestration.

Runs solar access ray-casting for multiple reference dates (solstices,
equinoxes) and produces per-point summary statistics suitable for
comparative seasonal analysis and irradiance estimation.
"""

from __future__ import annotations

import logging

import numpy as np
import pyvista as pv

from src.solar.sun import (
    compute_sun_positions,
    sun_position_to_direction,
    REFERENCE_DATES,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)
from src.solar.compute import compute_sunlit_matrix
from src.solar.irradiance import compute_daily_irradiance_array

logger = logging.getLogger(__name__)


def compute_seasonal_solar(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    svf_array: np.ndarray | None = None,
    dates: list[str] | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    interval_minutes: int = 30,
    n_jobs: int = 1,
    ray_length: float = 10_000.0,
    site_elevation_m: float = 100.0,
) -> dict:
    """Run solar access for multiple reference dates.

    For each date:
    1. Compute sun positions above the horizon.
    2. Compute the sunlit boolean matrix via ``compute_sunlit_matrix()``.
    3. If *svf_array* is provided, compute irradiance via
       ``compute_daily_irradiance_array()``.
    4. Derive ``solar_hours = sunlit_matrix.sum(axis=1) * interval_minutes / 60``.

    Parameters
    ----------
    observer_points : np.ndarray
        (N, 3) xyz observer positions in a projected CRS (metres).
    scene_mesh : pv.PolyData
        Combined scene mesh (terrain + buildings) for ray-casting.
    svf_array : np.ndarray | None
        (N,) Sky View Factor values used for diffuse irradiance estimation.
        If *None*, irradiance columns are omitted.
    dates : list[str] | None
        ISO date strings to analyse.  Defaults to the four
        ``REFERENCE_DATES`` values (two solstices, two equinoxes).
    latitude, longitude : float
        Site coordinates (defaults to Rio de Janeiro).
    interval_minutes : int
        Time step between sun position samples (default 30 min).
    n_jobs : int
        Number of parallel workers passed to ``compute_sunlit_matrix``.
    ray_length : float
        Maximum ray length for intersection tests (metres).
    site_elevation_m : float
        Mean site elevation above sea level (metres), passed to irradiance
        model for air-mass correction.

    Returns
    -------
    dict
        ``"dates"`` maps each date string to a sub-dict containing
        ``solar_hours``, ``irradiance_wh`` (if SVF provided),
        ``sunlit_matrix``, and ``sun_positions``.
        ``"summary"`` holds cross-date aggregated arrays.
    """
    if dates is None:
        dates = list(REFERENCE_DATES.values())

    n_pts = len(observer_points)
    logger.info(
        "Seasonal solar analysis: %d points, %d dates, %d-min interval",
        n_pts, len(dates), interval_minutes,
    )

    date_results: dict[str, dict] = {}
    all_solar_hours: list[np.ndarray] = []
    all_irradiance: list[np.ndarray] = []
    all_sunshine_ratios: list[np.ndarray] = []
    total_timesteps = 0
    total_shaded_counts = np.zeros(n_pts, dtype=np.float64)

    for date in dates:
        logger.info("Processing date %s ...", date)

        # 1. Sun positions
        sun_positions = compute_sun_positions(
            latitude=latitude,
            longitude=longitude,
            date=date,
            hour_start=5,
            hour_end=19,
            interval_minutes=interval_minutes,
        )
        n_sun = len(sun_positions)
        logger.info("  %d sun positions above horizon", n_sun)

        if n_sun == 0:
            solar_hours = np.zeros(n_pts, dtype=np.float64)
            sunlit_matrix = np.zeros((n_pts, 0), dtype=bool)
        else:
            # 2. Convert sun positions to direction vectors and compute sunlit matrix
            sun_directions = np.array(
                [sun_position_to_direction(alt, az) for alt, az in sun_positions],
                dtype=np.float64,
            )
            sunlit_matrix = compute_sunlit_matrix(
                observer_points=observer_points,
                sun_directions=sun_directions,
                scene_mesh=scene_mesh,
                ray_length=ray_length,
                n_jobs=n_jobs,
            )

            # 4. Solar hours
            solar_hours = (
                sunlit_matrix.sum(axis=1).astype(np.float64)
                * interval_minutes / 60.0
            )

        # Track shadow frequency across all dates
        if n_sun > 0:
            shaded_this_date = (~sunlit_matrix).sum(axis=1).astype(np.float64)
            total_shaded_counts += shaded_this_date
            total_timesteps += n_sun

        # Sunshine ratio for this date
        possible_hours = n_sun * interval_minutes / 60.0
        if possible_hours > 0:
            sunshine_ratio = solar_hours / possible_hours
        else:
            sunshine_ratio = np.zeros(n_pts, dtype=np.float64)
        all_sunshine_ratios.append(sunshine_ratio)

        entry: dict = {
            "solar_hours": solar_hours,
            "sunlit_matrix": sunlit_matrix,
            "sun_positions": sun_positions,
        }

        # 3. Irradiance (optional)
        if svf_array is not None and n_sun > 0:
            from datetime import datetime as _dt
            _day_of_year = _dt.fromisoformat(date).timetuple().tm_yday
            irradiance_wh = compute_daily_irradiance_array(
                sun_positions=sun_positions,
                sunlit_matrix=sunlit_matrix,
                svf_array=svf_array,
                interval_minutes=interval_minutes,
                day_of_year=_day_of_year,
                site_elevation_m=site_elevation_m,
            )
            entry["irradiance_wh"] = irradiance_wh
            all_irradiance.append(irradiance_wh)
        elif svf_array is not None:
            entry["irradiance_wh"] = np.zeros(n_pts, dtype=np.float64)
            all_irradiance.append(np.zeros(n_pts, dtype=np.float64))

        date_results[date] = entry
        all_solar_hours.append(solar_hours)

        logger.info(
            "  Solar hours: mean=%.2f, min=%.2f, max=%.2f",
            solar_hours.mean(), solar_hours.min(), solar_hours.max(),
        )

    # ------------------------------------------------------------------
    # Summary statistics across dates
    # ------------------------------------------------------------------
    hours_stack = np.column_stack(all_solar_hours)  # (N, n_dates)

    summary: dict[str, np.ndarray] = {
        "solar_hours_mean": hours_stack.mean(axis=1),
        "solar_hours_min": hours_stack.min(axis=1),
        "solar_hours_max": hours_stack.max(axis=1),
        "seasonal_range": hours_stack.max(axis=1) - hours_stack.min(axis=1),
    }

    # Mean sunshine ratio across dates
    ratio_stack = np.column_stack(all_sunshine_ratios)
    summary["sunshine_ratio_mean"] = ratio_stack.mean(axis=1)

    # Shadow frequency: fraction of all timesteps shaded across all dates
    if total_timesteps > 0:
        summary["shadow_frequency"] = total_shaded_counts / total_timesteps
    else:
        summary["shadow_frequency"] = np.ones(n_pts, dtype=np.float64)

    # Irradiance summary
    if all_irradiance:
        irr_stack = np.column_stack(all_irradiance)
        summary["irradiance_mean"] = irr_stack.mean(axis=1)
    else:
        summary["irradiance_mean"] = np.zeros(n_pts, dtype=np.float64)

    logger.info(
        "Seasonal summary: mean solar hours=%.2f, seasonal range=%.2f",
        summary["solar_hours_mean"].mean(),
        summary["seasonal_range"].mean(),
    )

    return {
        "dates": date_results,
        "summary": summary,
    }
