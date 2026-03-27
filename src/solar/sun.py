"""
Sun position computation for urban solar access analysis.

Provides solar altitude/azimuth via pvlib (preferred) or a pure-analytical
fallback (Spencer / NOAA approximation).  All azimuths follow the meteorological
convention: 0 = north, 90 = east, 180 = south, clockwise.

Multi-day helpers produce positions for reference dates (solstices, equinoxes)
used in seasonal irradiance integration.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_DATES = {
    "winter_solstice": "2026-06-21",
    "summer_solstice": "2026-12-21",
    "equinox_march": "2026-03-20",
    "equinox_september": "2026-09-22",
}

DEFAULT_LATITUDE = -22.97  # Rio de Janeiro
DEFAULT_LONGITUDE = -43.17

DEFAULT_DATE = "2026-06-21"  # Southern-Hemisphere winter solstice


# ---------------------------------------------------------------------------
# Core: single-day sun positions
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
    declination = math.radians(23.45 * math.sin(B))

    lat_rad = math.radians(latitude)

    positions: List[Tuple[float, float]] = []
    minutes = hour_start * 60
    end_minutes = hour_end * 60

    while minutes <= end_minutes:
        solar_hour = minutes / 60.0
        # Hour angle: 0 at solar noon (12h), 15 deg per hour
        hour_angle = math.radians(15.0 * (solar_hour - 12.0))

        # Solar altitude (elevation)
        sin_alt = math.sin(lat_rad) * math.sin(declination) + math.cos(
            lat_rad
        ) * math.cos(declination) * math.cos(hour_angle)
        sin_alt = max(-1.0, min(1.0, sin_alt))
        altitude_rad = math.asin(sin_alt)
        altitude_deg = math.degrees(altitude_rad)

        if altitude_deg > 0:
            # Solar azimuth (from north, clockwise)
            cos_az = (math.sin(declination) - math.sin(lat_rad) * sin_alt) / (
                math.cos(lat_rad) * math.cos(altitude_rad) + 1e-12
            )
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
# Direction vector conversion
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
    z = np.sin(alt)  # up

    vec = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Multi-day helper
# ---------------------------------------------------------------------------


def compute_sun_positions_multi_day(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    dates: list[str] | None = None,
    hour_start: int = 5,
    hour_end: int = 19,
    interval_minutes: int = 30,
) -> dict[str, list[tuple[float, float]]]:
    """Compute sun positions for multiple dates.

    Useful for seasonal analysis (solstices, equinoxes) where irradiance is
    integrated over several reference days.

    Parameters
    ----------
    latitude, longitude : float
        Site coordinates (defaults to Rio de Janeiro).
    dates : list[str] | None
        ISO date strings.  Defaults to ``REFERENCE_DATES.values()``.
    hour_start, hour_end : int
        Hour window for sun position sampling.
    interval_minutes : int
        Time step between samples (default 30 min for irradiance accuracy).

    Returns
    -------
    dict[str, list[tuple[float, float]]]
        Mapping from date string to list of ``(altitude_deg, azimuth_deg)``.
    """
    if dates is None:
        dates = list(REFERENCE_DATES.values())

    result: dict[str, list[tuple[float, float]]] = {}
    for date in dates:
        positions = compute_sun_positions(
            latitude=latitude,
            longitude=longitude,
            date=date,
            hour_start=hour_start,
            hour_end=hour_end,
            interval_minutes=interval_minutes,
        )
        result[date] = positions
        logger.info("Date %s: %d sun positions above horizon", date, len(positions))

    return result


# ---------------------------------------------------------------------------
# Extra-terrestrial irradiance
# ---------------------------------------------------------------------------


def compute_extraterrestrial_irradiance(day_of_year: int) -> float:
    """Solar constant corrected for Earth-Sun distance (Spencer 1971).

    Parameters
    ----------
    day_of_year : int
        Day of the year (1-365).

    Returns
    -------
    float
        Extra-terrestrial irradiance in W/m^2.
    """
    SOLAR_CONSTANT = 1361.0  # W/m^2
    B = 2 * np.pi * (day_of_year - 1) / 365
    return float(
        SOLAR_CONSTANT
        * (
            1.00011
            + 0.034221 * np.cos(B)
            + 0.00128 * np.sin(B)
            + 0.000719 * np.cos(2 * B)
            + 0.000077 * np.sin(2 * B)
        )
    )
