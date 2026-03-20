"""
Solar access analysis package.

Provides sun position computation, clear-sky irradiance models, and
ray-casting-based sunlit/shadow determination for urban environments.
"""

from src.solar.sun import (
    compute_sun_positions,
    sun_position_to_direction,
    compute_sun_positions_multi_day,
    compute_extraterrestrial_irradiance,
    REFERENCE_DATES,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)
from src.solar.irradiance import (
    direct_normal_irradiance,
    diffuse_horizontal_irradiance,
    compute_instantaneous_ghi,
    integrate_daily_irradiance,
    compute_daily_irradiance_array,
)
from src.solar.compute import (
    compute_sunlit_matrix,
    compute_solar_access_grid,
    compute_solar_access_streets,
)

__all__ = [
    # sun
    "compute_sun_positions",
    "sun_position_to_direction",
    "compute_sun_positions_multi_day",
    "compute_extraterrestrial_irradiance",
    "REFERENCE_DATES",
    "DEFAULT_LATITUDE",
    "DEFAULT_LONGITUDE",
    # irradiance
    "direct_normal_irradiance",
    "diffuse_horizontal_irradiance",
    "compute_instantaneous_ghi",
    "integrate_daily_irradiance",
    "compute_daily_irradiance_array",
    # compute
    "compute_sunlit_matrix",
    "compute_solar_access_grid",
    "compute_solar_access_streets",
]
