"""
Facade-specific SVF and solar potential estimation.

For a vertical surface, SVF is the fraction of the hemisphere (oriented along
the outward normal) that is unobstructed.  An unobstructed vertical wall
facing open sky has SVF ~ 0.5 (upper half of the forward hemisphere is sky;
lower half is ground).
"""

import logging

import geopandas as gpd
import numpy as np
import pyvista as pv

from src.svf_v2.compute import compute_svf_raycasting, generate_sky_directions

logger = logging.getLogger(__name__)


def compute_facade_svf(
    facade_gdf: gpd.GeoDataFrame,
    scene_mesh: pv.PolyData,
    n_sky_patches: int = 145,
    max_ray_length: float = 500.0,
) -> gpd.GeoDataFrame:
    """
    Compute SVF at facade sample points.

    Uses ray-casting with normals: only sky directions in the forward
    hemisphere of each facade point are tested.

    Args:
        facade_gdf: GeoDataFrame from ``sample_facade_points()`` with
            columns x, y, z, normal_x, normal_y, normal_z.
        scene_mesh: Combined scene mesh.
        n_sky_patches: Number of sky directions.
        max_ray_length: Maximum ray distance (m).

    Returns:
        Input GeoDataFrame with an added ``svf`` column.
    """
    observer_points = np.column_stack(
        [
            facade_gdf["x"].values,
            facade_gdf["y"].values,
            facade_gdf["z"].values,
        ]
    )
    normals = np.column_stack(
        [
            facade_gdf["normal_x"].values,
            facade_gdf["normal_y"].values,
            facade_gdf["normal_z"].values,
        ]
    )

    sky_dirs = generate_sky_directions(n_sky_patches)
    svf = compute_svf_raycasting(
        observer_points,
        scene_mesh,
        sky_dirs,
        max_ray_length=max_ray_length,
        normals=normals,
    )

    facade_gdf = facade_gdf.copy()
    facade_gdf["svf"] = svf
    return facade_gdf


def compute_facade_solar_potential(
    facade_gdf: gpd.GeoDataFrame,
    latitude: float = -22.95,
    longitude: float = -43.23,
) -> gpd.GeoDataFrame:
    """
    Estimate annual solar irradiance potential per facade point.

    Combines facade SVF with orientation relative to optimal solar azimuth.
    Uses ``pvlib`` for solar position calculation.

    Args:
        facade_gdf: GeoDataFrame with ``svf`` and ``facade_azimuth`` columns.
        latitude: Site latitude (default: Vidigal).
        longitude: Site longitude (default: Vidigal).

    Returns:
        Input GeoDataFrame with added ``solar_potential`` column (relative 0-1).
    """
    try:
        import pvlib
    except ImportError:
        logger.warning("pvlib not installed -- skipping solar potential estimation")
        facade_gdf = facade_gdf.copy()
        facade_gdf["solar_potential"] = np.nan
        return facade_gdf

    import pandas as pd

    # Calculate representative solar positions across the year
    # Sample one day per month at solar noon
    times = pd.date_range("2024-01-15", periods=12, freq="MS") + pd.Timedelta(hours=12)
    sp = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    # Mean solar azimuth at noon across the year (weighted by elevation > 0)
    above_horizon = sp[sp["elevation"] > 0]
    if len(above_horizon) == 0:
        facade_gdf = facade_gdf.copy()
        facade_gdf["solar_potential"] = np.nan
        return facade_gdf

    mean_az = above_horizon["azimuth"].mean()
    mean_el = above_horizon["elevation"].mean()

    # Solar potential ~ SVF * cos(angle between facade normal and solar direction)
    facade_az = facade_gdf["facade_azimuth"].values
    az_diff = np.abs(facade_az - mean_az)
    az_diff = np.minimum(az_diff, 360 - az_diff)

    # Orientation factor: 1.0 when facing sun, 0.0 when facing away
    orientation_factor = np.maximum(0.0, np.cos(np.radians(az_diff)))

    # Elevation factor: higher sun angles give more irradiance on vertical surfaces
    # For vertical surfaces, irradiance ~ sin(elevation) * cos(azimuth_diff)
    elevation_factor = np.sin(np.radians(mean_el))

    facade_gdf = facade_gdf.copy()
    facade_gdf["solar_potential"] = (
        facade_gdf["svf"] * orientation_factor * elevation_factor
    )

    # Normalise to [0, 1]
    max_val = facade_gdf["solar_potential"].max()
    if max_val > 0:
        facade_gdf["solar_potential"] /= max_val

    logger.info(
        f"Solar potential: mean={facade_gdf['solar_potential'].mean():.3f}, "
        f"solar az={mean_az:.1f}, el={mean_el:.1f}"
    )
    return facade_gdf
