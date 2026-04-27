"""
Solar access computation: sunlit matrix via ray-casting and grid/street wrappers.

The sunlit matrix records, for every observer point and every sun position,
whether the point receives direct beam radiation (True) or is shadowed by the
scene geometry (False).  This is the foundation for both solar-hour counts and
physically-based irradiance integration.

Parallelization follows the same joblib pattern as ``svf_v2.compute``: the
scene mesh is written to a temporary VTK file so each worker process can
rebuild its own OBB tree without pickling VTK objects.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pyvista as pv
from shapely.geometry import Point

from src.solar.irradiance import compute_daily_irradiance_array
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

MAX_RAY_LENGTH = 10_000.0  # metres


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_obb_tree(mesh: pv.PolyData):
    """Build a VTK OBBTree for fast ray intersection tests."""
    from vtkmodules.vtkFiltersGeneral import vtkOBBTree

    obb = vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()
    return obb


def _ray_hits(
    obb_tree,
    origin: np.ndarray,
    endpoint: np.ndarray,
    min_hit_distance: float = 0.0,
) -> bool:
    """Return True if the ray from *origin* to *endpoint* hits the mesh.

    Parameters
    ----------
    min_hit_distance : float
        Ignore intersections closer than this distance (metres) to avoid
        self-intersection with the observer's own building geometry.
    """
    import vtk

    pts = vtk.vtkPoints()
    cell_ids = vtk.vtkIdList()
    obb_tree.IntersectWithLine(origin.tolist(), endpoint.tolist(), pts, cell_ids)
    n_hits = pts.GetNumberOfPoints()
    if n_hits == 0:
        return False
    if min_hit_distance <= 0.0:
        return True
    # Filter out self-intersections: only count hits beyond min distance
    origin_arr = np.asarray(origin, dtype=np.float64)
    for i in range(n_hits):
        hit = np.array(pts.GetPoint(i))
        if np.linalg.norm(hit - origin_arr) > min_hit_distance:
            return True
    return False


def _compute_sun_directions(
    sun_positions: List[Tuple[float, float]],
) -> np.ndarray:
    """Convert a list of (altitude, azimuth) to Nx3 direction array."""
    dirs = np.array(
        [sun_position_to_direction(alt, az) for alt, az in sun_positions],
        dtype=np.float64,
    )
    return dirs


# ---------------------------------------------------------------------------
# Sunlit matrix — core computation
# ---------------------------------------------------------------------------


def _sunlit_matrix_chunk(
    point_indices: np.ndarray,
    observer_points: np.ndarray,
    sun_directions: np.ndarray,
    ray_length: float,
    mesh_file: str,
    min_hit_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Worker: compute sunlit matrix rows for a chunk of point indices.

    Each worker rebuilds the OBB tree from the temporary mesh file so that
    no VTK objects need to be pickled across process boundaries.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(point_indices, sunlit_chunk)`` where sunlit_chunk has shape
        ``(len(point_indices), n_sun_directions)``.
    """
    mesh = pv.read(mesh_file)
    obb = _build_obb_tree(mesh)
    n_sun = len(sun_directions)
    chunk_size = len(point_indices)
    sunlit = np.zeros((chunk_size, n_sun), dtype=np.bool_)

    for k, pi in enumerate(point_indices):
        origin = observer_points[pi]
        for si in range(n_sun):
            endpoint = origin + sun_directions[si] * ray_length
            if not _ray_hits(obb, origin, endpoint, min_hit_distance):
                sunlit[k, si] = True

    return point_indices, sunlit


def compute_sunlit_matrix(
    observer_points: np.ndarray,
    sun_directions: np.ndarray,
    scene_mesh: pv.PolyData,
    ray_length: float = MAX_RAY_LENGTH,
    n_jobs: int = 1,
    min_hit_distance: float = 0.0,
) -> np.ndarray:
    """Compute the sunlit boolean matrix via OBB-tree ray-casting.

    For each observer point and each sun direction, casts a ray and records
    whether the path to the sun is unobstructed.

    Parameters
    ----------
    observer_points : np.ndarray
        (N, 3) array of observer positions.
    sun_directions : np.ndarray
        (M, 3) array of unit direction vectors toward the sun.
    scene_mesh : pv.PolyData
        Combined scene mesh (terrain + buildings).
    ray_length : float
        Maximum ray length for intersection tests (metres).
    n_jobs : int
        Number of parallel workers.  ``1`` = sequential,
        ``-1`` = all available cores.
    min_hit_distance : float
        Ignore ray intersections closer than this distance (metres).
        Use >0 for facade points to avoid self-intersection with the
        observer's own building roof.  Default 0.0 (no filtering).

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N, M)`` where True means the point is
        sunlit for that sun position.
    """
    n_pts = len(observer_points)
    n_sun = len(sun_directions)

    # Resolve n_jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, n_jobs)

    logger.info(
        "Computing sunlit matrix: %d points x %d sun positions = %d rays (n_jobs=%d)",
        n_pts,
        n_sun,
        n_pts * n_sun,
        n_jobs,
    )

    # ------------------------------------------------------------------
    # Parallel path
    # ------------------------------------------------------------------
    if n_jobs > 1:
        from joblib import Parallel, delayed

        tmp_dir = tempfile.mkdtemp(prefix="solar_parallel_")
        mesh_file = os.path.join(tmp_dir, "scene.vtk")
        scene_mesh.save(mesh_file)

        try:
            all_indices = np.arange(n_pts)
            chunks = np.array_split(all_indices, n_jobs)
            chunks = [c for c in chunks if len(c) > 0]

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_sunlit_matrix_chunk)(
                    chunk,
                    observer_points,
                    sun_directions,
                    ray_length,
                    mesh_file,
                    min_hit_distance,
                )
                for chunk in chunks
            )

            sunlit = np.zeros((n_pts, n_sun), dtype=np.bool_)
            for chunk_indices, chunk_sunlit in results:
                sunlit[chunk_indices] = chunk_sunlit

        finally:
            try:
                os.remove(mesh_file)
                os.rmdir(tmp_dir)
            except OSError:
                pass

        logger.info(
            "Sunlit matrix complete: %.1f%% sunlit overall",
            100.0 * sunlit.sum() / max(sunlit.size, 1),
        )
        return sunlit

    # ------------------------------------------------------------------
    # Sequential path
    # ------------------------------------------------------------------
    obb = _build_obb_tree(scene_mesh)
    sunlit = np.zeros((n_pts, n_sun), dtype=np.bool_)

    for pi in range(n_pts):
        origin = observer_points[pi]
        for si in range(n_sun):
            endpoint = origin + sun_directions[si] * ray_length
            if not _ray_hits(obb, origin, endpoint, min_hit_distance):
                sunlit[pi, si] = True

        if (pi + 1) % 100 == 0 or pi == n_pts - 1:
            logger.debug("  Sunlit matrix: %d/%d points", pi + 1, n_pts)

    logger.info(
        "Sunlit matrix complete: %.1f%% sunlit overall",
        100.0 * sunlit.sum() / max(sunlit.size, 1),
    )
    return sunlit


# ---------------------------------------------------------------------------
# Grid-based solar access
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
    interval_minutes: int = 30,
    ray_length: float = MAX_RAY_LENGTH,
    n_jobs: int = 1,
    svf_array: Optional[np.ndarray] = None,
    site_elevation_m: float = 100.0,
) -> gpd.GeoDataFrame:
    """Compute ground-level solar access on a regular grid.

    Generates ground points (excluding building interiors), computes the
    sunlit matrix, and derives solar hours and irradiance metrics.

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
        Time step between sun position samples (default 30 min).
    ray_length : float
        Maximum ray length for intersection tests.
    n_jobs : int
        Number of parallel workers for ray-casting.
    svf_array : np.ndarray, optional
        Pre-computed SVF values for each grid point.  If provided, used
        for diffuse irradiance weighting; otherwise SVF=1.0 is assumed.
    site_elevation_m : float
        Site elevation for irradiance model (metres).

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``geometry``, ``x``, ``y``, ``z``, ``solar_hours``,
        ``irradiance_wh``, ``sunshine_ratio``, ``shadow_frequency``,
        ``n_sun_positions``.
    """
    from src.svf_v2.sampling import sample_grid_points

    logger.info(
        "Computing grid solar access (spacing=%.1fm, height=%.1fm)",
        grid_spacing,
        evaluation_height,
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
            columns=[
                "geometry",
                "x",
                "y",
                "z",
                "solar_hours",
                "irradiance_wh",
                "sunshine_ratio",
                "shadow_frequency",
                "n_sun_positions",
            ],
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

    # --- Sunlit matrix ---
    sunlit_matrix = compute_sunlit_matrix(
        observers,
        sun_dirs,
        scene_mesh,
        ray_length,
        n_jobs=n_jobs,
    )

    # --- Solar hours ---
    counts = sunlit_matrix.sum(axis=1)  # (N,) int
    solar_hours = counts.astype(np.float64) * timestep_hours

    # --- Sunshine ratio & shadow frequency ---
    if n_sun > 0:
        sunshine_ratio = counts.astype(np.float64) / n_sun
        shadow_frequency = 1.0 - sunshine_ratio
    else:
        sunshine_ratio = np.zeros(len(observers), dtype=np.float64)
        shadow_frequency = np.ones(len(observers), dtype=np.float64)

    # --- Irradiance (Wh/m^2/day) ---
    if svf_array is None:
        svf_vals = np.ones(len(observers), dtype=np.float64)
    else:
        svf_vals = np.asarray(svf_array, dtype=np.float64)

    day_of_year = datetime.fromisoformat(date).timetuple().tm_yday

    if n_sun > 0:
        irradiance_wh = compute_daily_irradiance_array(
            sun_positions=sun_positions,
            sunlit_matrix=sunlit_matrix,
            svf_array=svf_vals,
            interval_minutes=interval_minutes,
            day_of_year=day_of_year,
            site_elevation_m=site_elevation_m,
        )
    else:
        irradiance_wh = np.zeros(len(observers), dtype=np.float64)

    # --- Assemble GeoDataFrame ---
    ground_z = observers[:, 2] - evaluation_height

    gdf = gpd.GeoDataFrame(
        {
            "x": observers[:, 0],
            "y": observers[:, 1],
            "z": ground_z,
            "solar_hours": solar_hours,
            "irradiance_wh": irradiance_wh,
            "sunshine_ratio": sunshine_ratio,
            "shadow_frequency": shadow_frequency,
            "n_sun_positions": n_sun,
        },
        geometry=[Point(x, y) for x, y in zip(observers[:, 0], observers[:, 1])],
        crs=footprints_gdf.crs,
    )

    logger.info(
        "Grid solar access complete: mean=%.2fh, min=%.2fh, max=%.2fh, "
        "mean_irradiance=%.0f Wh/m2",
        solar_hours.mean(),
        solar_hours.min(),
        solar_hours.max(),
        irradiance_wh.mean(),
    )
    return gdf


# ---------------------------------------------------------------------------
# Street-based solar access
# ---------------------------------------------------------------------------


def compute_solar_access_streets(
    street_gdf: gpd.GeoDataFrame,
    scene_mesh: pv.PolyData,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    date: str = DEFAULT_DATE,
    hour_start: int = 6,
    hour_end: int = 18,
    interval_minutes: int = 30,
    evaluation_height: float = 1.5,
    ray_length: float = MAX_RAY_LENGTH,
    n_jobs: int = 1,
    svf_col: Optional[str] = None,
    site_elevation_m: float = 100.0,
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
        Time step (default 30 min).
    evaluation_height : float
        Fallback height above ground when ``z_observer`` is unavailable.
    ray_length : float
        Maximum ray length.
    n_jobs : int
        Number of parallel workers for ray-casting.
    svf_col : str, optional
        Name of column in *street_gdf* containing pre-computed SVF values.
        Used for diffuse irradiance weighting.  If None, SVF=1.0 is assumed.
    site_elevation_m : float
        Site elevation for irradiance model (metres).

    Returns
    -------
    gpd.GeoDataFrame
        Copy of *street_gdf* with columns added: ``solar_hours``,
        ``irradiance_wh``, ``sunshine_ratio``, ``shadow_frequency``,
        ``n_sun_positions``.
    """
    street_gdf = street_gdf.copy()
    n_pts = len(street_gdf)
    logger.info("Computing street solar access for %d points", n_pts)

    if n_pts == 0:
        street_gdf["solar_hours"] = np.float64(0)
        street_gdf["irradiance_wh"] = np.float64(0)
        street_gdf["sunshine_ratio"] = np.float64(0)
        street_gdf["shadow_frequency"] = np.float64(1)
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
        logger.warning(
            "No z/z_observer column -- using evaluation_height=%.1f", evaluation_height
        )
        zs = np.full(n_pts, evaluation_height, dtype=np.float64)

    observers = np.column_stack([xs, ys, zs])

    # Sun positions
    sun_positions = compute_sun_positions(
        latitude, longitude, date, hour_start, hour_end, interval_minutes
    )
    n_sun = len(sun_positions)
    sun_dirs = _compute_sun_directions(sun_positions)
    timestep_hours = interval_minutes / 60.0

    # Sunlit matrix
    sunlit_matrix = compute_sunlit_matrix(
        observers,
        sun_dirs,
        scene_mesh,
        ray_length,
        n_jobs=n_jobs,
    )

    # Solar hours
    counts = sunlit_matrix.sum(axis=1)
    solar_hours = counts.astype(np.float64) * timestep_hours

    # Sunshine ratio & shadow frequency
    if n_sun > 0:
        sunshine_ratio = counts.astype(np.float64) / n_sun
        shadow_frequency = 1.0 - sunshine_ratio
    else:
        sunshine_ratio = np.zeros(n_pts, dtype=np.float64)
        shadow_frequency = np.ones(n_pts, dtype=np.float64)

    # Irradiance (Wh/m^2/day)
    if svf_col is not None and svf_col in street_gdf.columns:
        svf_vals = street_gdf[svf_col].values.astype(np.float64)
    else:
        svf_vals = np.ones(n_pts, dtype=np.float64)

    day_of_year = datetime.fromisoformat(date).timetuple().tm_yday

    if n_sun > 0:
        irradiance_wh = compute_daily_irradiance_array(
            sun_positions=sun_positions,
            sunlit_matrix=sunlit_matrix,
            svf_array=svf_vals,
            interval_minutes=interval_minutes,
            day_of_year=day_of_year,
            site_elevation_m=site_elevation_m,
        )
    else:
        irradiance_wh = np.zeros(n_pts, dtype=np.float64)

    # Assign results
    street_gdf["solar_hours"] = solar_hours
    street_gdf["irradiance_wh"] = irradiance_wh
    street_gdf["sunshine_ratio"] = sunshine_ratio
    street_gdf["shadow_frequency"] = shadow_frequency
    street_gdf["n_sun_positions"] = n_sun

    logger.info(
        "Street solar access complete: mean=%.2fh, min=%.2fh, max=%.2fh, "
        "mean_irradiance=%.0f Wh/m2",
        solar_hours.mean(),
        solar_hours.min(),
        solar_hours.max(),
        irradiance_wh.mean(),
    )
    return street_gdf
