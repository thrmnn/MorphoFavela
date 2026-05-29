"""
Per-timestep sunlit GIS layer for dataviz animations.

This module is a thin wrapper around ``src.solar.compute.compute_sunlit_matrix``
that reshapes its boolean ``(N points, M sun positions)`` output into a
"wide" GeoDataFrame with one ``lit_T{HHMM}`` column per timestep, plus a
sidecar manifest summarising the sun position and sunlit count for each
frame.

The aggregated outputs from ``compute_solar_access_grid`` (one ``solar_hours``
float per point) are unchanged — this module exists so the dataviz team can
animate the shadow sweep frame-by-frame without re-running the ray cast.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib
from shapely.geometry import Point

from src.solar.sun import sun_position_to_direction


def compute_sun_positions_with_times(
    latitude: float,
    longitude: float,
    date: str,
    hour_start: int = 7,
    hour_end: int = 17,
    interval_minutes: int = 60,
    timezone: str = "America/Sao_Paulo",
) -> list[dict]:
    """Sun positions above the horizon, paired with local timestamps.

    Unlike ``src.solar.sun.compute_sun_positions``, this returns the local
    time alongside ``(altitude, azimuth)`` so we can label animation frames.
    Falls back to a daylight-window scan if pvlib is unavailable.

    Returns
    -------
    list[dict]
        Each entry has ``time`` (pd.Timestamp, tz-aware), ``altitude_deg``,
        ``azimuth_deg``, and ``direction`` (unit ENU vector toward the sun).
    """
    dt = datetime.fromisoformat(date)
    times = pd.date_range(
        start=f"{dt.date()} {hour_start:02d}:00",
        end=f"{dt.date()} {hour_end:02d}:00",
        freq=f"{interval_minutes}min",
        tz=timezone,
    )

    location = pvlib.location.Location(latitude, longitude, tz=timezone)
    solar_pos = location.get_solarposition(times)

    out: list[dict] = []
    for ts, row in solar_pos.iterrows():
        alt = float(row["elevation"])
        az = float(row["azimuth"])
        if alt <= 0:
            continue
        out.append(
            {
                "time": ts,
                "altitude_deg": alt,
                "azimuth_deg": az,
                "direction": sun_position_to_direction(alt, az),
            }
        )
    return out


def _timestamp_label(ts: pd.Timestamp) -> str:
    """``2026-06-21 07:00:00-03:00`` -> ``T0700``."""
    return f"T{ts.strftime('%H%M')}"


def sunlit_matrix_to_wide_gdf(
    observers: np.ndarray,
    sunlit_matrix: np.ndarray,
    sun_frames: Sequence[dict],
    crs=None,
    *,
    interval_minutes: int,
    include_solar_hours: bool = True,
    base_gdf: gpd.GeoDataFrame | None = None,
    solar_hours_col: str = "solar_hours",
) -> tuple[gpd.GeoDataFrame, dict]:
    """Reshape an (N, M) sunlit matrix into a wide-format GeoDataFrame.

    There are two output modes:

    * **Grid mode** (``base_gdf`` is ``None``): the function constructs a fresh
      GeoDataFrame with ``geometry``, ``x``, ``y``, ``z`` (from ``observers``)
      and the per-frame ``lit_T{HHMM}`` columns.  ``crs`` must be supplied.
    * **Append mode** (``base_gdf`` is provided): the function returns a *copy*
      of ``base_gdf`` with the per-frame columns appended.  Geometry, CRS, and
      any pre-existing columns (e.g. ``svf``, ``solar_hours_winter``) are
      preserved.  Row order in ``base_gdf`` must match the rows of
      ``observers`` / ``sunlit_matrix``.

    Parameters
    ----------
    observers
        ``(N, 3)`` observer xyz array.
    sunlit_matrix
        ``(N, M)`` boolean matrix from ``compute_sunlit_matrix``.
    sun_frames
        ``M`` entries as returned by ``compute_sun_positions_with_times``;
        their order must match ``sunlit_matrix``'s columns.
    crs
        Target CRS for the GeoDataFrame (grid mode only — ignored in append
        mode, where the CRS comes from ``base_gdf``).
    interval_minutes
        Time step between frames (used to derive the ``solar_hours`` column).
    include_solar_hours
        If ``True``, add a ``solar_hours``-style float column = row sum × step.
        In append mode the column is named ``solar_hours_col`` to avoid
        clobbering pre-existing columns.
    base_gdf
        Optional pre-built GeoDataFrame to append columns onto.  Must have
        ``len(base_gdf) == N``.
    solar_hours_col
        Column name for the convenience sum-of-lit-hours.  Default
        ``"solar_hours"`` (grid mode); the script uses ``"lit_hours_total"``
        in append mode so it doesn't shadow ``solar_hours_winter``.

    Returns
    -------
    (gdf, manifest)
        ``gdf`` has all the per-frame ``lit_T{HHMM}`` columns plus the
        convenience hours column when requested.  ``manifest`` is a
        JSON-serialisable dict.
    """
    n_pts, n_sun = sunlit_matrix.shape
    if n_sun != len(sun_frames):
        raise ValueError(
            f"sunlit_matrix has {n_sun} columns but {len(sun_frames)} sun frames provided"
        )
    if observers.shape[0] != n_pts:
        raise ValueError(f"observers has {observers.shape[0]} rows but sunlit_matrix has {n_pts}")
    if base_gdf is not None and len(base_gdf) != n_pts:
        raise ValueError(f"base_gdf has {len(base_gdf)} rows but sunlit_matrix has {n_pts}")

    timestep_hours = interval_minutes / 60.0

    if base_gdf is not None:
        gdf = base_gdf.copy()
    else:
        if crs is None:
            raise ValueError("crs is required when base_gdf is not provided")
        columns_seed = {
            "x": observers[:, 0],
            "y": observers[:, 1],
            "z": observers[:, 2],
        }
        geometry = [Point(x, y) for x, y in zip(observers[:, 0], observers[:, 1])]
        gdf = gpd.GeoDataFrame(columns_seed, geometry=geometry, crs=crs)

    if include_solar_hours:
        gdf[solar_hours_col] = sunlit_matrix.sum(axis=1).astype(np.float64) * timestep_hours

    frame_records: list[dict] = []
    seen_labels: set[str] = set()
    for i, frame in enumerate(sun_frames):
        label = _timestamp_label(frame["time"])
        if label in seen_labels:
            # Disambiguate frames that share the same HHMM (sub-hourly close together).
            label = f"{label}_{i:02d}"
        seen_labels.add(label)
        col_name = f"lit_{label}"
        col_values = sunlit_matrix[:, i].astype(bool)
        gdf[col_name] = col_values
        frame_records.append(
            {
                "col": col_name,
                "time_local": frame["time"].strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sun_alt_deg": round(float(frame["altitude_deg"]), 3),
                "sun_az_deg": round(float(frame["azimuth_deg"]), 3),
                "n_sunlit": int(col_values.sum()),
            }
        )

    manifest = {
        "n_points": int(n_pts),
        "n_frames": int(n_sun),
        "interval_minutes": int(interval_minutes),
        "frames": frame_records,
    }
    return gdf, manifest


def build_animation_manifest(
    *,
    site: str,
    date: str,
    timezone: str,
    crs: str,
    hour_start: int,
    hour_end: int,
    frame_manifest: dict,
    observer_source: str | None = None,
    grid_spacing_m: float | None = None,
    evaluation_height_m: float | None = None,
) -> dict:
    """Wrap the per-frame manifest with run-level provenance.

    Either ``observer_source`` (street-points mode) or ``grid_spacing_m``
    (grid mode) should be set so the dataviz team knows where the
    geometry came from.
    """
    manifest = {
        "site": site,
        "date": date,
        "timezone": timezone,
        "crs": crs,
        "hour_window_local": [int(hour_start), int(hour_end)],
    }
    if observer_source is not None:
        manifest["observer_source"] = observer_source
    if grid_spacing_m is not None:
        manifest["grid_spacing_m"] = float(grid_spacing_m)
    if evaluation_height_m is not None:
        manifest["evaluation_height_m"] = float(evaluation_height_m)
    manifest.update(frame_manifest)
    return manifest


__all__ = [
    "compute_sun_positions_with_times",
    "sunlit_matrix_to_wide_gdf",
    "build_animation_manifest",
]
