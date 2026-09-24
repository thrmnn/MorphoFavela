"""P-05 — building shade per OM2 point per 5-minute interval.

Campaign dates are NOT known yet (PI, 2026-09-23), so v0.1 ships this as a
function + CLI that takes an explicit list of dates and a time window —
never a guessed campaign date. Running it for real needs a DSM built over
the OM2 route corridor (src/brisa_solar/wp02_surface.build_surface) plus a
horizon-angle march (src/brisa_solar/wp02_horizon.patch_visibility(...,
return_horizon=True)), the same engine WP-04 uses for its direct-sun-hours
number (wp04_sites.py) — that is a real, GPU/CPU-costed pipeline step, not
"trivially cheap", so v0.1 ships the wiring (this module) and an
EMPTY-SCHEMA output (see ``build_empty_shade_table``) rather than a live
run against an unvalidated demo date. Once real campaign dates land, call
``compute_shade`` with them.

Tree shade is PENDING — no DSM/canopy layer for Maré on disk; there is no
tree-shade column in v0.1, only a data-dictionary PENDING row.

Method (compute_shade, once a horizon profile is supplied):
  1. sun_positions(): pvlib solar position (altitude, azimuth) for every
     5-min step in the window, for each requested date, at the OM2
     route's own coordinates (site is small enough that Maré's single
     Galeão-EPW sun position is used for every point, same simplification
     WP-04/WP-05 make for a single site).
  2. is_shaded(): a point is shaded at time t if the sun's altitude at t
     is at or below the marched horizon angle at the sun's azimuth (the
     nearest-building silhouette in that direction blocks it), or if the
     sun is below the geometric horizon (altitude <= 0, night).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHADE_TABLE_COLUMNS = [
    "point_id",
    "timestamp",
    "date",
    "sun_altitude_deg",
    "sun_azimuth_deg",
    "shaded",
]


#: Field campaign clock times are assumed local Rio de Janeiro time, not
#: UTC — matches the Octopus GPS timestamp convention question in
#: OCTOPUS_JOIN_EXAMPLE below (confirm before joining real campaign data).
CAMPAIGN_TZ = "America/Sao_Paulo"


def sun_positions(
    dates: list[str], time_window: tuple[str, str], step_min: int, lat: float, lon: float, tz: str = CAMPAIGN_TZ
) -> pd.DataFrame:
    """pvlib solar position at (lat, lon) for every step_min interval in
    time_window (local clock time, tz), on each date. Returns columns:
    timestamp (tz-aware), date, altitude_deg, azimuth_deg."""
    import pvlib

    start_h, end_h = time_window
    rows = []
    for d in dates:
        idx = pd.date_range(f"{d} {start_h}", f"{d} {end_h}", freq=f"{step_min}min", tz=tz)
        solpos = pvlib.solarposition.get_solarposition(idx, lat, lon)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "date": d,
                    "sun_altitude_deg": solpos["apparent_elevation"].to_numpy(),
                    "sun_azimuth_deg": solpos["azimuth"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["timestamp", "date", "sun_altitude_deg", "sun_azimuth_deg"]
    )


def is_shaded(sun_altitude_deg: np.ndarray, sun_azimuth_deg: np.ndarray, horizon_deg: np.ndarray, horizon_azimuths_deg: np.ndarray) -> np.ndarray:
    """Vectorised nearest-azimuth horizon comparison (same rule WP-04 uses
    against pvlib: docstring of wp02_horizon.patch_visibility)."""
    az_diff = np.abs(sun_azimuth_deg[:, None] - horizon_azimuths_deg[None, :]) % 360
    az_diff = np.minimum(az_diff, 360 - az_diff)
    nearest = np.argmin(az_diff, axis=1)
    horizon_at_sun_az = horizon_deg[nearest]
    return (sun_altitude_deg <= 0) | (sun_altitude_deg <= horizon_at_sun_az)


def point_horizon_profiles(points_gdf, paths, device: str | None = None):
    """Marched horizon-angle profile per OM2 point (real engine call — not
    exercised by v0.1's default build; see module docstring). Requires
    src/brisa_solar/wp02_surface.build_surface over a DSM covering the
    route corridor and src/brisa_solar/wp02_horizon.patch_visibility(...,
    return_horizon=True). Returns (horizon_deg [n_points, n_az],
    azimuths_deg [n_az])."""
    raise NotImplementedError(
        "point_horizon_profiles needs a DSM built over the OM2 corridor "
        "(wp02_surface.build_surface) and a horizon march "
        "(wp02_horizon.patch_visibility) — a real but non-trivial pipeline "
        "step out of scope for v0.1's default build. Wire this in when "
        "campaign dates are known and the run is worth costing."
    )


def compute_shade(
    points_gdf,
    dates: list[str],
    time_window: tuple[str, str],
    step_min: int,
    lat: float,
    lon: float,
    horizon_deg: np.ndarray | None = None,
    horizon_azimuths_deg: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full P-05 table: one row per (point, 5-min timestamp). Requires a
    horizon profile per point (see point_horizon_profiles) — pass it in
    once computed; this function does not compute it itself."""
    if horizon_deg is None or horizon_azimuths_deg is None:
        raise ValueError(
            "compute_shade needs horizon_deg/horizon_azimuths_deg from "
            "point_horizon_profiles(); v0.1 has not run that (see module docstring)."
        )
    sp = sun_positions(dates, time_window, step_min, lat, lon)
    rows = []
    for pid, h in zip(points_gdf["point_id"], horizon_deg):
        shaded = is_shaded(sp["sun_altitude_deg"].to_numpy(), sp["sun_azimuth_deg"].to_numpy(), h, horizon_azimuths_deg)
        df = sp.copy()
        df["point_id"] = pid
        df["shaded"] = shaded
        rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SHADE_TABLE_COLUMNS)
    return out[SHADE_TABLE_COLUMNS]


def build_empty_shade_table() -> pd.DataFrame:
    """v0.1's shipped P-05 table: correct schema, zero rows — campaign
    dates unknown, live horizon-march run out of scope for v0.1."""
    return pd.DataFrame(columns=SHADE_TABLE_COLUMNS)


#: Example: joining a per-5-min shade table against an Octopus CSV
#: (Timestamp,Latitude,Longitude,Temperature,Humidity,PM1.0,PM2.5,PM4.0,
#: PM10.0 — SCL_octopuss/octopus-firmware/src/Octopus_Firmware.cpp:46).
#: ASSUMPTION TO CONFIRM WITH THE OCTOPUS TEAM: GPS-sourced Timestamp rows
#: are UTC (u-blox NMEA/UBX time is always UTC — Octopus_Firmware.cpp
#: getGPSTime(), no offset applied); RTC-fallback rows (no GPS fix) use
#: whatever local time the RTC was set to in the field, which may be
#: America/Sao_Paulo (UTC-3) rather than UTC — the firmware does not
#: record which source produced a given row. Confirm before joining.
OCTOPUS_JOIN_EXAMPLE = """
import pandas as pd

octopus = pd.read_csv("log0.csv", parse_dates=["Timestamp"])
# octopus["Timestamp"] = octopus["Timestamp"].dt.tz_localize("UTC")  # ASSUMPTION — confirm source (GPS vs RTC) first

shade = pd.read_parquet("p05_building_shade.parquet")  # point_id, timestamp, date, sun_altitude_deg, sun_azimuth_deg, shaded
shade["timestamp"] = pd.to_datetime(shade["timestamp"], utc=True)

# nearest OM2 point in space and nearest 5-min shade timestamp in time:
merged = pd.merge_asof(
    octopus.sort_values("Timestamp"),
    shade.sort_values("timestamp"),
    left_on="Timestamp", right_on="timestamp",
    direction="nearest", tolerance=pd.Timedelta("150s"),
)
"""
