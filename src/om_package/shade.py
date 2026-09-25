"""P-05 — building shade per OM2 point per 5-minute interval.

Campaign dates are NOT known yet: the team will be asked for the raw OM2
CSVs and dates/walk times will be inferred from that data (PI ruling
2026-09-24; see ``infer_campaign_windows``), not supplied separately. So
v0.1.1 still ships this as a function + CLI that takes an explicit list of
dates and a time window — never a guessed campaign date. Running it for
real needs a DSM built over the OM2 route corridor
(src/brisa_solar/wp02_surface.build_surface) plus a horizon-angle march
(src/brisa_solar/wp02_horizon.patch_visibility(..., return_horizon=True)),
the same engine WP-04 uses for its direct-sun-hours number (wp04_sites.py)
— that is a real, GPU/CPU-costed pipeline step, not "trivially cheap", so
v0.1.1 ships the wiring (this module) and an EMPTY-SCHEMA output (see
``build_empty_shade_table``) rather than a live run against an
unvalidated demo date. Once real campaign dates AND the timezone land,
call ``compute_shade`` with them.

Building-only shade releases once dates are known (PI ruling 2026-09-24).
Tree shade stays PENDING — no DSM/canopy layer for Maré on disk — but is
now a RESERVED column in the shade schema (``tree_shade``), always null,
rather than an absent column, so the table's shape does not change again
once canopy data lands.

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
    # Reserved, explicitly empty (null) column — building-only P-05 releases
    # once campaign dates are known; tree shade needs a canopy/DSM layer
    # this package does not have (PI ruling 2026-09-24). Never inferred or
    # guessed: always null until a real value is computed.
    "tree_shade",
]


#: Timezone is UNRESOLVED (PI ruling 2026-09-24): GPS-fix rows are UTC per
#: firmware (u-blox NMEA/UBX time, Octopus_Firmware.cpp getGPSTime(), no
#: offset applied); RTC-fallback rows (no GPS fix) may be local
#: (America/Sao_Paulo, UTC-3) or something else the firmware does not
#: record. There is deliberately NO default here — every caller must pass
#: an explicit tz, so a silent wrong-timezone assumption cannot ship again.
#: The team will be asked for the raw OM2 CSVs; use infer_campaign_windows()
#: below to read off per-file dates/timestamps once they arrive.


def sun_positions(
    dates: list[str], time_window: tuple[str, str], step_min: int, lat: float, lon: float, tz: str
) -> pd.DataFrame:
    """pvlib solar position at (lat, lon) for every step_min interval in
    time_window (local clock time, tz), on each date. Returns columns:
    timestamp (tz-aware), date, altitude_deg, azimuth_deg.

    ``tz`` has no default — timezone is UNRESOLVED for this campaign (see
    module docstring); the caller must supply it explicitly."""
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


#: WP-04's own citywide default (MAX_DIST_M = 500 m) is unsafe on
#: dtm_extended_300m.tif/buildings_extended_300m.gpkg: that 300m-buffer
#: layer has real nodata starting ~104-330 m from OM2 route points
#: (measured empirically 2026-09-25 — the buffer's raster bounding box is
#: a rectangle, but valid DTM coverage inside it is not, so some march
#: rays exit real data before reaching 500 m). wp02_horizon.py's running
#: max is not NaN-safe (torch.maximum propagates NaN), so any ray that
#: touches nodata poisons that whole direction's horizon value — this was
#: caught as an all-NaN pilot result before landing v0.1.2's real run
#: (never silently patched into the shared WP-02 engine, which P1's
#: citywide/WP-04 defended numbers also depend on). 100 m is safely under
#: the measured 104.15 m worst-case floor across all 1559 OM2 points, and
#: is a reasonable near-field radius for pedestrian-height shade in a
#: dense settlement (a 2-5-storey building beyond 100 m casts a
#: horizon-relevant shadow only at very low sun altitudes already handled
#: by is_shaded()'s altitude<=0 branch). Revisit if a wider gap-free
#: extended layer lands.
OM2_SHADE_MAX_DIST_M = 100.0


def point_horizon_profiles(points_gdf, paths, device: str | None = None, tmp_dir: Path | None = None, max_dist_m: float = OM2_SHADE_MAX_DIST_M):
    """Marched horizon-angle profile per OM2 point — the real engine call
    (wired v0.1.2; v0.1/v0.1.1 shipped only this wiring's
    NotImplementedError). Builds the obstruction surface once
    (buildings_extended_300m.gpkg rasterised onto dtm_extended_300m.tif —
    the same 300m-buffer Maré layer WP-02/WP-04 use, cell_m=1.0 to match
    WP-04's CELL_M) via src/brisa_solar/wp02_surface.build_surface, then
    marches src/brisa_solar/wp02_horizon.patch_visibility(...,
    return_horizon=True) from each OM2 point at pedestrian height (1.5 m,
    WP-04's OBS_HEIGHT_M) over the real 145-patch Tregenza direction set
    (src.svf_v2.compute.generate_tregenza_patches — the same set WP-04
    uses, not a second sky), same engine WP-04 uses for its direct-sun-
    hours number (wp04_sites.py, docstring above).

    horizon_deg is (n_points, 145): multiple Tregenza patches share an
    azimuth band, so several columns repeat the same marched value at
    that azimuth (WP-04's own docstring) — is_shaded()'s nearest-azimuth
    lookup handles that correctly without deduplication.

    Returns (horizon_deg [n_points, 145] float16, azimuths_deg [145]
    float64 — one per Tregenza patch, from patch_azimuth_deg)."""
    import tempfile

    import rasterio

    from src.brisa_solar.wp02_horizon import patch_visibility
    from src.brisa_solar.wp02_surface import build_surface
    from src.brisa_solar.wp04_sites import CELL_M, OBS_HEIGHT_M, patch_azimuth_deg
    from src.svf_v2.compute import generate_tregenza_patches

    with tempfile.TemporaryDirectory() as td:
        out_stem = (Path(tmp_dir) if tmp_dir else Path(td)) / "om2_horizon_surface"
        surface_tif = build_surface(paths.dtm_extended_300m, paths.buildings_extended_300m, CELL_M, out_stem)
        with rasterio.open(surface_tif) as src:
            surface = src.read(1)
            transform = src.transform
        with rasterio.open(f"{out_stem}_is_building.tif") as src:
            is_building = src.read(1).astype(bool)

    directions, _weights = generate_tregenza_patches()
    azimuths_deg = patch_azimuth_deg(directions)

    obs_xy = np.column_stack([points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()])
    _vis, _on_building, horizon_deg = patch_visibility(
        surface, transform, obs_xy, directions=directions, is_building=is_building,
        obs_height_m=OBS_HEIGHT_M, max_dist_m=max_dist_m, march_sampling="nearest",
        device=device, return_horizon=True,
    )
    return horizon_deg, azimuths_deg


def compute_shade(
    points_gdf,
    dates: list[str],
    time_window: tuple[str, str],
    step_min: int,
    lat: float,
    lon: float,
    tz: str,
    horizon_deg: np.ndarray | None = None,
    horizon_azimuths_deg: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full P-05 table: one row per (point, 5-min timestamp). Requires a
    horizon profile per point (see point_horizon_profiles) — pass it in
    once computed; this function does not compute it itself.

    ``tz`` has no default — timezone is UNRESOLVED for this campaign (see
    module docstring); the caller must supply it explicitly. Building-only:
    ``tree_shade`` ships as an explicitly null column (PI ruling
    2026-09-24) — no tree/canopy layer exists for Maré yet."""
    if horizon_deg is None or horizon_azimuths_deg is None:
        raise ValueError(
            "compute_shade needs horizon_deg/horizon_azimuths_deg from "
            "point_horizon_profiles(); v0.1 has not run that (see module docstring)."
        )
    sp = sun_positions(dates, time_window, step_min, lat, lon, tz=tz)
    rows = []
    for pid, h in zip(points_gdf["point_id"], horizon_deg):
        shaded = is_shaded(sp["sun_altitude_deg"].to_numpy(), sp["sun_azimuth_deg"].to_numpy(), h, horizon_azimuths_deg)
        df = sp.copy()
        df["point_id"] = pid
        df["shaded"] = shaded
        df["tree_shade"] = None
        rows.append(df)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SHADE_TABLE_COLUMNS)
    return out[SHADE_TABLE_COLUMNS]


def build_empty_shade_table() -> pd.DataFrame:
    """v0.1's shipped P-05 table: correct schema (including the reserved,
    always-null tree_shade column), zero rows — campaign dates unknown,
    live horizon-march run out of scope for v0.1."""
    return pd.DataFrame(columns=SHADE_TABLE_COLUMNS)


def drop_nofix_rows(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> pd.DataFrame:
    """Drop GPS no-fix sentinel rows (Latitude == Longitude == 0.0 — the
    firmware's no-fix sentinel, octopus_outdoor.ino) before any spatial
    join. Must-fix 5 (panel 2026-09-24): OCTOPUS_JOIN_EXAMPLE below always
    calls this first."""
    no_fix = (df[lat_col].to_numpy() == 0.0) & (df[lon_col].to_numpy() == 0.0)
    return df.loc[~no_fix].reset_index(drop=True)


def infer_campaign_windows(csv_paths: list) -> pd.DataFrame:
    """Per raw Octopus CSV: date, first/last timestamp, and GPS-fix vs
    no-fix row counts — so campaign dates/walk windows can be read off the
    moment the team's raw CSVs arrive (PI ruling 2026-09-24: dates/walk
    times are inferred from the data, not asked for separately). Does NOT
    resolve the timezone of the Timestamp column — see module docstring;
    that stays a required, explicit parameter everywhere else in this
    module.

    Two real CSV schemas exist on the team's Drive (confirmed empirically
    on the Zenodo_release/fixed_data pull, 2026-09-25): the GPS-track
    schema documented in OCTOPUS_JOIN_EXAMPLE (Timestamp, Latitude,
    Longitude, ...), and a Latitude/Longitude-FREE fixed-site
    indoor/outdoor logger schema (Timestamp, Temperature, Humidity,
    PM1.0, PM2.5, PM2.5_cal, PM4.0, PM10.0 — device codes I_1/I_3/I_4/
    O_3/O_4). Files of the second kind report ``n_fix = n_rows``,
    ``n_no_fix = 0`` and ``has_gps = False`` rather than raising —  the
    fix/no-fix distinction does not apply to them, and whether they are
    the OM2 walking-route device under another name or a separate fixed
    sensor deployment is UNVERIFIED (see
    docs/research/octopus_lidar_sources.md §5); never assumed either way.

    Also flags epoch-reset rows (Timestamp == 2000-01-01, the common
    GPS-clock power-on default before a fix is ever acquired) in
    ``n_epoch_reset`` so a clock-not-yet-set campaign start is visible
    rather than silently averaged into first_timestamp.

    Returns one row per input path: csv_path, date (first row's date),
    first_timestamp, last_timestamp, n_rows, n_fix, n_no_fix, has_gps,
    n_epoch_reset.
    """
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p, parse_dates=["Timestamp"])
        has_gps = "Latitude" in df.columns and "Longitude" in df.columns
        if has_gps:
            no_fix = (df["Latitude"] == 0.0) & (df["Longitude"] == 0.0)
            n_fix = int((~no_fix).sum())
            n_no_fix = int(no_fix.sum())
        else:
            n_fix = len(df)
            n_no_fix = 0
        n_epoch_reset = int((df["Timestamp"].dt.date == pd.Timestamp("2000-01-01").date()).sum()) if len(df) else 0
        rows.append(
            {
                "csv_path": str(p),
                "date": df["Timestamp"].iloc[0].date() if len(df) else None,
                "first_timestamp": df["Timestamp"].min() if len(df) else pd.NaT,
                "last_timestamp": df["Timestamp"].max() if len(df) else pd.NaT,
                "n_rows": len(df),
                "n_fix": n_fix,
                "n_no_fix": n_no_fix,
                "has_gps": has_gps,
                "n_epoch_reset": n_epoch_reset,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["csv_path", "date", "first_timestamp", "last_timestamp", "n_rows", "n_fix", "n_no_fix", "has_gps", "n_epoch_reset"],
    )


#: Example: joining a per-5-min shade table against an Octopus CSV
#: (Timestamp,Latitude,Longitude,Temperature,Humidity,PM1.0,PM2.5,PM4.0,
#: PM10.0 — SCL_octopuss/octopus-firmware/src/Octopus_Firmware.cpp:46).
#: TIMEZONE IS UNRESOLVED (see module docstring): GPS-sourced Timestamp
#: rows are UTC (u-blox NMEA/UBX time is always UTC — Octopus_Firmware.cpp
#: getGPSTime(), no offset applied); RTC-fallback rows (no GPS fix) use
#: whatever local time the RTC was set to in the field, which may be
#: America/Sao_Paulo (UTC-3) rather than UTC — the firmware does not
#: record which source produced a given row. There is no default to fall
#: back on; confirm before joining real data. drop_nofix_rows() below
#: always runs FIRST, before any spatial join — Latitude == Longitude ==
#: 0.0 is the no-fix sentinel (octopus_outdoor.ino), not a real position.
OCTOPUS_JOIN_EXAMPLE = """
import pandas as pd
from src.om_package.shade import drop_nofix_rows

octopus = pd.read_csv("log0.csv", parse_dates=["Timestamp"])
octopus = drop_nofix_rows(octopus)  # must run before any spatial join
# octopus["Timestamp"] = octopus["Timestamp"].dt.tz_localize(...)  # UNRESOLVED — confirm source (GPS vs RTC) first, see module docstring

shade = pd.read_parquet("p05_building_shade.parquet")  # point_id, timestamp, date, sun_altitude_deg, sun_azimuth_deg, shaded, tree_shade
shade["timestamp"] = pd.to_datetime(shade["timestamp"], utc=True)

# nearest OM2 point in space and nearest 5-min shade timestamp in time:
merged = pd.merge_asof(
    octopus.sort_values("Timestamp"),
    shade.sort_values("timestamp"),
    left_on="Timestamp", right_on="timestamp",
    direction="nearest", tolerance=pd.Timedelta("150s"),
)
"""
