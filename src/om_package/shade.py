"""P-05 — building shade per OM2 point per 5-minute interval, on the walk
dates, daylight only, in Rio local time (America/Sao_Paulo, UTC-3, no DST).
Loggers record UTC; every shipped time column carries the local time and a
UTC twin.

Building and terrain shade only.

Method (compute_shade_local, given a marched horizon profile per point):
  1. sun_positions(): pvlib solar position (altitude, azimuth) for every
     5-min local step of each walk date, at the OM2 route's own centroid
     (the route is small enough that one sun position serves every point).
  2. is_shaded(): a point is shaded at time t if the sun's altitude at t
     is at or below the marched horizon angle at the sun's azimuth (the
     nearest-building silhouette in that direction blocks it).
  Only steps with the sun above the geometric horizon are kept, so every row
  is a daylight row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHADE_TABLE_COLUMNS = [
    "point_id",
    "timestamp_local",
    "timestamp_utc",
    "date",
    "sun_altitude_deg",
    "sun_azimuth_deg",
    "shaded",
]


def sun_positions(
    dates: list[str], time_window: tuple[str, str], step_min: int, lat: float, lon: float, tz: str
) -> pd.DataFrame:
    """pvlib solar position at (lat, lon) for every step_min interval in
    time_window (clock time in ``tz``), on each date. Returns columns:
    timestamp (tz-aware), date, sun_altitude_deg, sun_azimuth_deg."""
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


def daylight_rows(shade_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with the sun above the horizon. ``shaded`` is also True at
    night (no direct sun), so every shade SHARE the package reports is
    taken over these rows only; a share over all rows would count night
    as building shade."""
    return shade_df[shade_df["sun_altitude_deg"] > 0]


def daylight_shade_fraction_pct(shade_df: pd.DataFrame) -> float:
    """Percent of daylight (point x timestamp) rows in building shade,
    rounded to 0.1; 0.0 for a table with no daylight rows."""
    day = daylight_rows(shade_df)
    return round(100 * float(day["shaded"].mean()), 1) if len(day) else 0.0


#: WP-04's own citywide default (MAX_DIST_M = 500 m) is unsafe on
#: dtm_extended_300m.tif/buildings_extended_300m.gpkg: that 300m-buffer
#: layer has real nodata starting some distance from OM2 route points —
#: see ``nodata_floor_m()`` below for the measured per-point floor (the
#: buffer's raster bounding box is a rectangle, but valid DTM coverage
#: inside it is not, so some march rays exit real data before reaching
#: 500 m). wp02_horizon.py's running max is not NaN-safe (torch.maximum
#: propagates NaN), so any ray that touches nodata poisons that whole
#: direction's horizon value — this was caught as an all-NaN pilot result
#: before landing v0.1.2's real run (never silently patched into the
#: shared WP-02 engine, which P1's citywide/WP-04 defended numbers also
#: depend on). 100 m is a reasonable near-field radius for pedestrian-
#: height shade in a dense settlement regardless (a 2-5-storey building
#: beyond 100 m casts a horizon-relevant shadow only at very low sun
#: altitudes already handled by is_shaded()'s altitude<=0 branch); the
#: build asserts it stays under the measured floor's minimum every run
#: (see build_om_package.py) rather than trusting a number typed here.
#: Revisit if a wider gap-free extended layer lands.
OM2_SHADE_MAX_DIST_M = 100.0
SHADE_STEP_MIN = 5


def nodata_floor_m(points_gdf, paths) -> dict:
    """Measured per-point distance from each OM2 point to the nearest
    nodata cell of the extended DTM (``paths.dtm_extended_300m``) — the
    empirical floor behind ``OM2_SHADE_MAX_DIST_M``'s scoping decision
    above. Never hardcoded (CLAUDE.md's 'never fabricate a value'): a
    Euclidean distance transform (scipy.ndimage.distance_transform_edt)
    over the raster's own valid/nodata mask gives every pixel's distance
    to the nearest nodata cell in raster units; multiplying by the pixel
    size and sampling at each point's nearest cell gives that point's
    floor. Returns {"min", "median", "max"} in metres across
    ``points_gdf``. The build stores this dict verbatim into
    ``manifest.json``'s ``p05_shade.nodata_floor_m`` and renders it into
    the README/CHANGELOG — see ``require_nodata_floor_m`` in
    ``package_docs.py``, which refuses to render a default when this key
    is missing."""
    import rasterio
    from rasterio.transform import rowcol
    from scipy.ndimage import distance_transform_edt

    with rasterio.open(paths.dtm_extended_300m) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        pixel_m = abs(transform.a)

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= ~np.isclose(arr, nodata, rtol=1e-3)
    dist_m = distance_transform_edt(valid) * pixel_m

    xs = points_gdf.geometry.x.to_numpy()
    ys = points_gdf.geometry.y.to_numpy()
    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.asarray(rows), 0, arr.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, arr.shape[1] - 1)
    per_point = dist_m[rows, cols]
    return {
        "min": float(np.min(per_point)),
        "median": float(np.median(per_point)),
        "max": float(np.max(per_point)),
    }


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


def compute_shade_local(
    point_ids,
    dates: list[str],
    step_min: int,
    lat: float,
    lon: float,
    tz: str,
    horizon_deg: np.ndarray,
    horizon_azimuths_deg: np.ndarray,
    out_path: Path,
    figure_step_min: int = 15,
) -> tuple[dict, pd.DataFrame]:
    """Full P-05 table, one row per (point, local step) on each date, sun
    above the horizon only, streamed date by date to ``out_path`` (parquet:
    ~1.9 k points x ~45 dates x ~150 steps is too large to hold as one
    frame). Requires a horizon profile per point (point_horizon_profiles).

    Returns (summary, subsample): the summary holds row/date counts and the
    share of rows in building shade; the subsample is the table thinned to
    every ``figure_step_min`` minutes, for the figures."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = np.asarray(point_ids)
    horizon = np.asarray(horizon_deg, dtype=float)
    az_grid = np.asarray(horizon_azimuths_deg, dtype=float)
    schema = pa.schema([
        ("point_id", pa.string()),
        ("timestamp_local", pa.timestamp("ns", tz=tz)),
        ("timestamp_utc", pa.timestamp("ns", tz="UTC")),
        ("date", pa.string()),
        ("sun_altitude_deg", pa.float32()),
        ("sun_azimuth_deg", pa.float32()),
        ("shaded", pa.bool_()),
    ])
    n_rows = n_shaded = 0
    thinned = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
        for d in dates:
            sp = sun_positions([d], ("00:00", "23:59"), step_min, lat, lon, tz=tz)
            sp = sp[sp["sun_altitude_deg"] > 0].reset_index(drop=True)
            if sp.empty:
                continue
            alt, az = sp["sun_altitude_deg"].to_numpy(), sp["sun_azimuth_deg"].to_numpy()
            shaded = np.stack([is_shaded(alt, az, h, az_grid) for h in horizon])
            n_pts, n_t = shaded.shape
            local = pd.DatetimeIndex(sp["timestamp"])
            frame = pd.DataFrame({
                "point_id": np.repeat(ids, n_t),
                "timestamp_local": np.tile(local, n_pts),
                "timestamp_utc": np.tile(local.tz_convert("UTC"), n_pts),
                "date": d,
                "sun_altitude_deg": np.tile(alt, n_pts).astype("float32"),
                "sun_azimuth_deg": np.tile(az, n_pts).astype("float32"),
                "shaded": shaded.reshape(-1),
            })
            writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
            n_rows += len(frame)
            n_shaded += int(shaded.sum())
            thinned.append(frame[frame["timestamp_local"].dt.minute % figure_step_min == 0])
    subsample = pd.concat(thinned, ignore_index=True) if thinned else pd.DataFrame(columns=SHADE_TABLE_COLUMNS)
    summary = {
        "n_rows": n_rows, "n_dates": len(dates),
        "shade_fraction_daylight_pct": round(100 * n_shaded / n_rows, 1) if n_rows else 0.0,
        "figure_step_min": figure_step_min,
    }
    return summary, subsample


def build_empty_shade_table() -> pd.DataFrame:
    """Correct schema, zero rows."""
    return pd.DataFrame(columns=SHADE_TABLE_COLUMNS)


def drop_nofix_rows(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> pd.DataFrame:
    """Drop GPS no-fix sentinel rows (Latitude == Longitude == 0.0 — the
    firmware's no-fix sentinel, octopus_outdoor.ino) before any spatial
    join. Must run before any spatial join."""
    no_fix = (df[lat_col].to_numpy() == 0.0) & (df[lon_col].to_numpy() == 0.0)
    return df.loc[~no_fix].reset_index(drop=True)
