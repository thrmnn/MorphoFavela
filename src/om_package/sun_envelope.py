"""Sun-exposure envelope for OM2 points: protect time of day, not the date.

The campaign date/clock is uncertain (device clocks may log UTC or Rio
local time; Rio local time is a fixed offset since 2019). PI ruling: stop
chasing the exact date. Each point's marched horizon profile is computed
once (shade.point_horizon_profiles); every date or window then costs only
a sun-position lookup.

Every quantity here is a GEOMETRY-DERIVED PROXY (building horizon vs. sun
position), never measured sunlight, air temperature or airflow. The dose
uses a clear-sky model, so it is an UPPER BOUND (no cloud, no tree shade).
Geometry enters only through the horizon table: swapping the 2019 geometry
for a later epoch means passing a different ``paths`` to ``horizon_profiles``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .shade import OM2_SHADE_MAX_DIST_M, point_horizon_profiles

ENVELOPE_SLOT_MIN = 5

DEFAULT_TZ = "America/Sao_Paulo"
HORIZON_COLUMNS = ["point_id", "azimuth_deg", "horizon_deg"]
ENVELOPE_COLUMNS = ["point_id", "local_slot", "class", "sunlit_day_share", "n_days_sun_up"]
CLASSES = ("always_sunlit", "always_shaded", "date_dependent", "night")
_CHUNK_ELEMS = 20_000_000


def route_centroid_latlon(points_gdf) -> tuple[float, float]:
    """(lat, lon) of the route centroid in WGS84, from the points' own CRS.
    One sun position serves the whole route (it spans ~1.5 km)."""
    pts = points_gdf.to_crs("EPSG:4326")
    return float(pts.geometry.y.mean()), float(pts.geometry.x.mean())


def horizon_profiles(
    points_gdf,
    paths,
    device: str | None = None,
    tmp_dir: Path | None = None,
    max_dist_m: float = OM2_SHADE_MAX_DIST_M,
    persist_to: str | Path | None = None,
) -> pd.DataFrame:
    """Tidy horizon table (point_id, azimuth_deg, horizon_deg), one row per
    point and distinct azimuth. Reuses shade.point_horizon_profiles; the
    145 Tregenza patches repeat the same marched value within an azimuth
    band, so repeats collapse to one row. ``paths`` carries the geometry
    epoch. Persisted as parquet when ``persist_to`` is given."""
    horizon_deg, az = point_horizon_profiles(points_gdf, paths, device=device, tmp_dir=tmp_dir, max_dist_m=max_dist_m)
    table = _tidy(points_gdf["point_id"].to_numpy(), np.asarray(horizon_deg, dtype=float), np.asarray(az, dtype=float))
    if persist_to is not None:
        Path(persist_to).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(persist_to, index=False)
    return table


def _tidy(point_ids, horizon_deg: np.ndarray, azimuths_deg: np.ndarray) -> pd.DataFrame:
    az = np.round(azimuths_deg % 360, 6)
    uniq = np.unique(az)
    cols = [horizon_deg[:, az == a].max(axis=1) for a in uniq]
    wide = np.column_stack(cols)
    return pd.DataFrame(
        {
            "point_id": np.repeat(point_ids, len(uniq)),
            "azimuth_deg": np.tile(uniq, len(point_ids)),
            "horizon_deg": wide.reshape(-1),
        }
    )[HORIZON_COLUMNS]


def _horizon_array(horizons: pd.DataFrame):
    wide = horizons.pivot(index="point_id", columns="azimuth_deg", values="horizon_deg")
    wide = wide.loc[pd.unique(horizons["point_id"])]
    wide = wide.sort_index(axis=1)
    return wide.index.to_numpy(), wide.columns.to_numpy(dtype=float), wide.to_numpy(dtype=float)


def _sun(times: pd.DatetimeIndex, lat: float, lon: float):
    import pvlib

    sp = pvlib.solarposition.get_solarposition(times, lat, lon)
    return sp["apparent_elevation"].to_numpy(), sp["azimuth"].to_numpy()


def _nearest_az_index(sun_az: np.ndarray, az_grid: np.ndarray) -> np.ndarray:
    d = np.abs(sun_az[:, None] - az_grid[None, :]) % 360
    return np.argmin(np.minimum(d, 360 - d), axis=1)


def _sunlit(H: np.ndarray, alt: np.ndarray, az_idx: np.ndarray) -> np.ndarray:
    """(P, T) bool: sun above both the geometric and the marched horizon."""
    return (alt[None, :] > 0) & (alt[None, :] > H[:, az_idx])


def _chunks(n_points: int, n_times: int):
    step = max(1, _CHUNK_ELEMS // max(n_times, 1))
    for s in range(0, n_points, step):
        yield slice(s, min(s + step, n_points))


def _slot_labels(slot_min: int) -> list[str]:
    return [f"{m // 60:02d}:{m % 60:02d}" for m in range(0, 24 * 60, slot_min)]


def _day_slot_grid(start: str, end: str, slot_min: int, tz: str):
    days = pd.date_range(start, end, freq="D")
    n_slots = len(_slot_labels(slot_min))
    naive = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=slot_min * k) for d in days for k in range(n_slots)]
    )
    return len(days), n_slots, naive.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")


def sun_envelope(
    horizons: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    window_start: str = "2025-12-01",
    window_end: str = "2026-04-30",
    slot_min: int = ENVELOPE_SLOT_MIN,
    tz: str = DEFAULT_TZ,
) -> tuple[pd.DataFrame, dict]:
    """Per point and local slot of day, classify over every day in the
    window, counting only days with the sun above the horizon:
    always_sunlit / always_shaded / date_dependent; 'night' when the sun is
    down on every day. Shaded means building horizon (proxy), not measured.

    Returns (table[point_id, local_slot, class, sunlit_day_share,
    n_days_sun_up], summary). ``summary['date_dependent_share']`` is the
    share of daylight point-slots that are date_dependent: how much the
    missing campaign date costs."""
    ids, az_grid, H = _horizon_array(horizons)
    n_days, n_slots, times = _day_slot_grid(window_start, window_end, slot_min, tz)
    ok = ~times.isna()
    alt = np.full(len(times), -90.0)
    az = np.zeros(len(times))
    alt[ok], az[ok] = _sun(times[ok], lat, lon)
    az_idx = _nearest_az_index(az, az_grid)
    up = (alt > 0).reshape(n_days, n_slots)
    n_up = up.sum(axis=0)

    sunlit_cnt = np.zeros((len(ids), n_slots), dtype=np.int32)
    for sl in _chunks(len(ids), len(times)):
        s = _sunlit(H[sl], alt, az_idx).reshape(-1, n_days, n_slots)
        sunlit_cnt[sl] = s.sum(axis=1)

    night = n_up == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(night[None, :], np.nan, sunlit_cnt / np.where(night, 1, n_up)[None, :])
    cls = np.full(sunlit_cnt.shape, "date_dependent", dtype=object)
    cls[sunlit_cnt == 0] = "always_shaded"
    cls[sunlit_cnt == n_up[None, :]] = "always_sunlit"
    cls[:, night] = "night"

    table = pd.DataFrame(
        {
            "point_id": np.repeat(ids, n_slots),
            "local_slot": np.tile(_slot_labels(slot_min), len(ids)),
            "class": cls.reshape(-1),
            "sunlit_day_share": share.reshape(-1),
            "n_days_sun_up": np.tile(n_up, len(ids)),
        }
    )[ENVELOPE_COLUMNS]
    day = table[table["class"] != "night"]
    summary = {
        "window": [window_start, window_end],
        "tz": tz,
        "n_days": int(n_days),
        "n_points": int(len(ids)),
        "n_daylight_point_slots": int(len(day)),
        "date_dependent_share": float((day["class"] == "date_dependent").mean()) if len(day) else float("nan"),
        "class_share_of_daylight": {
            c: (float((day["class"] == c).mean()) if len(day) else float("nan"))
            for c in CLASSES
            if c != "night"
        },
    }
    return table, summary


def _clearsky_dni(times: pd.DatetimeIndex, lat: float, lon: float, altitude_m: float) -> np.ndarray:
    from pvlib.location import Location

    loc = Location(lat, lon, tz=str(times.tz), altitude=altitude_m)
    return loc.get_clearsky(times, model="ineichen")["dni"].to_numpy()


def _beam_horizontal(H, az_grid, times, lat, lon, altitude_m, slot_min):
    """(P, T) clear-sky direct-beam energy on the horizontal plane per slot
    (Wh/m2): DNI x sin(altitude) x slot length; 0 when shaded or sun down."""
    alt, az = _sun(times, lat, lon)
    dni = np.nan_to_num(_clearsky_dni(times, lat, lon, altitude_m))
    beam = dni * np.sin(np.deg2rad(np.clip(alt, 0, None))) * (slot_min / 60.0)
    az_idx = _nearest_az_index(az, az_grid)
    return _sunlit(H, alt, az_idx) * beam[None, :]



def _trailing_sums(x: np.ndarray, n_slots_back: list[int]) -> list[np.ndarray]:
    """x: (..., S) per-slot energy within a day; sum over the trailing
    windows (current slot and the n-1 before it), clipped at 00:00."""
    c = np.cumsum(x, axis=-1)
    out = []
    for n in n_slots_back:
        shifted = np.zeros_like(c)
        shifted[..., n:] = c[..., :-n] if n < c.shape[-1] else 0
        out.append(c - shifted)
    return out


def direct_sun_dose(
    horizons: pd.DataFrame,
    dates: list[str],
    *,
    lat: float,
    lon: float,
    hours: tuple[int, ...] = (1, 2, 3),
    window_start: str = "2025-12-01",
    window_end: str = "2026-04-30",
    slot_min: int = ENVELOPE_SLOT_MIN,
    tz: str = DEFAULT_TZ,
    site_altitude_m: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Clear-sky direct-beam dose on the horizontal plane (pvlib Ineichen
    DNI x sin(sun altitude)), zero when the point is shaded by the building
    horizon or the sun is down, summed over the prior 1/2/3 h (the slot and
    the slots before it, same day). Units Wh/m2. Clear sky and no tree
    shade make it an UPPER BOUND on real direct sunlight; it is a
    geometry-derived proxy, not a measurement.

    Returns {'campaign': point_id, date, local_slot, dose_{h}h_wh_m2...;
             'envelope': point_id, local_slot, dose_{h}h_{min|median|max}_wh_m2}."""
    ids, az_grid, H = _horizon_array(horizons)
    n_slots = len(_slot_labels(slot_min))
    labels = _slot_labels(slot_min)
    back = [int(round(h * 60 / slot_min)) for h in hours]

    camp_days = pd.DatetimeIndex(pd.to_datetime(dates))
    camp_times = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=slot_min * k) for d in camp_days for k in range(n_slots)]
    ).tz_localize(tz)
    n_win, _, win_times = _day_slot_grid(window_start, window_end, slot_min, tz)

    camp_cols = {h: np.zeros((len(ids), len(camp_days), n_slots)) for h in hours}
    stats = {
        (h, k): np.zeros((len(ids), n_slots)) for h in hours for k in ("min", "median", "max")
    }
    # Chunked over points: (P, days, slots) float64 stays within _CHUNK_ELEMS.
    for sl in _chunks(len(ids), max(len(camp_times), len(win_times))):
        e = _beam_horizontal(H[sl], az_grid, camp_times, lat, lon, site_altitude_m, slot_min).reshape(-1, len(camp_days), n_slots)
        for h, s in zip(hours, _trailing_sums(e, back)):
            camp_cols[h][sl] = s
        e = _beam_horizontal(H[sl], az_grid, win_times, lat, lon, site_altitude_m, slot_min).reshape(-1, n_win, n_slots)
        for h, s in zip(hours, _trailing_sums(e, back)):
            stats[(h, "min")][sl] = s.min(axis=1)
            stats[(h, "median")][sl] = np.median(s, axis=1)
            stats[(h, "max")][sl] = s.max(axis=1)

    campaign = pd.DataFrame(
        {
            "point_id": np.repeat(ids, len(camp_days) * n_slots),
            "date": np.tile(np.repeat([d.strftime("%Y-%m-%d") for d in camp_days], n_slots), len(ids)),
            "local_slot": np.tile(labels, len(ids) * len(camp_days)),
            **{f"dose_{h}h_wh_m2": camp_cols[h].reshape(-1) for h in hours},
        }
    )
    envelope = pd.DataFrame(
        {
            "point_id": np.repeat(ids, n_slots),
            "local_slot": np.tile(labels, len(ids)),
            **{f"dose_{h}h_{k}_wh_m2": stats[(h, k)].reshape(-1) for h in hours for k in ("min", "median", "max")},
        }
    )
    return {"campaign": campaign, "envelope": envelope}


def annual_sun_hours(
    horizons: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    year: int = 2026,
    step_min: int = 10,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Hours of geometric direct sun (sun above both the geometric and the
    marched building horizon) per point over the year: a static column like
    sky view factor. Not cloud-aware."""
    ids, az_grid, H = _horizon_array(horizons)
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=f"{step_min}min", inclusive="left").tz_localize(
        tz, ambiguous="NaT", nonexistent="NaT"
    )
    times = times[~times.isna()]
    alt, az = _sun(times, lat, lon)
    az_idx = _nearest_az_index(az, az_grid)
    hours = np.zeros(len(ids))
    for sl in _chunks(len(ids), len(times)):
        hours[sl] = _sunlit(H[sl], alt, az_idx).sum(axis=1) * step_min / 60.0
    return pd.DataFrame({"point_id": ids, "annual_sun_hours": hours})


def clock_readings(timestamps, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """Map device timestamps to local wall-clock time under two readings of
    the unknown device clock. A: the device logged UTC, so local = UTC
    converted to ``tz``. B: the device logged local time already. Offsets
    come from ``tz``, not a typed constant. Returns timestamp, local_A,
    local_B (naive local datetimes) and slot_A / slot_B (HH:MM, 5-min floor)."""
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps))).tz_localize(None)
    a = ts.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    b = ts
    return pd.DataFrame(
        {
            "timestamp": ts,
            "local_A": a,
            "local_B": b,
            "slot_A": a.floor("5min").strftime("%H:%M"),
            "slot_B": b.floor("5min").strftime("%H:%M"),
        }
    )


def exact_date_agreement(
    horizons: pd.DataFrame,
    dates: list[str],
    *,
    lat: float,
    lon: float,
    slot_min: int = 5,
    tz: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """For each campaign date and each logged-clock slot, the sun state
    (sunlit / shaded / night) of every point under reading A (clock = UTC)
    vs reading B (clock = local). Returns per date (and an 'all' row) the
    share of daylight point-slots whose state is identical under A and B;
    daylight = sun up under A or B for that clock slot. Night counts as its
    own state, so a slot that is sunlit under one reading and dark under the
    other is a disagreement."""
    ids, az_grid, H = _horizon_array(horizons)
    n_slots = len(_slot_labels(slot_min))
    rows = []
    for d in dates:
        clock = pd.date_range(d, periods=n_slots, freq=f"{slot_min}min")
        states = []
        for t in (clock.tz_localize("UTC").tz_convert(tz), clock.tz_localize(tz)):
            alt, az = _sun(t, lat, lon)
            s = np.empty((len(ids), n_slots), dtype=np.int8)
            for sl in _chunks(len(ids), n_slots):
                lit = _sunlit(H[sl], alt, _nearest_az_index(az, az_grid))
                s[sl] = np.where(alt[None, :] <= 0, 0, np.where(lit, 2, 1))
            states.append(s)
        a, b = states
        day = (a > 0) | (b > 0)
        rows.append(
            {
                "date": d,
                "n_daylight_point_slots": int(day.sum()),
                "agreement_share": float((a == b)[day].mean()) if day.any() else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    tot = out["n_daylight_point_slots"].sum()
    overall = float((out["agreement_share"] * out["n_daylight_point_slots"]).sum() / tot) if tot else float("nan")
    return pd.concat(
        [out, pd.DataFrame([{"date": "all", "n_daylight_point_slots": int(tot), "agreement_share": overall}])],
        ignore_index=True,
    )
