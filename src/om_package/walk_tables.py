"""Shipped walk tables for OM2: walks (one row per logger walk) and
walk_points (one row per walk and route point the walk reached).

Times: loggers record UTC; Rio local time is America/Sao_Paulo (UTC-3, no
DST). Shipped time columns are ISO 8601 strings, ``*_local`` with the -03:00
offset and ``*_utc`` with Z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .shade import is_shaded
from .sensor_match import DEFAULT_TAUS_S, sensor_matched
from .sun_envelope import DEFAULT_TZ, _sun
from .walk_dose import walk_dose
from .walks import all_arrivals

#: static point measures that get sensor-matched columns; the regime ones are added by the caller.
STATIC_MEASURES = ["sky_view_factor", "height_width_ratio", "building_height_m", "plan_density_lambda_p"]
PER_WALK_MEASURES = ["shaded_at_arrival", "dose_1h_before_wh_m2"]
WALK_TAG_COLUMNS = {
    "regime": "wind_regime",
    "report_time_local": "wind_report_time_local",
    "report_time_utc": "wind_report_time_utc",
    "direction_deg": "wind_direction_deg",
    "speed_ms": "wind_speed_ms",
    "minutes_from_mid": "wind_report_minutes_from_mid",
}


def iso_local(t, tz: str = DEFAULT_TZ) -> pd.Series:
    t = pd.Series(pd.to_datetime(t, utc=True)).reset_index(drop=True)
    local = t.dt.tz_convert(tz)
    return local.map(lambda x: x.isoformat(timespec="seconds") if pd.notna(x) else None)


def iso_utc(t) -> pd.Series:
    t = pd.Series(pd.to_datetime(t, utc=True)).reset_index(drop=True)
    return t.map(lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(x) else None)


def walks_table(walks: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """walks: walk timing and coverage plus the airport-wind regime tag."""
    w = walks.merge(tags, on="walk_id", how="left")
    out = pd.DataFrame({
        "walk_id": w["walk_id"],
        "date": w["date"].astype(str),
        "period": w["period"],
        "start_local": iso_local(w["start_utc"]),
        "start_utc": iso_utc(w["start_utc"]),
        "end_local": iso_local(w["end_utc"]),
        "end_utc": iso_utc(w["end_utc"]),
        "duration_min": w["duration_min"].round(2),
        "coverage_share": w["coverage_share"],
        "share_on_route": w["share_on_route"],
        "share_interpolated": w["share_interpolated"],
        "max_gap_s": w["max_gap_s"],
        "partial": w["partial"],
        WALK_TAG_COLUMNS["regime"]: w["regime"],
        WALK_TAG_COLUMNS["report_time_local"]: iso_local(w["report_time_utc"]),
        WALK_TAG_COLUMNS["report_time_utc"]: iso_utc(w["report_time_utc"]),
        WALK_TAG_COLUMNS["direction_deg"]: w["direction_deg"],
        WALK_TAG_COLUMNS["speed_ms"]: w["speed_ms"],
        WALK_TAG_COLUMNS["minutes_from_mid"]: w["minutes_from_mid"].round(1),
    })
    return out


def shaded_at_arrival(arrivals: pd.DataFrame, horizon_deg: np.ndarray, azimuths_deg: np.ndarray,
                      point_ids: np.ndarray, *, lat: float, lon: float) -> np.ndarray:
    """Float 0/1 per arrivals row (NaN without an arrival time): is the point
    in building shade (or below the horizon) when the walk reaches it."""
    out = np.full(len(arrivals), np.nan)
    t = pd.DatetimeIndex(arrivals["t_arrival_utc"]).tz_convert("UTC")
    ok = ~t.isna()
    if not ok.any():
        return out
    alt, az = _sun(t[ok], lat, lon)
    row_of = pd.Series(np.arange(len(point_ids)), index=point_ids)
    rows = row_of.reindex(arrivals["point_id"].to_numpy()[ok]).to_numpy()
    idx = np.flatnonzero(ok)
    res = np.full(len(idx), np.nan)
    order = np.argsort(rows, kind="stable")
    bounds = np.flatnonzero(np.diff(rows[order], prepend=-1))
    for lo, hi in zip(bounds, [*bounds[1:], len(order)]):
        sel = order[lo:hi]
        res[sel] = is_shaded(alt[sel], az[sel], horizon_deg[int(rows[sel[0]])], azimuths_deg)
    out[idx] = res
    return out


def walk_points_table(points: pd.DataFrame, fixes: pd.DataFrame, walks: pd.DataFrame,
                      horizons: pd.DataFrame, horizon_deg: np.ndarray, azimuths_deg: np.ndarray, *,
                      regime_measures: list[str], lat: float, lon: float,
                      taus=DEFAULT_TAUS_S) -> pd.DataFrame:
    """walk_points. One row per (walk, point) with an arrival time
    (outside_walk rows are dropped): arrival time, whether the point was
    shaded then, the clear-sky direct dose in the 1 h and 3 h before arrival,
    and for each measure its sensor-matched value at every tau in ``taus``.

    points: point_id, distance_along_m and the measure columns, in the same
    order as ``horizon_deg``."""
    arr = all_arrivals(points[["point_id", "distance_along_m"]], fixes, walks)
    arr["shaded_at_arrival"] = shaded_at_arrival(arr, horizon_deg, azimuths_deg, points["point_id"].to_numpy(),
                                                 lat=lat, lon=lon)
    dose = walk_dose(horizons, arr, lat=lat, lon=lon)
    dose_cols = [c for c in dose.columns if c.startswith("dose_")]
    arr = pd.concat([arr.reset_index(drop=True), dose[dose_cols].reset_index(drop=True)], axis=1)

    measures = [*STATIC_MEASURES, *regime_measures, *PER_WALK_MEASURES]
    static = points[["point_id", *STATIC_MEASURES, *regime_measures]]
    dist = points.set_index("point_id")["distance_along_m"]
    parts = []
    for wid, a in arr.groupby("walk_id", sort=False):
        a = a.reset_index(drop=True)
        vals = static.merge(a[["point_id", *PER_WALK_MEASURES]], on="point_id", how="left")
        sm = sensor_matched(a[["point_id", "walk_id", "t_arrival_utc"]], vals, columns=measures, taus=taus)
        parts.append(pd.concat([a, sm.drop(columns=["point_id", "walk_id"])], axis=1))
    full = pd.concat(parts, ignore_index=True)
    full = full[full["arrival_source"] != "outside_walk"].reset_index(drop=True)

    out = pd.DataFrame({
        "walk_id": full["walk_id"],
        "point_id": full["point_id"],
        "distance_along_m": full["point_id"].map(dist).to_numpy(),
        "t_arrival_local": iso_local(full["t_arrival_utc"]),
        "t_arrival_utc": iso_utc(full["t_arrival_utc"]),
        "arrival_source": full["arrival_source"],
        "shaded_at_arrival": full["shaded_at_arrival"].astype(bool),
        "dose_1h_before_wh_m2": full["dose_1h_before_wh_m2"].round(1),
        "dose_3h_before_wh_m2": full["dose_3h_before_wh_m2"].round(1),
    })
    matched = full[matched_column_names(measures, taus)].round(6)
    for c in matched.columns:
        if c.startswith("shaded_at_arrival_tau"):
            matched[c] = matched[c].clip(0.0, 1.0) + 0.0
    return pd.concat([out, matched], axis=1)


def matched_column_names(measures: list[str], taus=DEFAULT_TAUS_S) -> list[str]:
    return [f"{m}_tau{float(t):g}s" for m in measures for t in taus]
