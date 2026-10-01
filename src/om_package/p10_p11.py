"""Assemble the P-10 (sun exposure robust to campaign date and device
clock) and P-11 (derived ventilation indices with time-matched wind)
package tables from the sun_envelope / wind_obs / vent_indices modules.

Every ventilation quantity is a geometry-derived PROXY; the wind columns are
SBGL airport observations (10 m), never wind measured at the route. Geometry
enters only through ``horizons`` (the marched horizon table) and the points
frame, so swapping the 2019 epoch for a later one means passing different
inputs, not editing this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sun_envelope import (
    DEFAULT_TZ,
    _tidy,
    annual_sun_hours,
    direct_sun_dose,
    exact_date_agreement,
    sun_envelope,
)
from .vent_indices import compute_indices, prevailing_direction_deg
from .wind_obs import MAX_GAP_MIN, device_to_utc, wind_at

DOSE_HOURS = (1, 2, 3)
DOSE_COLUMNS = [f"dose_{h}h_wh_m2" for h in DOSE_HOURS]
ENVELOPE_STATS = ("min", "median", "max")
DOSE_COLUMN_ORDER = ["point_id", "scope", "local_slot", *DOSE_COLUMNS]

#: package column <- vent_indices column
PREVAILING_COLUMNS = {
    "windward_lambda_f_proxy": "windward_lambda_f_prevailing",
    "canyon_alignment_deg_proxy": "canyon_alignment_prevailing_deg",
    "upwind_shelter_deg_proxy": "upwind_shelter_deg_prevailing",
    "z0_m_proxy": "z0_macdonald_m",
    "zd_m_proxy": "zd_macdonald_m",
    "open_space_fraction_proxy": "open_space_fraction",
}
NEW_POINT_COLUMNS = ["annual_sun_hours", *PREVAILING_COLUMNS.values()]

WIND_COLUMNS = [
    "valid_utc", "valid_local", "drct", "speed_ms", "calm", "variable_direction",
    "used_if_device_clock_utc", "used_if_device_clock_local",
]


def horizon_arrays_and_table(points_gdf, paths, device: str | None = None):
    """One horizon march for every consumer: (horizon_deg (n, 145),
    azimuths_deg (145,), tidy table point_id/azimuth_deg/horizon_deg)."""
    from .shade import point_horizon_profiles

    h, az = point_horizon_profiles(points_gdf, paths, device=device)
    h = np.asarray(h, dtype=float)
    az = np.asarray(az, dtype=float)
    return h, az, _tidy(points_gdf["point_id"].to_numpy(), h, az)


def new_point_columns(points: pd.DataFrame, horizon_deg, azimuths_deg, horizons: pd.DataFrame, *,
                      lat: float, lon: float, root, year: int = 2026) -> tuple[pd.DataFrame, float]:
    """The seven per-point columns added in v0.2.0, plus the prevailing
    bearing they were computed at. Wind direction is the 2015-2024 SBGL
    climatology's circular mean (the same bearing P-06 uses)."""
    wind = prevailing_direction_deg(root)
    idx = compute_indices(points, wind, horizon_deg, azimuths_deg).rename(columns=PREVAILING_COLUMNS)
    sun = annual_sun_hours(horizons, lat=lat, lon=lon, year=year)
    out = idx[["point_id", *PREVAILING_COLUMNS.values()]].merge(sun, on="point_id", how="left")
    return out[["point_id", *NEW_POINT_COLUMNS]], float(wind)


def dose_long(dose: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long dose table: one row per point x local slot x scope, where scope
    is a campaign date (YYYY-MM-DD) or envelope_min / envelope_median /
    envelope_max over the window. Rows for slots where the sun is down on
    every day of the window and all doses are zero are omitted (a point
    that is shaded or dark has dose 0 by construction; the slot grid is the
    same for every scope)."""
    camp = dose["campaign"].rename(columns={"date": "scope"})
    env = dose["envelope"]
    frames = [camp]
    for stat in ENVELOPE_STATS:
        f = env[["point_id", "local_slot"]].copy()
        f["scope"] = f"envelope_{stat}"
        for h in DOSE_HOURS:
            f[f"dose_{h}h_wh_m2"] = env[f"dose_{h}h_{stat}_wh_m2"].to_numpy()
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)[DOSE_COLUMN_ORDER]
    return out


def drop_dark_zero_rows(dose: pd.DataFrame, envelope: pd.DataFrame) -> pd.DataFrame:
    """Drop slots that are night in the envelope AND have zero dose in every
    scope and for every point (same slot set for all rows)."""
    night = set(envelope.loc[envelope["class"] == "night", "local_slot"].unique())
    any_dose = (dose[DOSE_COLUMNS] > 0).any(axis=1)
    live_slots = set(dose.loc[any_dose, "local_slot"].unique())
    drop = night - live_slots
    return dose[~dose["local_slot"].isin(drop)].reset_index(drop=True)


#: Dose rows are on a coarser slot grid than the envelope (the dose is a 1-3 h
#: trailing sum, so 5-min rows add size, not information).
DEFAULT_DOSE_SLOT_MIN = 15
DOSE_DECIMALS = 1


def sun_tables(horizons: pd.DataFrame, campaign_dates: list[str], *, lat: float, lon: float,
               window_start: str, window_end: str, tz: str = DEFAULT_TZ,
               dose_slot_min: int = DEFAULT_DOSE_SLOT_MIN) -> dict:
    """Envelope, dose (long), clock agreement and a summary for the report."""
    env, summary = sun_envelope(horizons, lat=lat, lon=lon, window_start=window_start,
                                window_end=window_end, tz=tz)
    dose = direct_sun_dose(horizons, list(campaign_dates), lat=lat, lon=lon, hours=DOSE_HOURS,
                           window_start=window_start, window_end=window_end, tz=tz, slot_min=dose_slot_min)
    long = drop_dark_zero_rows(dose_long(dose), env)
    long[DOSE_COLUMNS] = long[DOSE_COLUMNS].round(DOSE_DECIMALS)
    agree = exact_date_agreement(horizons, list(campaign_dates), lat=lat, lon=lon, tz=tz)
    agree = agree.rename(columns={"date": "scope"})
    summary = {**summary, "dose_slot_min": int(dose_slot_min), "dose_hours": list(DOSE_HOURS)}
    return {"envelope": env, "dose": long, "clock_agreement": agree, "summary": summary}


def used_obs_by_reading(obs: pd.DataFrame, windows: pd.DataFrame, clock: str, step_min: int = 5) -> dict:
    """{valid_utc Timestamp: campaign date} of the SBGL observations that
    wind_at would return for some 5-min step inside a campaign walk window,
    if the device clock logged ``clock`` ('utc' or 'local')."""
    used: dict = {}
    for _, row in windows.iterrows():
        steps = pd.date_range(row["first_timestamp"], row["last_timestamp"], freq=f"{step_min}min")
        for t in steps:
            w = wind_at(device_to_utc(t, clock), obs=obs)
            if w is not None:
                used.setdefault(w["valid_utc"], str(row["date"]))
    return used


def wind_observed_table(obs: pd.DataFrame, windows: pd.DataFrame | None, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """The SBGL observations of the cached window, flagged with the campaign
    date they would be matched to under each reading of the device clock
    (empty string when not used)."""
    out = pd.DataFrame({
        "valid_utc": obs["valid_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_local": obs["valid_utc"].dt.tz_convert(tz).dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "drct": obs["drct"],
        "speed_ms": obs["speed_ms"],
        "calm": obs["calm"],
        "variable_direction": obs["variable"],
    })
    for clock in ("utc", "local"):
        used = used_obs_by_reading(obs, windows, clock) if windows is not None and len(windows) else {}
        out[f"used_if_device_clock_{clock}"] = obs["valid_utc"].map(used).fillna("").to_numpy()
    return out[WIND_COLUMNS]


def wind_summary(table: pd.DataFrame) -> dict:
    return {
        "n_obs": int(len(table)),
        "n_calm": int(table["calm"].sum()),
        "n_variable_direction": int(table["variable_direction"].sum()),
        "n_used_if_device_clock_utc": int((table["used_if_device_clock_utc"] != "").sum()),
        "n_used_if_device_clock_local": int((table["used_if_device_clock_local"] != "").sum()),
        "max_gap_min": MAX_GAP_MIN,
    }
