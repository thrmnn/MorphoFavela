"""Assemble the P-10 (sun exposure over the campaign season) and P-11
(ventilation indices at two wind regimes) package tables from the
sun_envelope / wind_regimes / vent_indices modules.

Every ventilation quantity is a geometry-derived PROXY; the wind regimes come
from SBGL airport observations (10 m), never wind measured at the route.
Geometry enters only through ``horizons`` (the marched horizon table) and the
points frame, so swapping the 2019 epoch for a later one means passing
different inputs, not editing this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sun_envelope import (
    DEFAULT_TZ,
    _tidy,
    annual_sun_hours,
    direct_sun_dose,
    sun_envelope,
)
from .vent_indices import compute_indices
from .wind_regimes import circ_dist, hourly_frequency, load_campaign, load_climatology

DOSE_HOURS = (1, 2, 3)
DOSE_COLUMNS = [f"dose_{h}h_wh_m2" for h in DOSE_HOURS]
ENVELOPE_STATS = ("min", "median", "max")
DOSE_COLUMN_ORDER = ["point_id", "scope", "local_slot", *DOSE_COLUMNS]

def regime_slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


#: point-column stem <- vent_indices column; the regime slug is appended.
REGIME_COLUMN_STEMS = {
    "frontal_area_density_windward": "windward_lambda_f_proxy",
    "canyon_alignment_deg": "canyon_alignment_deg_proxy",
    "upwind_shelter_angle_deg": "upwind_shelter_deg_proxy",
    "z0_macdonald_m": "z0_m_proxy",
}
#: the stems that get sensor-matched columns in walk_points (z0 does not)
REGIME_MEASURE_STEMS = ["frontal_area_density_windward", "canyon_alignment_deg", "upwind_shelter_angle_deg"]
REGIME_INDEPENDENT_COLUMNS = {
    "zd_macdonald_m": "zd_m_proxy",
    "open_space_fraction": "open_space_fraction_proxy",
}
STATIC_NEW_COLUMNS = ["annual_sun_hours", *REGIME_INDEPENDENT_COLUMNS]


def regime_column_names(slugs: list[str]) -> list[str]:
    return [f"{stem}_{slug}" for slug in slugs for stem in REGIME_COLUMN_STEMS]


def campaign_regime_list(season: dict) -> list[dict]:
    """The campaign-season regimes as [{key, name, slug, mean_direction_deg}],
    in key order. Names decide the point-column slugs."""
    out = [{"key": g["key"], "name": g["name"], "slug": regime_slug(g["name"]),
            "mean_direction_deg": float(g["mean_direction_deg"])} for g in season["campaign"]["regimes"]]
    if len({g["slug"] for g in out}) != len(out):
        raise ValueError(f"two regimes share a name: {[g['name'] for g in out]}")
    return out


def horizon_arrays_and_table(points_gdf, paths, device: str | None = None):
    """One horizon march for every consumer: (horizon_deg (n, 145),
    azimuths_deg (145,), tidy table point_id/azimuth_deg/horizon_deg)."""
    from .shade import point_horizon_profiles

    h, az = point_horizon_profiles(points_gdf, paths, device=device)
    h = np.asarray(h, dtype=float)
    az = np.asarray(az, dtype=float)
    return h, az, _tidy(points_gdf["point_id"].to_numpy(), h, az)


def new_point_columns(points: pd.DataFrame, horizon_deg, azimuths_deg, horizons: pd.DataFrame, *,
                      regimes: list[dict], lat: float, lon: float, year: int = 2026) -> pd.DataFrame:
    """Per-point columns added in v0.2.0 and reworked in v0.3.0: annual sun
    hours, zd and open-space fraction (wind independent), and for each wind
    regime the windward frontal-area density, canyon alignment, upwind
    shelter angle and z0, evaluated at that regime's mean direction."""
    out = pd.DataFrame({"point_id": points["point_id"].to_numpy()})
    idx0 = None
    for g in regimes:
        idx = compute_indices(points, g["mean_direction_deg"], horizon_deg, azimuths_deg)
        idx0 = idx if idx0 is None else idx0
        for stem, src in REGIME_COLUMN_STEMS.items():
            out[f"{stem}_{g['slug']}"] = idx[src].to_numpy()
    sun = annual_sun_hours(horizons, lat=lat, lon=lon, year=year)
    static = pd.DataFrame({"point_id": idx0["point_id"].to_numpy(),
                           **{c: idx0[src].to_numpy() for c, src in REGIME_INDEPENDENT_COLUMNS.items()}})
    out = out.merge(sun, on="point_id", how="left").merge(static, on="point_id", how="left")
    return out[["point_id", *STATIC_NEW_COLUMNS, *regime_column_names([g["slug"] for g in regimes])]]


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
    """Envelope, dose (long) and a summary for the report."""
    env, summary = sun_envelope(horizons, lat=lat, lon=lon, window_start=window_start,
                                window_end=window_end, tz=tz)
    dose = direct_sun_dose(horizons, list(campaign_dates), lat=lat, lon=lon, hours=DOSE_HOURS,
                           window_start=window_start, window_end=window_end, tz=tz, slot_min=dose_slot_min)
    long = drop_dark_zero_rows(dose_long(dose), env)
    long[DOSE_COLUMNS] = long[DOSE_COLUMNS].round(DOSE_DECIMALS)
    summary = {**summary, "dose_slot_min": int(dose_slot_min), "dose_hours": list(DOSE_HOURS)}
    return {"envelope": env, "dose": long, "summary": summary}


def _mixture_columns(result: dict) -> list[dict]:
    mix = result["mixture"]
    comp = np.asarray(mix["mean_direction_deg"], float)
    rows = []
    for g in result["regimes"]:
        d = circ_dist(comp, g["mean_direction_deg"])
        i = int(np.argmin(d))
        rows.append({
            "mixture_component_direction_deg": float(comp[i]),
            "mixture_difference_deg": float(d[i]),
            "mixture_component_weight": float(mix["weights"][i]),
            "mixture_component_kappa": float(mix["kappa"][i]),
            "mixture_background_weight": float(mix["background_weight"]),
        })
    return rows


def wind_regimes_table(season: dict) -> pd.DataFrame:
    """wind_regimes: one row per period (campaign season, 2015-2024
    climatology) and regime, with the von Mises mixture check beside it."""
    rows = []
    for period in ("campaign", "climatology"):
        res = season[period]
        for g, mx in zip(res["regimes"], _mixture_columns(res)):
            rows.append({
                "period": period, "regime_key": g["key"], "name": g["name"], "column_slug": regime_slug(g["name"]),
                "mean_direction_deg": g["mean_direction_deg"], "share": g["share_of_reports"],
                "mean_speed_ms": g["mean_speed_ms"], "n_reports": g["n_reports"],
                "period_calm_share": res["calm_share"], **mx,
            })
    return pd.DataFrame(rows)


def regime_by_hour_table(season: dict, root) -> pd.DataFrame:
    """wind_regime_by_hour: share of each regime and of calm by Rio local
    hour, per period; each period's own regimes classify its own reports."""
    obs = {"campaign": load_campaign(root), "climatology": load_climatology(root)}
    frames = []
    for period, o in obs.items():
        res = season[period]
        names = {g["key"]: g["name"] for g in res["regimes"]}
        tab = hourly_frequency(o, res)
        long = tab.reset_index().melt(id_vars="local_hour", var_name="regime_key", value_name="share")
        long["regime"] = long["regime_key"].map(names).fillna("calm")
        long.insert(0, "period", period)
        frames.append(long[["period", "local_hour", "regime", "regime_key", "share"]])
    return pd.concat(frames, ignore_index=True).sort_values(["period", "local_hour", "regime_key"]).reset_index(drop=True)
