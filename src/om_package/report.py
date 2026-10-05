"""The report of the Octopus OM2 data package (report.md / report.pdf).

Every number in the text is computed from the built package files by
``compute_facts`` and formatted here, never typed. Where a value is also
recorded in manifest.json or the quality report, the value recomputed from
the table must match it, or the build stops (``_require``). The README
(package_docs.render_readme) reads the same facts, so the two documents
cannot disagree on a shared number.

Percentages go through ``_Pcts``: two different quantities that round to
the same printed percentage stop the render unless the pair is listed in
``_UNCONFUSABLE`` with wording that keeps them apart (R9).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.om_package.figures import SEGMENT_LENGTH_M
from src.om_package.report_pdf import render_markdown_pdf, report_css
from src.om_package.routes import ROUTE_FLAG_MAX_STREET_DIST_M
from src.om_package.sensor_match import DEFAULT_TAUS_S, TRUNCATION_TAUS
from src.om_package import temp_pairing
from src.om_package.shade import daylight_rows, daylight_shade_fraction_pct
from src.om_package.walks import GAP_FLAG_S, PARTIAL_COVERAGE
from src.om_package.wind_obs import CLIM_YEAR_END, CLIM_YEAR_START
from src.om_package.wind_regimes import N_SECTORS, TAG_MAX_GAP_MIN

PROJECT_FORM = "Brisa+ (MorphoFavela)"
AUTHOR = "Théo Alessandro Hermann"
TITLE = "Street form, sun and wind along the OM2 walking route, Complexo da Maré"
SUBTITLE = "Data package and first pairing with the walk temperature readings"
BYLINE = f"{AUTHOR} · {PROJECT_FORM} · Octopus team"
STUDY_TITLE = (
    "Street by street: explaining air temperature differences across streets "
    "and over time in Complexo da Maré"
)
_SOURCES = {
    "walks": "walks collected by residents of Maré, cleaned and structured by Cassiano and Vincent (Octopus team)",
    "geometry": "buildings and terrain 2019",
    "airport": "Galeão airport hourly weather reports",
    "loggers": "outdoor fixed temperature loggers (Octopus team)",
}
#: Each caption names only the sources its figure draws on.
FIGURE_SOURCES = {
    "fig_route.png": ("walks", "geometry"),
    "fig_form.png": ("geometry",),
    "fig_shade_map.png": ("walks", "geometry"),
    "fig_shade_calendar.png": ("walks", "geometry"),
    "fig_sun_dose.png": ("walks", "geometry"),
    "fig_wind.png": ("airport",),
    "fig_shelter_maps.png": ("geometry", "airport"),
    "fig_vent_profiles.png": ("geometry", "airport"),
    "fig_svf_sensor.png": ("walks", "geometry"),
    "fig_temp_profile.png": ("walks", "loggers"),
    "fig_temp_tau.png": ("walks", "loggers", "geometry"),
}


def source_line(name: str) -> str:
    return "Data: " + "; ".join(_SOURCES[k] for k in FIGURE_SOURCES[name]) + "."
SEGMENT_M = int(SEGMENT_LENGTH_M)
#: The two time constants the sensor figure draws (fig_svf_sensor).
FIGURE_TAUS_S = (10, 30)
#: Dose below this (Wh/m2) is drawn as zero in fig_sun_dose; used when figure_facts.json is absent.
DOSE_ZERO_BELOW = 0.1
#: Canyon alignment within this many degrees of 0 counts as "along the wind", of 90 as "across".
ALIGN_BAND_DEG = 30.0
#: Walks at or above this coverage are "full" for picking the sensor figure's walk (as in figures.py).
FULL_COVERAGE = 0.95

#: Figures in report order. Each prints at 100 % of its designed size
#: (src/om_package/figures.py, vent_figures.py), 200 dpi.
FIGURES = [
    "fig_route.png",
    "fig_form.png",
    "fig_shade_map.png",
    "fig_shade_calendar.png",
    "fig_sun_dose.png",
    "fig_wind.png",
    "fig_shelter_maps.png",
    "fig_vent_profiles.png",
    "fig_svf_sensor.png",
    "fig_temp_profile.png",
    "fig_temp_tau.png",
]
FIGURE_DPI = 200
_FIG_NO = {name: i + 1 for i, name in enumerate(FIGURES)}

#: Pairs of percentage keys allowed to print the same rounded value, because
#: the sentences that carry them name different things (checked by
#: _Pcts.check).
_UNCONFUSABLE: set[frozenset] = {
    # "of daylight time in building shade" (sun section) against "of that
    # hour's airport reports" (wind section): different sections and units.
    frozenset(("shade_pt_q75", "regime2_peak")),
}

#: Shipped file stems in table order. A data file not listed here stops the
#: render, so a new file cannot ship undescribed.
FILE_ORDER = ["OM2/points", "p02b_walks", "p12_walk_points", "p05_building_shade", "p10_sun_dose",
              "p10_sun_envelope", "p10_horizon_profiles", "p11_wind_regimes", "p11_regime_by_hour",
              "p08_data_dictionary", "OM2/p07_quality_report", "OM2/aggregate_to_segments",
              "OM2/join_shade_example", "manifest.json"]


def file_roles(f: dict) -> dict:
    """What each shipped file holds, worded from the facts."""
    hours = _join([str(h) for h in f["p10_dose_hours"]])
    return {
        "OM2/points": f"One row per {f['spacing_m']:g} m point: street form, sun hours, ventilation",
        "p02b_walks": "One row per walk: times, route coverage, wind regime",
        "p12_walk_points": "One row per walk and point: arrival time, shade, sun dose, sensor-matched values",
        "p05_building_shade": f"Building shade per point every {f['shade_step_min']} minutes on each walk date",
        "p10_sun_dose": f"Clear-sky direct sun dose over the past {hours} hours",
        "p10_sun_envelope": "Per point and time of day: sunlit on all, some or no dates of the season",
        "p10_horizon_profiles": "Horizon angle per point and compass direction",
        "p11_wind_regimes": "The two wind regimes, campaign season and long term",
        "p11_regime_by_hour": "Share of each wind regime by hour of day",
        "p08_data_dictionary": "Definition and unit of every column",
        "OM2/p07_quality_report": "Coverage of every point column, flagged points",
        "OM2/aggregate_to_segments": "Script: means over segments of any length",
        "OM2/join_shade_example": "Script: joins logger readings to the shade table",
        "manifest.json": "Version, sources and a checksum for every file",
        "p13_temperature_pairing_readings": "One row per walk temperature reading: logger background, anomaly",
        "p13_temperature_pairing_tau_scan": "Variance explained by sensor-matched measures, per time constant",
        "p13_temperature_pairing_events": "Sharp sun and shade changes along each walk",
        "p13_temperature_pairing_event_response": "Mean temperature change around the sun and shade changes",
        "p13_temperature_pairing_coefficients": "Associations of the anomaly with street measures",
        "p13_temperature_pairing_segment_profile": "Mean anomaly and street measures per 20 m segment",
        "p13_temperature_pairing_warmup": "Readings against minutes since the walk started",
    }


#: Main columns named in the file table (only those present in the file are printed).
FILE_MAIN_COLUMNS = {
    "OM2/points": ["point_id", "distance_along_m", "sky_view_factor"],
    "p02b_walks": ["walk_id", "start_local", "wind_regime"],
    "p12_walk_points": ["walk_id", "point_id", "t_arrival_local"],
    "p05_building_shade": ["point_id", "timestamp_local", "shaded"],
    "p10_sun_dose": ["point_id", "scope", "local_slot"],
    "p10_sun_envelope": ["point_id", "local_slot", "class"],
    "p10_horizon_profiles": ["point_id", "azimuth_deg", "horizon_deg"],
    "p11_wind_regimes": ["name", "mean_direction_deg", "share"],
    "p11_regime_by_hour": ["local_hour", "regime", "share"],
    "p08_data_dictionary": ["id", "definition", "unit"],
}
_NOT_DATA = {"report", "README"}


# --- formatting ------------------------------------------------------------

def _n(x: float) -> str:
    return f"{x:,.0f}"


def _day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {d.strftime('%B %Y')}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def count_word(n: int) -> str:
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n] if n < 10 else str(n)


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _hour(h: int) -> str:
    return f"{int(h):02d}:00"


class _Pcts:
    """Formats fractions as percentages and remembers what each printed
    value stands for, so two different quantities cannot share a printed
    value unnoticed (R9)."""

    def __init__(self):
        self.seen: dict[str, str] = {}

    def __call__(self, key: str, x: float, digits: int = 0) -> str:
        s = f"{100 * x:.{digits}f}%"
        self.seen.setdefault(key, s)
        if self.seen[key] != s:
            raise ValueError(f"percentage key {key!r} printed with two values: {self.seen[key]} and {s}")
        return s

    def collisions(self) -> list[tuple[str, str, str]]:
        by_value: dict[str, list[str]] = {}
        for k, s in self.seen.items():
            by_value.setdefault(s, []).append(k)
        out = []
        for s, keys in by_value.items():
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    if frozenset((a, b)) not in _UNCONFUSABLE:
                        out.append((s, a, b))
        return out

    def check(self) -> None:
        bad = self.collisions()
        if bad:
            raise ValueError("percentages that round alike, reword or list in _UNCONFUSABLE: "
                             + "; ".join(f"{s}: {a} / {b}" for s, a, b in bad))


# --- facts -----------------------------------------------------------------

def _require(name: str, table_value, manifest_value, *, tol: float = 1e-9) -> None:
    """Stop the build when a value recomputed from a table disagrees with
    the manifest's (or quality report's) copy of it."""
    if isinstance(manifest_value, (int, np.integer)) and isinstance(table_value, (int, np.integer)):
        same = int(table_value) == int(manifest_value)
    elif isinstance(manifest_value, (str, dict, list)):
        same = table_value == manifest_value
    else:
        same = math.isclose(float(table_value), float(manifest_value), rel_tol=0, abs_tol=tol)
    if not same:
        raise ValueError(f"{name}: table gives {table_value!r}, manifest.json says {manifest_value!r}")


def _load(package_dir: Path) -> dict:
    def js(rel):
        return json.loads((package_dir / rel).read_text(encoding="utf-8"))

    ff = package_dir / "OM2" / "figure_facts.json"
    return dict(
        manifest=js("manifest.json"),
        quality=js("OM2/p07_quality_report.json"),
        figure_facts=js("OM2/figure_facts.json") if ff.exists() else {},
        points=pd.read_parquet(package_dir / "OM2" / "points.parquet"),
        shade=pd.read_parquet(package_dir / "p05_building_shade.parquet",
                              columns=["point_id", "timestamp_local", "date", "sun_altitude_deg", "shaded"]),
        walks=pd.read_parquet(package_dir / "p02b_walks.parquet"),
        p12=pd.read_parquet(package_dir / "p12_walk_points.parquet"),
        regimes=pd.read_csv(package_dir / "p11_wind_regimes.csv"),
        by_hour=pd.read_csv(package_dir / "p11_regime_by_hour.csv"),
        dictionary=pd.read_parquet(package_dir / "p08_data_dictionary.parquet"),
    )


def _segment_means(points: pd.DataFrame, col: str) -> pd.Series:
    """Mean of col per SEGMENT_M stretch, keyed by the stretch's start (m),
    over points not flagged by route_geometry_flag."""
    on_street = points[~points["route_geometry_flag"].astype(bool)]
    seg = (on_street["distance_along_m"] // SEGMENT_M * SEGMENT_M).astype(int)
    return on_street.groupby(seg)[col].mean().dropna()


def _quartiles(s: pd.Series) -> tuple[float, float, float]:
    s = s.dropna()
    return float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))


def _route_facts(f: dict, d: dict) -> None:
    m, pts = d["manifest"], d["points"]
    om2 = next(r for r in m["routes"] if r["route_id"] == "OM_2")
    _require("OM2 n_points", len(pts), om2["n_points"])
    f["n_points"] = om2["n_points"]
    f["length_m"] = om2["length_m"]
    f["spacing_m"] = float(pts.sort_values("distance_along_m")["distance_along_m"].diff().median())
    f["height_m"] = float(pts["height_m"].median())
    ext = pts.dropna(subset=["neighbourhood"]).groupby("neighbourhood")["distance_along_m"].agg(["min", "max"])
    ext = ext.sort_values("min")
    _require("communities crossed", sorted(ext.index), sorted(om2["communities_crossed"]))
    f["stretches"] = [(name, float(r["min"]), float(r["max"])) for name, r in ext.iterrows()]
    n_flag = d["quality"]["route_geometry_flagged_points"]
    _require("route_geometry_flagged_points", int(pts["route_geometry_flag"].sum()), n_flag)
    _require("manifest route_geometry_flagged_points", n_flag, om2["quality_summary"]["route_geometry_flagged_points"])
    f["n_flagged"] = n_flag
    f["flag_dist_m"] = ROUTE_FLAG_MAX_STREET_DIST_M


def _walk_facts(f: dict, d: dict) -> None:
    w, m = d["walks"], d["manifest"]
    _require("walks n_walks", len(w), m["walks"]["n_walks"])
    _require("p05 n_walks", len(w), m["p05_shade"]["n_walks"])
    dates = sorted(pd.to_datetime(w["date"]).dt.date.astype(str).unique())
    _require("walks n_dates", len(dates), m["walks"]["n_dates"])
    _require("walk dates", dates, sorted(m["p05_shade"]["campaign_dates"]))
    f["n_walks"], f["n_dates"] = len(w), len(dates)
    f["first_date"], f["last_date"] = dates[0], dates[-1]
    start = pd.to_datetime(w["start_local"].str[:19])
    minutes = start.dt.hour * 60 + start.dt.minute
    f["periods"] = {}
    for period, g in w.groupby("period"):
        med = float(minutes[g.index].median())
        f["periods"][period] = {"n": len(g), "start": f"{int(med // 60):02d}:{int(med % 60):02d}"}
    f["duration_median_min"] = float(w["duration_min"].median())
    n_partial = int(w["partial"].sum())
    _require("walks n_partial", n_partial, m["walks"]["n_partial"])
    _require("partial rule", int((w["coverage_share"] < PARTIAL_COVERAGE).sum()), n_partial)
    f["n_partial"] = n_partial
    f["partial_coverage"] = PARTIAL_COVERAGE
    f["gap_flag_s"] = GAP_FLAG_S
    p12 = d["p12"]
    _require("p12 rows", len(p12), m["walks"]["n_walk_point_rows"])
    f["n_walk_point_rows"] = len(p12)
    f["gap_share"] = float((p12["arrival_source"] == "gap_interpolated").mean())


def _form_facts(f: dict, pts: pd.DataFrame) -> None:
    f["bh_median"] = float(pts["building_height_m"].median())
    f["sw_median"] = float(pts["street_width_m"].median())
    f["hw_median_of_ratios"] = float(pts["height_width_ratio"].median())
    f["hw_ratio_of_medians"] = f["bh_median"] / f["sw_median"]
    f["hw_q25"], _, f["hw_q75"] = _quartiles(pts["height_width_ratio"])
    hw_seg = _segment_means(pts, "height_width_ratio")
    f["hw_max_seg"], f["hw_max_val"] = int(hw_seg.idxmax()), float(hw_seg.max())
    f["svf_q25"], f["svf_median"], f["svf_q75"] = _quartiles(pts["sky_view_factor"])
    svf_seg = _segment_means(pts, "sky_view_factor")
    f["svf_min_seg"], f["svf_min_val"] = int(svf_seg.idxmin()), float(svf_seg.min())
    f["svf_max_seg"], f["svf_max_val"] = int(svf_seg.idxmax()), float(svf_seg.max())
    f["lp_q25"], f["lp_median"], f["lp_q75"] = _quartiles(pts["plan_density_lambda_p"])


def _shade_facts(f: dict, d: dict) -> None:
    sh, p05 = d["shade"], d["manifest"]["p05_shade"]
    _require("p05 n_rows", len(sh), p05["n_rows"])
    _require("p05 shade_fraction_daylight_pct", daylight_shade_fraction_pct(sh), p05["shade_fraction_daylight_pct"])
    day = daylight_rows(sh)
    ts = pd.DatetimeIndex(day["timestamp_local"])
    day = day.assign(hour=ts.hour, minute=ts.hour * 60 + ts.minute)
    step = day.sort_values(["point_id", "timestamp_local"]).groupby("point_id")["minute"].diff()
    f["shade_step_min"] = int(step[step > 0].median())
    f["shade_daylight"] = float(day["shaded"].mean())
    per_point = day.groupby("point_id")["shaded"].mean()
    f["shade_pt_q25"], _, f["shade_pt_q75"] = _quartiles(per_point)
    by_hour = day.groupby("hour")["shaded"].mean()
    f["shade_hour_min"], f["shade_hour_min_val"] = int(by_hour.idxmin()), float(by_hour.min())
    f["shade_first_hour"], f["shade_first_val"] = int(by_hour.index[0]), float(by_hour.iloc[0])
    f["shade_last_hour"], f["shade_last_val"] = int(by_hour.index[-1]), float(by_hour.iloc[-1])
    hour_sets = [set(g["hour"]) for _, g in day.groupby("date")]
    common = sorted(set.intersection(*hour_sets))
    f["common_hours"] = (common[0], common[-1])
    per_date = day[day["hour"].isin(common)].groupby("date")["shaded"].mean()
    f["shade_date_min"], f["shade_date_min_val"] = str(per_date.idxmin()), float(per_date.min())
    f["shade_date_max"], f["shade_date_max_val"] = str(per_date.idxmax()), float(per_date.max())
    f["shade_max_dist_m"] = float(p05["max_dist_m"])
    f["nodata_floor_m"] = p05["nodata_floor_m"]


def _dose_facts(f: dict, d: dict) -> None:
    """Sun dose before arrival, in the units the dose figure draws: the
    figure colours 10 m means per walk and draws a mean below the zero
    threshold grey. The share of grey cells is computed the same way here."""
    p12, w = d["p12"], d["walks"]
    fig = d["figure_facts"].get("sun_dose", {})
    zero_below = float(fig.get("zero_drawn_below_wh_m2", DOSE_ZERO_BELOW))
    bin_m = float(fig.get("bin_m", SEGMENT_LENGTH_M))
    p = p12[["walk_id", "distance_along_m", "dose_1h_before_wh_m2", "dose_3h_before_wh_m2"]].merge(
        w[["walk_id", "period"]], on="walk_id")
    cells = p.assign(_b=np.floor(p["distance_along_m"] / bin_m).astype(int)).groupby(["walk_id", "_b"])
    cell = cells[["dose_1h_before_wh_m2", "dose_3h_before_wh_m2"]].mean()
    f["dose_bin_m"] = bin_m
    f["dose_zero_below"] = zero_below
    f["dose_cells_zero_1h"] = float((cell["dose_1h_before_wh_m2"] < zero_below).mean())
    f["dose_cells_zero_3h"] = float((cell["dose_3h_before_wh_m2"] < zero_below).mean())
    f["dose_rows_zero_1h"] = float((p["dose_1h_before_wh_m2"] < zero_below).mean())
    f["dose_period"] = {}
    for period, g in p.groupby("period"):
        f["dose_period"][period] = {
            "median_1h": float(g["dose_1h_before_wh_m2"].median()),
            "zero_1h": float((g["dose_1h_before_wh_m2"] < zero_below).mean()),
        }
    f["dose_max_1h"] = float(p["dose_1h_before_wh_m2"].max())
    f["dose_max_3h"] = float(p["dose_3h_before_wh_m2"].max())
    p10 = d["manifest"]["p10"]
    f["p10_window"] = p10["window"]
    f["p10_dose_hours"] = p10["dose_hours"]
    f["p10_dose_slot_min"] = p10["dose_slot_min"]
    f["p10_envelope_slot_min"] = p10["envelope_slot_min"]


def _wind_facts(f: dict, d: dict) -> None:
    reg, m = d["regimes"], d["manifest"]
    for row in m["p11"]["regimes"]:
        mine = reg[(reg["period"] == row["period"]) & (reg["regime_key"] == row["regime_key"])].iloc[0]
        for k in ("mean_direction_deg", "share", "mean_speed_ms"):
            _require(f"p11 {row['period']} {row['regime_key']} {k}", float(mine[k]), row[k])
    f["wind_window"] = m["p11"]["window_utc"]
    f["clim_years"] = (CLIM_YEAR_START, CLIM_YEAR_END)
    f["n_sectors"] = N_SECTORS
    f["regimes"] = {}
    for period, g in reg.groupby("period"):
        f["regimes"][period] = {r["regime_key"]: {
            "name": r["name"], "slug": r["column_slug"], "dir": float(r["mean_direction_deg"]),
            "share": float(r["share"]), "speed": float(r["mean_speed_ms"]), "n": int(r["n_reports"]),
            "mix_diff": float(r["mixture_difference_deg"]), "mix_dir": float(r["mixture_component_direction_deg"]),
            "calm": float(r["period_calm_share"]),
        } for _, r in g.iterrows()}
    camp = f["regimes"]["campaign"]
    bh = d["by_hour"]
    bh = bh[bh["period"] == "campaign"].pivot(index="local_hour", columns="regime_key", values="share")
    f["regime_peak"] = {k: (int(bh[k].idxmax()), float(bh[k].max())) for k in camp}
    f["regime_major_hours"] = {k: [int(h) for h in bh.index[bh[k] > 0.5]] for k in camp}

    w = d["walks"]
    tags = w["wind_regime"].value_counts().to_dict()
    _require("walks tagged_by_regime", {k: int(v) for k, v in tags.items()}, m["walks"]["tagged_by_regime"])
    f["walk_tags"] = {k: int(v) for k, v in tags.items()}
    f["walk_tags_by_period"] = {p: g["wind_regime"].value_counts().to_dict() for p, g in w.groupby("period")}
    untagged = w[~w["wind_regime"].isin([g["name"] for g in camp.values()])]
    f["untagged"] = [{"regime": r["wind_regime"],
                      "no_direction": bool(pd.isna(r["wind_direction_deg"]) and
                                           abs(r["wind_report_minutes_from_mid"]) <= TAG_MAX_GAP_MIN)}
                     for _, r in untagged.iterrows()]
    f["tag_max_gap_min"] = TAG_MAX_GAP_MIN
    ws = m["provenance"]["wind_source"]
    f["wind_fetched"] = str(ws["fetched_utc"])[:10]
    f["wind_n_obs"] = int(ws["n_obs"])


def _vent_facts(f: dict, pts: pd.DataFrame) -> None:
    f["vent"] = {}
    for key, g in f["regimes"]["campaign"].items():
        slug = g["slug"]
        shelter = pts[f"upwind_shelter_angle_deg_{slug}"]
        align = pts[f"canyon_alignment_deg_{slug}"].dropna()
        f["vent"][key] = {
            "shelter": _quartiles(shelter),
            "align_along": float((align < ALIGN_BAND_DEG).mean()),
            "align_across": float((align > 90 - ALIGN_BAND_DEG).mean()),
            "frontal_median": float(pts[f"frontal_area_density_windward_{slug}"].median()),
        }
    f["align_band_deg"] = ALIGN_BAND_DEG
    z0 = [c for c in pts.columns if c.startswith("z0_macdonald_m_")]
    f["z0_median"] = {c: float(pts[c].median()) for c in z0}
    f["zd_median"] = float(pts["zd_macdonald_m"].median())


def _representative_walk(w: pd.DataFrame) -> pd.Series:
    """Same rule as figures.pick_representative_walk: among walks covering
    at least FULL_COVERAGE of the route, the one closest to the median duration."""
    med = float(w["duration_min"].median())
    full = w[w["coverage_share"] >= FULL_COVERAGE]
    return full.loc[(full["duration_min"] - med).abs().idxmin()]


def _sensor_facts(f: dict, d: dict) -> None:
    p12, w, pts = d["p12"], d["walks"], d["points"]
    for tau in DEFAULT_TAUS_S:
        if f"sky_view_factor_tau{tau}s" not in p12.columns:
            raise ValueError(f"p12_walk_points lacks sky_view_factor_tau{tau}s")
    f["taus"] = list(DEFAULT_TAUS_S)
    f["fig_taus"] = list(FIGURE_TAUS_S)
    f["truncation_taus"] = TRUNCATION_TAUS
    f["ln10"] = math.log(10.0)
    rep = _representative_walk(w)
    fig = d["figure_facts"].get("svf_sensor")
    if fig:
        _require("fig_svf_sensor walk_id", str(rep["walk_id"]), fig["walk_id"])
    one = p12[p12["walk_id"] == rep["walk_id"]].merge(pts[["point_id", "sky_view_factor"]], on="point_id")
    f["rep_walk"] = {"walk_id": str(rep["walk_id"]), "date": str(rep["date"]), "period": str(rep["period"]),
                     "start": str(rep["start_local"])[11:16], "duration_min": float(rep["duration_min"]),
                     "coverage": float(rep["coverage_share"])}
    f["rep_spread"] = {"1m": float(one["sky_view_factor"].std())}
    for tau in FIGURE_TAUS_S:
        f["rep_spread"][tau] = float(one[f"sky_view_factor_tau{tau}s"].std())
    f["tau_cols"] = sorted({re.sub(r"_tau\d+s$", "", c) for c in p12.columns if re.search(r"_tau\d+s$", c)})


def compute_facts(package_dir: Path) -> dict:
    """Every number the report and the README state, computed from the package files."""
    d = _load(Path(package_dir))
    m = d["manifest"]
    f: dict = {"version": m["package_version"], "built_at": m["built_at_utc"], "use_terms": m["use_terms"], "crs": m["crs"],
               "geometry_epoch": m["geometry_epoch"], "decisions": m["provenance"]["decisions"],
               "wind_source": m["provenance"]["wind_source"]}
    _route_facts(f, d)
    _walk_facts(f, d)
    _form_facts(f, d["points"])
    _shade_facts(f, d)
    _dose_facts(f, d)
    _wind_facts(f, d)
    _vent_facts(f, d["points"])
    _sensor_facts(f, d)
    f["dictionary"] = d["dictionary"]
    return f


# --- shared text -------------------------------------------------------------

def opening_paragraph(f: dict) -> str:
    """The package in one paragraph; the README opens with the same text."""
    names = [s[0] for s in f["stretches"]]
    return (
        f"This data package describes the street along the OM2 walking route in Complexo da Maré, Rio de Janeiro: "
        f"{_n(f['n_points'])} points, one every {f['spacing_m']:g} m over {_n(f['length_m'])} m, through "
        f"{_join(names)}. For each point it gives the street form, the building shade on the {f['n_dates']} walk "
        "dates, the direct sun before each walk and ventilation measures for the two wind regimes of the "
        f"season. It supports the Octopus team's study \"{STUDY_TITLE}\" (lead Jingxue, PI Simone). "
        f"{AUTHOR} is part of the Octopus team and built it within the {PROJECT_FORM} research line. The package "
        "also gives a first look at pairing the street measures with the walk temperature readings. All times are Rio local time (UTC-3, no daylight saving); the data "
        "tables also carry the UTC time.\n"
    )


def regime_title(name: str) -> str:
    return f"{name} wind"


def _file_groups(package_dir: Path) -> list[tuple[str, list[str]]]:
    """(stem, [extensions]) for every shipped file except figures and the report."""
    groups: dict[str, list[str]] = {}
    for p in sorted(package_dir.rglob("*")):
        if not p.is_file() or p.name.startswith("_"):
            continue
        rel = p.relative_to(package_dir).as_posix()
        if rel.endswith(".png") or rel in ("OM2/figure_facts.json", "OM2/temp_facts.json"):
            continue
        stem, ext = (rel, "") if rel == "manifest.json" else rel.rsplit(".", 1)
        groups.setdefault(stem, []).append(ext)
    return sorted(groups.items(), key=lambda kv: FILE_ORDER.index(kv[0]) if kv[0] in FILE_ORDER else 99)


def _columns_of(package_dir: Path, stem: str, exts: list[str]) -> list[str]:
    if "parquet" in exts:
        import pyarrow.parquet as pq
        return pq.read_schema(package_dir / f"{stem}.parquet").names
    if "csv" in exts:
        return list(pd.read_csv(package_dir / f"{stem}.csv", nrows=0).columns)
    return []


def file_table(package_dir: Path, f: dict, *, spec_items: dict | None = None) -> str:
    """Markdown table of the shipped files, built from the files present.
    spec_items (stem -> item ids) adds a column, used by the README."""
    roles = file_roles(f)
    rows = []
    for stem, exts in _file_groups(Path(package_dir)):
        if stem in _NOT_DATA:
            continue
        if stem not in roles:
            raise ValueError(f"shipped file {stem} has no description in report.file_roles")
        present = _columns_of(Path(package_dir), stem, exts)
        cols = [c for c in FILE_MAIN_COLUMNS.get(stem, []) if c in present]
        name = stem if not exts[0] else f"{stem}.{exts[0]}" if len(exts) == 1 else \
            f"{stem}.{'/'.join(sorted(exts, key=lambda e: e != 'parquet'))}"
        cells = [f"`{name}`", roles[stem], " ".join(f"`{c}`" for c in cols)]
        if spec_items is not None:
            cells.append(spec_items.get(stem, ""))
        rows.append("| " + " | ".join(cells) + " |")
    head = ["File", "What it holds", "Main columns"] + (["Spec item"] if spec_items is not None else [])
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *rows]) + "\n"


# --- report ------------------------------------------------------------------

def _figure_width_cm(path: Path) -> float:
    """Printed width: the image's own size at FIGURE_DPI."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size[0] / FIGURE_DPI * 2.54


def _figure(package_dir: Path, name: str, caption: str) -> str:
    width = _figure_width_cm(package_dir / "OM2" / name)
    cap = caption + " " + source_line(name)
    return f"![Figure {_FIG_NO[name]}. {cap}](OM2/{name}){{width={width:.2f}cm}}\n"


def _fig(name: str) -> str:
    return f"Figure {_FIG_NO[name]}"


def render_report_markdown(package_dir: Path, *, _pct: _Pcts | None = None) -> str:
    """Sections in the R6 order. Each section opens with its finding, then
    says how to read the figure, then shows it; definitions and the other
    numbers follow the figure."""
    package_dir = Path(package_dir)
    f = compute_facts(package_dir)
    pct = _pct or _Pcts()
    camp = f["regimes"]["campaign"]
    clim = f["regimes"]["climatology"]
    k1, k2 = sorted(camp)
    r1, r2 = camp[k1], camp[k2]
    per = f["periods"]
    y0, y1 = f["clim_years"]
    out: list[str] = []

    # 1 ------------------------------------------------------------------
    built = pd.Timestamp(f["built_at"])
    out.append(
        "::: {.titleblock}\n"
        f"# {TITLE}\n\n"
        f"<p class=\"subtitle\">{SUBTITLE}</p>\n\n"
        f"<p class=\"byline\">{BYLINE}</p>\n\n"
        f"<p class=\"issue\">{built.day} {built.strftime('%B %Y')}<br>Version {f['version']}</p>\n"
        ":::\n"
    )
    out.append(opening_paragraph(f))

    # 2 ------------------------------------------------------------------
    out.append("## What is in the package\n")
    out.append(file_table(package_dir, f))
    out.append(
        "This report presents each measure, shows it along the route and explains how to pair it with "
        "temperature readings. The README holds the full method and every column. The files follow the data "
        "package specification.\n"
    )

    # 3 ------------------------------------------------------------------
    stretches = _join([f"{name} ({_n(a)} to {_n(b)} m)" for name, a, b in f["stretches"]])
    out += [
        "## The route\n",
        f"The route runs {_n(f['length_m'])} m through {stretches} ({_fig('fig_route.png')}). The map shows it "
        "over the 2019 building footprints, labelled in metres from the start.\n",
        _figure(package_dir, "fig_route.png", "The OM2 route over the building footprints of Nova Holanda, Parque Rubens Vaz "
                "and Parque União. Labels give metres from the route start."),
        f"Residents of Maré walked the route {f['n_walks']} times on {f['n_dates']} dates between "
        f"{_day(f['first_date'])} and {_day(f['last_date'])}: {per['morning']['n']} morning walks starting around "
        f"{per['morning']['start']} and {per['evening']['n']} evening walks starting around "
        f"{per['evening']['start']}, each taking about {f['duration_median_min']:.0f} minutes. Every walk goes "
        "from the route start towards its end. Cassiano and Vincent (Octopus team) clean and structure the walk dataset.\n",
    ]

    # 4 ------------------------------------------------------------------
    out += [
        "## Street form\n",
        f"The street is narrow and enclosed: the median of the point height-to-width ratios is "
        f"{f['hw_median_of_ratios']:.1f}, and the median sky view factor is {f['svf_median']:.2f} "
        f"({_fig('fig_form.png')}). "
        f"Read the four profiles from the top: grey lines are the 1 m values, dark lines the {SEGMENT_M} m means. "
        "Where the height-to-width ratio rises and the sky view factor drops together, the street is a deep "
        "canyon.\n",
        _figure(package_dir, "fig_form.png", f"Street form along the route: 1 m values (grey) and {SEGMENT_M} m means (dark). "
                "The band on top names the neighbourhood of each stretch."),
        "Four measures describe the street form, all from 2019 building and terrain geometry. "
        f"**Building height** is the height of the buildings flanking the street (median {f['bh_median']:.1f} m). "
        f"**Height-to-width ratio** is that height divided by the street width, face to face, at the same point; "
        f"half of the points lie between {f['hw_q25']:.1f} and {f['hw_q75']:.1f}. Among points on the walked "
        f"street, the deepest {SEGMENT_M} m stretch starts at {_n(f['hw_max_seg'])} m, with a mean ratio of "
        f"{f['hw_max_val']:.1f}. **Sky view factor** is the share of the sky hemisphere visible "
        f"{f['height_m']:g} m above the street, from 0 (none) to 1 (open sky); half of the points lie between "
        f"{f['svf_q25']:.2f} and {f['svf_q75']:.2f}. Among points on the walked street, the most enclosed {SEGMENT_M} m stretch starts at "
        f"{_n(f['svf_min_seg'])} m (mean {f['svf_min_val']:.2f}) and the most open at {_n(f['svf_max_seg'])} m "
        f"(mean {f['svf_max_val']:.2f}). **Plan area density** is the share of ground covered by buildings in the "
        f"10 m grid cell of the point: median {f['lp_median']:.2f}, half of the points between "
        f"{f['lp_q25']:.2f} and {f['lp_q75']:.2f}.\n",
    ]

    # 5 ------------------------------------------------------------------
    h0, h1 = f["common_hours"]
    out += [
        "## Sun and shade on the walk dates\n",
        f"On the {f['n_dates']} walk dates, the route is in building shade for "
        f"{pct('shade_daylight', f['shade_daylight'])} of daylight time ({_fig('fig_shade_map.png')}). "
        "The map colours each point by the share of daylight time it spends in direct sun: lighter means more "
        "direct sun, and the rest of the time the point is in building shade.\n",
        _figure(package_dir, "fig_shade_map.png", f"Share of daylight time each point spends in direct sun, over the "
                f"{f['n_dates']} walk dates. Lighter = more direct sun; the rest of the time the point is in building shade."),
        "A point is in **building shade** when buildings or terrain block the direct sun. Sun and shade are "
        "computed from 2019 building and terrain geometry, every "
        f"{f['shade_step_min']} minutes of daylight on each walk date. Half of the points spend between "
        f"{pct('shade_pt_q25', f['shade_pt_q25'])} and {pct('shade_pt_q75', f['shade_pt_q75'])} of daylight time "
        "in building shade.\n",
        f"Shade changes more with the time of day than with the date ({_fig('fig_shade_calendar.png')}). Read "
        "the calendar row by row: each row is one walk date, time of day runs left to right in Rio local time, "
        "and the colour gives the share of route points in direct sun (lighter means more direct sun; the rest are in building shade).\n",
        f"In the {_hour(f['shade_hour_min'])} hour only {pct('shade_hour_min', f['shade_hour_min_val'])} of route "
        f"points are shaded, against {pct('shade_last', f['shade_last_val'])} in the "
        f"{_hour(f['shade_last_hour'])} hour. Over the clock hours of daylight that all walk dates share ({_hour(h0)} "
        f"to {h1:02d}:59), the shaded share of route points goes from "
        f"{pct('shade_date_min', f['shade_date_min_val'])} on {_day(f['shade_date_min'])} to "
        f"{pct('shade_date_max', f['shade_date_max_val'])} on {_day(f['shade_date_max'])}.\n",
        _figure(package_dir, "fig_shade_calendar.png", "Share of route points in direct sun by walk date (rows) and time "
                "of day (Rio local time). Lighter = more direct sun; the rest of the route points are in building shade. "
                "White: sun below the horizon."),
    ]

    # 6 ------------------------------------------------------------------
    mo, ev = f["dose_period"]["morning"], f["dose_period"]["evening"]
    out += [
        "## Direct sun before each walk\n",
        "Morning walkers reach streets that have had direct sun in the past hour; evening walkers reach many "
        f"that have had none ({_fig('fig_sun_dose.png')}). "
        "Each row is one walk, labelled by date and start time, mornings above evenings; one colour scale serves both panels, grey is zero and white marks points the walk did not reach.\n",
        _figure(package_dir, "fig_sun_dose.png", "Clear-sky direct sun dose in the hour (left) and the three hours (right) "
                f"before each walk reached each point, in {SEGMENT_M} m means along the route."),
        "The **direct sun dose** is the direct sunlight energy that reached a horizontal surface at the point "
        "in the hour, or the three hours, before the walker arrived, in Wh/m². The arrival time comes from the "
        "walk's own GPS timestamps. The dose comes from 2019 building and terrain geometry and assumes a clear "
        "sky, so it is an upper bound.\n",
        f"In the hour before arrival, the median dose per 1 m walk point is {_n(mo['median_1h'])} Wh/m² on morning walks and "
        f"{_n(ev['median_1h'])} Wh/m² on evening walks. Counted over the {SEGMENT_M} m stretches of each walk, "
        "as the figure draws them, "
        f"{pct('dose_cells_zero_1h', f['dose_cells_zero_1h'])} of walk stretches got no direct sun at all in the "
        f"hour before arrival (grey in the left panel), and {pct('dose_cells_zero_3h', f['dose_cells_zero_3h'])} "
        f"got none in the three hours before (grey in the right panel). Counted over single 1 m points, "
        f"{pct('dose_rows_zero_1h', f['dose_rows_zero_1h'])} of walk points got no direct sun in the hour before "
        "arrival: a stretch with sun on some of its points is not grey.\n",
    ]

    # 7 ------------------------------------------------------------------
    peak1, peak2 = f["regime_peak"][k1], f["regime_peak"][k2]
    tags, tbp, untag = f["walk_tags"], f["walk_tags_by_period"], f["untagged"]
    untag_txt = ""
    if untag:
        why = ("the nearest report gave no wind direction" if all(u["no_direction"] for u in untag)
               else f"no report with a direction lies within {f['tag_max_gap_min']} minutes")
        untag_txt = f" and {count_word(len(untag))} walk{'s' if len(untag) > 1 else ''} with no tag, because {why}"
    out += [
        "## Wind: two regimes\n",
        f"Two winds alternate at Galeão airport: an {r1['name']} wind most of the afternoon and a {r2['name']} "
        f"wind most of the morning ({_fig('fig_wind.png')}). "
        "The roses show how often the wind comes from each direction, coloured by regime, with a line at each "
        "regime's mean direction; the lower panel gives each regime's share of the reports by hour of day.\n",
        f"The regimes come from the Galeão airport hourly weather reports of the campaign season "
        f"({_day(f['wind_window'][0])} to {_day(f['wind_window'][1])}) and of {y0} to {y1}. Each report goes to "
        f"the nearer of the two peaks of the {f['n_sectors']}-sector wind rose. In the campaign season, the "
        f"**{r1['name']}** regime has a mean direction of {r1['dir']:.0f}°, holds "
        f"{pct('regime1_share', r1['share'])} of the reports and has a mean speed of {r1['speed']:.1f} m/s. The "
        f"**{r2['name']}** regime has a mean direction of {r2['dir']:.0f}°, holds "
        f"{pct('regime2_share', r2['share'])} of the reports and has a mean speed of {r2['speed']:.1f} m/s. "
        f"The {r2['name']} regime spreads over a broad northern arc rather than one narrow direction: a check "
        "that fits two circular distributions and a uniform background confirms the "
        f"{r1['name']} direction (within {r1['mix_diff']:.0f}°) but not the {r2['name']} one. Over {y0} to {y1} "
        f"the two regimes point the same way ({clim[k1]['dir']:.0f}° and {clim[k2]['dir']:.0f}°).\n",
        _figure(package_dir, "fig_wind.png", f"Wind at Galeão airport. Top: wind roses for the campaign season (left) and "
                f"{y0} to {y1} (right), coloured by regime. Bottom: share of each regime by hour of day in Rio "
                f"local time (solid: campaign season; dashed: {y0} to {y1})."),
        f"The {r2['name']} wind is most frequent at {_hour(peak2[0])}, with "
        f"{pct('regime2_peak', peak2[1])} of that hour's airport reports; the {r1['name']} wind peaks at "
        f"{_hour(peak1[0])}, with {pct('regime1_peak', peak1[1])}. Each walk carries the regime of the airport "
        f"report nearest its middle time: {tags.get(r1['name'], 0)} walks {r1['name']}, "
        f"{tags.get(r2['name'], 0)} walks {r2['name']}{untag_txt}. {tbp['morning'].get(r2['name'], 0)} of the "
        f"{per['morning']['n']} morning walks had the {r2['name']} wind, and "
        f"{tbp['evening'].get(r1['name'], 0)} of the {per['evening']['n']} evening walks the {r1['name']} wind. "
        "The airport wind is a regional reference, not the wind in the streets.\n",
    ]

    # 8 ------------------------------------------------------------------
    v1, v2 = f["vent"][k1], f["vent"][k2]
    more, less = ((r1, v1), (r2, v2)) if v1["shelter"][1] > v2["shelter"][1] else ((r2, v2), (r1, v1))
    band = f["align_band_deg"]
    out += [
        "## Ventilation for both regimes\n",
        f"Buildings rise higher towards the {more[0]['name']} wind than towards the {less[0]['name']} wind: "
        f"the median upwind shelter angle is {more[1]['shelter'][1]:.0f}° against {less[1]['shelter'][1]:.0f}° "
        f"({_fig('fig_shelter_maps.png')} and {_fig('fig_vent_profiles.png')}). "
        "The maps show the shelter angle for each regime side by side, with an arrow for the wind; the "
        "profiles overlay both regimes in their colours.\n",
        _figure(package_dir, "fig_shelter_maps.png", f"Upwind shelter angle for the {r1['name']} wind (left) and the "
                f"{r2['name']} wind (right). Dark: buildings rise steeply towards the wind."),
        "Three measures describe how open a point is to each wind. They are computed from 2019 building and "
        "terrain geometry at each regime's mean direction; none is a measured or simulated wind. "
        "**Frontal area density** is the building wall area facing the wind per unit of ground area, in the "
        f"10 m grid cell of the point (median {v1['frontal_median']:.2f} for the {r1['name']} wind, "
        f"{v2['frontal_median']:.2f} for the {r2['name']} wind). **Canyon alignment** is the angle between the "
        f"street and the wind: 0° means the wind blows along the street, 90° across it. For the {r1['name']} "
        f"wind, {pct('align1_along', v1['align_along'])} of points lie within {band:.0f}° of along and "
        f"{pct('align1_across', v1['align_across'])} within {band:.0f}° of across. The {r2['name']} wind meets "
        f"most streets at a slant: only {pct('align2_along', v2['align_along'])} of points lie within "
        f"{band:.0f}° of along and {pct('align2_across', v2['align_across'])} within {band:.0f}° of across. "
        "**Upwind shelter angle** is how high buildings and terrain rise above the horizon when you look into "
        f"the wind. Half of the points lie between {v1['shelter'][0]:.0f}° and {v1['shelter'][2]:.0f}° for the "
        f"{r1['name']} wind and between {v2['shelter'][0]:.0f}° and {v2['shelter'][2]:.0f}° for the "
        f"{r2['name']} wind.\n",
        _figure(package_dir, "fig_vent_profiles.png", f"Ventilation measures along the route for the two regimes (colours as "
                f"in {_fig('fig_wind.png')}): frontal area density facing the wind, canyon alignment and upwind "
                f"shelter angle, as {SEGMENT_M} m means."),
    ]

    # 9 ------------------------------------------------------------------
    rep, sp = f["rep_walk"], f["rep_spread"]
    tf = json.loads((package_dir / "OM2" / "temp_facts.json").read_text(encoding="utf-8"))
    t_lo, t_hi = f["fig_taus"]
    taus = _join([f"{t}" for t in f["taus"]])
    flag_share = f["n_flagged"] / f["n_points"]
    out += [
        "## Using the data with temperature readings\n",
        "A sensor carried at walking speed reads the air it has just passed, so the package also gives each "
        f"measure as the sensor would see it ({_fig('fig_svf_sensor.png')}).\n",
        _figure(package_dir, "fig_svf_sensor.png", f"Sky view factor along one full walk ({_day(rep['date'])}, "
                f"{rep['period']}, starting at {rep['start']}): 1 m values (grey) and sensor-matched values for "
                f"τ = {t_lo} s and τ = {t_hi} s (coloured)."),
        "A **sensor-matched** value at a point is a weighted mean of the measure over the points the walk had "
        "already passed. Each passed point gets the weight exp(-Δt/τ), where Δt is the time since the walker "
        f"was there, from the walk's GPS timestamps, and τ is the sensor's time constant. Points more than "
        f"{f['truncation_taus']:g}τ back are left out, and the weights are scaled to sum to one. The columns end "
        f"in `_tau5s`, `_tau10s`, `_tau30s` and `_tau60s`, for τ = {taus} s, one row per walk and point in "
        f"`p12_walk_points`. τ is the time to reach 63% of a step change; if only the 90% response time t90 is "
        f"known, τ = t90 / {f['ln10']:.3f}. The segment script averages these rows per walk, for example "
        "`--by walk_id --tau 30`. Along the walk in the figure, the standard deviation of the sky view factor "
        f"drops from {sp['1m']:.2f} at 1 m to {sp[t_lo]:.2f} for τ = {t_lo} s and {sp[t_hi]:.2f} for "
        f"τ = {t_hi} s: the slower the sensor, the smoother the profile it sees.\n",
        f"**Flagged points.** {_n(f['n_flagged'])} of the {_n(f['n_points'])} points "
        f"({pct('flag_share', flag_share)}) have `route_geometry_flag` set: they fall inside a building outline "
        f"or more than {f['flag_dist_m']:g} m from a street centre line, because some alleys cannot be mapped. "
        "Their street form values describe the nearest mapped street, not the alley walked. Run each analysis "
        "with and without them.\n",
        f"**Arrival times.** In `p12_walk_points`, `arrival_source` says how each arrival time was found: "
        f"`gps` between fixes less than {f['gap_flag_s']} s apart, `gap_interpolated` across a longer gap "
        f"({pct('gap_share', f['gap_share'])} of walk points). Down-weight or drop the interpolated rows. "
        f"In `p02b_walks`, {f['n_partial']} of the {f['n_walks']} walks are marked `partial`: their GPS covers "
        f"less than {pct('partial_rule', f['partial_coverage'])} of the route.\n",
        temp_pairing.team_question(tf),
    ]

    # 9b -----------------------------------------------------------------
    out.append("## Street measures and the walk temperature readings\n")
    captions = {k: v.format(seg=tf["segment_m"]) for k, v in temp_pairing.FIGURE_CAPTIONS.items()}
    for para in temp_pairing.report_paragraphs(tf):
        cited = [n for n in ("fig_temp_profile.png", "fig_temp_tau.png") if "{" + n[:-4] + "}" in para]
        out.append(para.replace("{fig_temp_profile}", _fig("fig_temp_profile.png"))
                   .replace("{fig_temp_tau}", _fig("fig_temp_tau.png")))
        out += [_figure(package_dir, n, captions[n]) for n in cited]

    # 10 -----------------------------------------------------------------
    out.append(f"**Contact.** {AUTHOR}, {PROJECT_FORM}.\n")
    pct.check()
    return "\n".join(out)


def write_report(package_dir: Path) -> tuple[Path, Path]:
    """Write package_dir/report.md and render package_dir/report.pdf."""
    from src.om_package.package_docs import USE_TERMS

    package_dir = Path(package_dir)
    md = package_dir / "report.md"
    md.write_text(render_report_markdown(package_dir), encoding="utf-8")
    version = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))["package_version"]
    pdf = render_markdown_pdf(md, package_dir / "report.pdf", css=report_css(version, USE_TERMS),
                              title=f"Octopus OM2 data package {version}", md_format="markdown")
    return md, pdf
