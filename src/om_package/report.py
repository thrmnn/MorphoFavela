"""The short human report for the Octopus OM2 package (PI, 2026-10-01):
"made for human, simple words, straight to the point ... show the results
and necessary context ... put the key figures in it". The README stays the
technical document; this is what the page's "Download report (PDF)" button
gives.

Every number in the text is read from the built package (manifest.json,
p00_spec_conformance.json, OM2/p07_quality_report.json, OM2/points.parquet,
p05_building_shade, p10_*, p11_wind_observed.csv) and formatted here, never
typed. Where a number is also recorded in manifest.json, the value
recomputed from the table must match it, or the build stops.

The text follows the research-writing style contract (no em dashes, no
bare hedges), checked with style_lint.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.om_package.figures import SEGMENT_LENGTH_M, dose_slots_for_figure
from src.om_package.report_pdf import report_css, render_markdown_pdf
from src.om_package.routes import ROUTE_FLAG_MAX_STREET_DIST_M
from src.om_package.shade import daylight_rows, daylight_shade_fraction_pct
from src.om_package.vent_indices import DEFAULT_BUFFER_M
from src.om_package.wind_obs import CLIM_YEAR_END, CLIM_YEAR_START

PROJECT_FORM = "Brisa+ (MorphoFavela)"
STUDY_TITLE = (
    "Street by street: explaining air temperature differences across streets "
    "and over time in Complexo da Maré"
)
SEGMENT_M = int(SEGMENT_LENGTH_M)
#: k time constants in the segment-length rule L = v k tau.
SEGMENT_K = 3

#: Plain words for what an unfinished spec item waits on.
_WAITS_ON = {
    "OCTOPUS_CSV": "more campaign files",
    "OCTOPUS_TZ": "the device clock",
}
#: Plain words for what the deliberate cut leaves out.
_DESCOPED_PLAIN = {
    "sky_view_factor_terrestrial": "terrestrial LiDAR (laser scans from street level)",
    "airborne_vs_terrestrial_comparison": "terrestrial LiDAR (laser scans from street level)",
    "height_change_2024_2026": "terrestrial LiDAR (laser scans from street level)",
    "tree_shade": "tree shade",
}

#: (file under OM2/, section heading, size class). Order = figure number.
#: The class caps the figure's height in report_css so a section's heading,
#: text and figure share one page.
FIGURES = [
    ("map_form.png", "The route", "hero"),
    ("profiles.png", "Street form", "tall"),
    ("map_shade.png", "Building shade on the campaign dates", "map"),
    ("shade_calendar.png", "Shade by date and time of day", "tall"),
    ("sun_envelope.png", "Does the date matter? Does the clock?", "wide"),
    ("sun_dose.png", "Direct sun dose", "tall"),
    ("map_vent_shelter.png", "Ventilation: shelter from the wind", "map"),
    ("profiles_vent.png", "Ventilation: street direction and roughness", "tall"),
    ("wind_rose_compare.png", "Wind during the campaign", "wide"),
]
_FIG_NO = {name: i + 1 for i, (name, _h, _c) in enumerate(FIGURES)}

_COMPASS_16 = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast",
               "southeast", "south-southeast", "south", "south-southwest", "southwest", "west-southwest",
               "west", "west-northwest", "northwest", "north-northwest"]


def _n(x: float) -> str:
    return f"{x:,.0f}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {d.strftime('%B %Y')}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def count_word(n: int) -> str:
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n] if n < 10 else str(n)


def _is_are(n: int) -> str:
    return "is" if n == 1 else "are"


def _compass(bearing: float) -> str:
    return _COMPASS_16[int(((bearing % 360) + 11.25) // 22.5) % 16]


def _circular_mean_deg(deg) -> float:
    th = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360)


def _fig(name: str) -> str:
    return f"Figure {_FIG_NO[name]}"


def _require(name: str, table_value, manifest_value, *, tol: float = 1e-9) -> None:
    """Stop the build when a value recomputed from a table disagrees with
    the manifest's copy of it."""
    if isinstance(manifest_value, (int, np.integer)) and isinstance(table_value, (int, np.integer)):
        same = int(table_value) == int(manifest_value)
    else:
        same = math.isclose(float(table_value), float(manifest_value), rel_tol=0, abs_tol=tol)
    if not same:
        raise ValueError(f"{name}: table gives {table_value!r}, manifest.json says {manifest_value!r}")


def _load(package_dir: Path) -> dict:
    def js(rel):
        return json.loads((package_dir / rel).read_text(encoding="utf-8"))

    return dict(
        manifest=js("manifest.json"),
        conformance=js("p00_spec_conformance.json"),
        quality=js("OM2/p07_quality_report.json"),
        points=pd.read_parquet(package_dir / "OM2" / "points.parquet"),
        shade=pd.read_parquet(package_dir / "p05_building_shade.parquet",
                              columns=["point_id", "timestamp", "date", "sun_altitude_deg", "shaded"]),
        envelope=pd.read_parquet(package_dir / "p10_sun_envelope.parquet"),
        dose=pd.read_parquet(package_dir / "p10_sun_dose.parquet", columns=["point_id", "scope", "local_slot",
                                                                            "dose_1h_wh_m2"]),
        agreement=pd.read_parquet(package_dir / "p10_clock_agreement.parquet"),
        wind=pd.read_csv(package_dir / "p11_wind_observed.csv", keep_default_na=False,
                         na_values={"drct": [""], "speed_ms": [""]}),
    )


def _segment_means(points: pd.DataFrame, col: str) -> pd.Series:
    """Mean of col per SEGMENT_M stretch, keyed by the stretch's start (m),
    over points NOT flagged as possibly off the walked street."""
    on_street = points[~points["route_geometry_flag"].astype(bool)]
    seg = (on_street["distance_along_m"] // SEGMENT_M * SEGMENT_M).astype(int)
    return on_street.groupby(seg)[col].mean().dropna()


def _spec_facts(f: dict, d: dict) -> None:
    items = d["conformance"]["items"]
    by_status: dict[str, list[dict]] = {}
    for it in items:
        by_status.setdefault(it["status"], []).append(it)
    f["spec_total"] = len(items)
    f["spec_by_status"] = {k: len(v) for k, v in by_status.items()}
    waits = []
    for it in by_status.get("partial", []) + by_status.get("pending", []):
        for tok in it.get("pending_on") or []:
            if tok not in waits:
                waits.append(tok)
    f["waits_on"] = waits
    cut = []
    for item in d["quality"].get("descoped_items", []):
        plain = _DESCOPED_PLAIN.get(item, item.replace("_", " "))
        if plain not in cut:
            cut.append(plain)
    f["descoped_plain"] = cut


def _form_facts(f: dict, pts: pd.DataFrame, quality: dict) -> None:
    svf = pts["sky_view_factor"].dropna()
    f["svf_n"] = int(svf.size)
    f["svf_median"] = float(svf.median())
    f["svf_q25"], f["svf_q75"] = (float(v) for v in svf.quantile([0.25, 0.75]))
    svf_seg = _segment_means(pts, "sky_view_factor")
    f["svf_min_seg"], f["svf_min_val"] = int(svf_seg.idxmin()), float(svf_seg.min())
    f["svf_max_seg"], f["svf_max_val"] = int(svf_seg.idxmax()), float(svf_seg.max())
    f["bh_median"] = float(pts["building_height_m"].median())
    f["sw_median"] = float(pts["street_width_m"].median())
    f["hw_median"] = float(pts["height_width_ratio"].median())
    hw_seg = _segment_means(pts, "height_width_ratio")
    f["hw_max_seg"], f["hw_max_val"] = int(hw_seg.idxmax()), float(hw_seg.max())
    n_flag = quality["route_geometry_flagged_points"]
    _require("route_geometry_flagged_points", int(pts["route_geometry_flag"].sum()), n_flag)
    f["n_flagged"] = n_flag
    f["flag_dist_m"] = ROUTE_FLAG_MAX_STREET_DIST_M


def _shade_facts(f: dict, sh: pd.DataFrame, p05: dict) -> None:
    f["shade_rows"] = len(sh)
    if not len(sh):
        return
    step = sh.sort_values(["point_id", "timestamp"]).groupby("point_id")["timestamp"].diff().dropna()
    f["step_min"] = int(step.median() / pd.Timedelta(minutes=1))
    day = daylight_rows(sh).assign(hour=lambda x: x["timestamp"].dt.hour)
    f["day_share"] = float(day["shaded"].mean())
    _require("p05 shade_fraction_daylight_pct", daylight_shade_fraction_pct(sh), p05["shade_fraction_daylight_pct"])
    per_point = day.groupby("point_id")["shaded"].mean()
    f["pt_q25"], f["pt_q75"] = (float(v) for v in per_point.quantile([0.25, 0.75]))
    hour_sets = [set(g["hour"]) for _, g in day.groupby("date")]
    common = sorted(set.intersection(*hour_sets))
    f["common_hours"] = common
    cc = day[day["hour"].isin(common)]
    per_date = cc.groupby("date")["shaded"].mean()
    f["date_min"], f["date_min_val"] = str(per_date.idxmin()), float(per_date.min())
    f["date_max"], f["date_max_val"] = str(per_date.idxmax()), float(per_date.max())


def _sun_facts(f: dict, d: dict, p10: dict) -> None:
    env = d["envelope"]
    day = env[env["class"] != "night"]
    _require("p10 n_daylight_point_slots", len(day), p10["n_daylight_point_slots"])
    shares = day["class"].value_counts(normalize=True)
    for cls, val in p10["class_share_of_daylight"].items():
        _require(f"p10 class share {cls}", float(shares.get(cls, 0.0)), val)
    _require("p10 date_dependent_share", float(shares.get("date_dependent", 0.0)), p10["date_dependent_share"])
    f["date_dependent"] = float(shares.get("date_dependent", 0.0))
    f["always_shaded"] = float(shares.get("always_shaded", 0.0))
    f["always_sunlit"] = float(shares.get("always_sunlit", 0.0))
    f["window"] = p10["window"]
    f["tz"] = p10["tz"]
    f["utc_offset_h"] = int(pd.Timestamp(p10["window"][0]).tz_localize(p10["tz"]).utcoffset()
                            / pd.Timedelta(hours=1))

    agree = d["agreement"]
    overall = agree.loc[agree["scope"] == "all", "agreement_share"]
    _require("p10 clock_agreement_all", float(overall.iloc[0]), p10["clock_agreement_all"])
    per_date = agree[agree["scope"] != "all"]
    f["clock_agree"] = float(overall.iloc[0])
    f["clock_agree_min"] = float(per_date["agreement_share"].min())
    f["clock_agree_max"] = float(per_date["agreement_share"].max())

    dose = d["dose"]
    slots = dose_slots_for_figure(env, dose)
    f["dose_slots"] = slots
    mid = slots[len(slots) // 2]
    f["dose_slot"] = mid
    at = dose[dose["local_slot"] == mid]
    camp = at[at["scope"].isin(p10["campaign_dates"])].groupby("scope")["dose_1h_wh_m2"].max()
    f["dose_hi_date"], f["dose_hi"] = str(camp.idxmax()), float(camp.max())
    f["dose_lo_date"], f["dose_lo"] = str(camp.idxmin()), float(camp.min())
    med = at[at["scope"] == "envelope_median"]["dose_1h_wh_m2"]
    f["dose_zero_share"] = float((med == 0).mean())
    f["dose_hours"] = p10["dose_hours"]
    f["envelope_slot_min"] = p10["envelope_slot_min"]


def _vent_facts(f: dict, pts: pd.DataFrame, d: dict, p11: dict) -> None:
    shelter = pts["upwind_shelter_deg_prevailing"].dropna()
    f["shelter_median"] = float(shelter.median())
    f["shelter_q25"], f["shelter_q75"] = (float(v) for v in shelter.quantile([0.25, 0.75]))
    align = pts["canyon_alignment_prevailing_deg"].dropna()
    third = 90 / 3
    f["align_third_deg"] = third
    f["align_along"] = float((align < third).mean())
    f["align_across"] = float((align > 90 - third).mean())
    z0 = pts["z0_macdonald_m"].dropna()
    f["z0_median"] = float(z0.median())
    h = pts[f"building_height_mean_buffer_{DEFAULT_BUFFER_M}m"]
    f["z0_over_h_median"] = float((pts["z0_macdonald_m"] / h).dropna().median())
    f["lp_buffer_m"] = DEFAULT_BUFFER_M
    f["lp_median"] = float(pts[f"lambda_p_buffer_{DEFAULT_BUFFER_M}m"].median())

    w = d["wind"]
    _require("p11 n_obs", len(w), p11["n_obs"])
    _require("p11 n_calm", int(w["calm"].sum()), p11["n_calm"])
    _require("p11 n_variable_direction", int(w["variable_direction"].sum()), p11["n_variable_direction"])
    directional = w[~w["calm"] & ~w["variable_direction"] & w["drct"].notna()]
    f["wind_window"] = p11["window_utc"]
    f["wind_station"] = p11["station"]
    f["wind_obs_bearing"] = _circular_mean_deg(directional["drct"])
    f["wind_clim_bearing"] = float(p11["prevailing_wind_bearing_deg"])
    for clock in ("utc", "local"):
        col = f"used_if_device_clock_{clock}"
        _require(f"p11 n_used_if_device_clock_{clock}", int((w[col] != "").sum()), p11[f"n_used_if_device_clock_{clock}"])
        walk = directional[directional[col] != ""]
        f[f"walk_n_{clock}"] = int((w[col] != "").sum())
        f[f"walk_bearing_{clock}"] = _circular_mean_deg(walk["drct"]) if len(walk) else float("nan")


def compute_facts(package_dir: Path) -> dict:
    """Every number the report states, computed from the package files."""
    d = _load(Path(package_dir))
    m, pts = d["manifest"], d["points"]
    om2 = next(r for r in m["routes"] if r["route_id"] == "OM_2")
    _require("OM2 n_points", len(pts), om2["n_points"])
    p05 = m["p05_shade"]
    f: dict = {
        "version": m["package_version"],
        "built": _day(m["built_at_utc"][:10]),
        "n_points": om2["n_points"],
        "length_m": om2["length_m"],
        "communities": list(om2["communities_crossed"]),
        "spacing_m": float(pts.sort_values("distance_along_m")["distance_along_m"].diff().median()),
        "height_m": float(pts["height_m"].median()),
        "campaign_dates": sorted(p05["campaign_dates"]),
        "use_terms": m["use_terms"],
        "geometry_epoch": m["geometry_epoch"],
        "k_tau": SEGMENT_K,
        "k_tau_response": 1 - math.exp(-SEGMENT_K),
    }
    _spec_facts(f, d)
    _form_facts(f, pts, d["quality"])
    _shade_facts(f, d["shade"], p05)
    _sun_facts(f, d, m["p10"])
    _vent_facts(f, pts, d, m["p11"])
    return f


def _spec_sentence(f: dict) -> str:
    s = f["spec_by_status"]
    parts = []
    if s.get("delivered"):
        parts.append(f"{s['delivered']} {_is_are(s['delivered'])} delivered")
    if s.get("delivered (scoped)"):
        n = s["delivered (scoped)"]
        parts.append(f"{n} {_is_are(n)} delivered with a deliberate cut")
    waits = _join([_WAITS_ON.get(t, "input from the team") for t in f["waits_on"]])
    if s.get("partial"):
        n = s["partial"]
        parts.append(f"{n} {_is_are(n)} partly delivered and wait{'s' if n == 1 else ''} on {waits}")
    if s.get("pending"):
        n = s["pending"]
        parts.append(f"{n} wait{'s' if n == 1 else ''} on {waits}")
    if s.get("descoped"):
        n = s["descoped"]
        parts.append(f"{n} {_is_are(n)} left out by decision")
    return f"Of the {f['spec_total']} items the team asked for, {_join(parts)}."


def _figure(name: str, caption: str) -> str:
    size = next(c for n, _h, c in FIGURES if n == name)
    return f"![{_fig(name)}. {caption}](OM2/{name}){{.{size}}}\n"


def _section(name: str, body: list[str], caption: str, *, lead: str = "") -> list[str]:
    heading = next(h for n, h, _c in FIGURES if n == name)
    return ["::: figsec", lead + f"### {heading}\n", *body, _figure(name, caption), ":::\n"]


def render_report_markdown(package_dir: Path) -> str:
    package_dir = Path(package_dir)
    f = compute_facts(package_dir)
    if not f["use_terms"].upper().startswith("INTERNAL REVIEW DRAFT") or "redistribution" not in f["use_terms"]:
        raise ValueError(f"use terms changed; update the report's wording to match: {f['use_terms']!r}")
    dates = f["campaign_dates"]
    utc = f"UTC{f['utc_offset_h']:+d}"
    year = f["geometry_epoch"].split()[0]
    out: list[str] = []

    out.append("# Street form, sun and ventilation along the OM2 route\n")
    out.append(f"::: meta\nOctopus OM2 data package, version {f['version']}, built {f['built']}. "
               "Internal review draft for the named Octopus team. Please do not share it further or cite it.\n:::\n")
    out.append(
        f"This package describes the street along the OM2 walking route in Maré: {_n(f['n_points'])} points, "
        f"one every {f['spacing_m']:g} m over {_n(f['length_m'])} m, through {_join(f['communities'])}. "
        "For each point it gives the street's shape, when it is in the sun or in building shade, "
        "and how open it is to the wind. "
        f"It supports the Octopus team's study \"{STUDY_TITLE}\" (lead Jingxue, PI Simone). "
        f"Théo Hermann built it for the {PROJECT_FORM} research project. "
        "It holds no temperature analysis: that is the Octopus team's work.\n"
    )
    out.append("::: hero")
    out.append(_figure("map_form.png",
                       "The OM2 route over the Maré buildings, coloured by sky view: the share of open sky "
                       "above a point, from 0 (none) to 1 (open). Dark stretches are the enclosed ones. "
                       "Labels give metres from the route start."))
    out.append(":::\n")

    out.append("## What is in it\n")
    out.append(_spec_sentence(f) + " The technical README lists every item and every column.\n")
    if f["descoped_plain"]:
        cut = _join(f["descoped_plain"])
        out.append(f"{cut[0].upper() + cut[1:]} {_is_are(len(f['descoped_plain']))} not in this version, "
                   "by decision.\n")
    out.append("What we need from the team, most useful first:\n")
    out.append(
        f"1. **The device clock.** Did the loggers record UTC or Rio local time ({utc})? "
        "This changes more of the results than anything else (see Figure "
        f"{_FIG_NO['sun_envelope.png']}).\n"
        "2. **The air-temperature sensor's time constant**, as mounted. It sets the right segment length.\n"
        "3. **The raw campaign files with GPS** and the route the team walked.\n"
    )

    # --- results ------------------------------------------------------
    out += _section("profiles.png", [
        f"The street is narrow and enclosed. The median building is {f['bh_median']:.1f} m tall and the median "
        f"street is {f['sw_median']:.1f} m wide, so buildings are {f['hw_median']:.1f} times as tall as the "
        f"street is wide. Among points on the walked street (excluding the {_n(f['n_flagged'])} flagged points), "
        f"the deepest {SEGMENT_M} m stretch starts at {_n(f['hw_max_seg'])} m, where the ratio "
        f"reaches {f['hw_max_val']:.1f}.\n",
        f"Sky view has a median of {f['svf_median']:.2f}; half of the points lie between {f['svf_q25']:.2f} "
        f"and {f['svf_q75']:.2f}. Among points on the walked street (excluding the {_n(f['n_flagged'])} flagged "
        f"points), the most enclosed {SEGMENT_M} m stretch starts at {_n(f['svf_min_seg'])} m "
        f"(mean {f['svf_min_val']:.2f}) and the most open one at {_n(f['svf_max_seg'])} m "
        f"(mean {f['svf_max_val']:.2f}).\n",
    ], f"Street form along the route. Grey: every metre. Blue: {SEGMENT_M} m means. Look for the stretches "
       "where height-to-width rises and sky view drops together: those are the deep canyons.",
       lead="## Results\n\n")

    if f["shade_rows"]:
        out += _section("map_shade.png", [
            f"On the {len(dates)} campaign dates ({_day(dates[0])} to {_day(dates[-1])}), the route is in "
            f"building shade {_pct(f['day_share'])} of daylight time. Half of the points are in shade for "
            f"{_pct(f['pt_q25'])} to {_pct(f['pt_q75'])} of daylight.\n",
        ], "Share of daylight each point spends in building shade, over all campaign dates. "
           "Dark: mostly shaded. Light: mostly in sun.")
        h0, h_end = f["common_hours"][0], f["common_hours"][-1]
        out += _section("shade_calendar.png", [
            "Each panel is one date. Over the daylight hours all dates share "
            f"({h0:02d}:00 to {h_end:02d}:59 on the device clock read as UTC), the shaded share goes from "
            f"{_pct(f['date_min_val'])} on {_day(f['date_min'])} to {_pct(f['date_max_val'])} on "
            f"{_day(f['date_max'])}.\n",
        ], "Building shade by date (one panel each): distance along the route against time of day, "
           "with the device clock read as UTC. Dark: shade. Light: sun. Grey: night. "
           "The red bracket marks the walk.")

    out += _section("sun_envelope.png", [
        f"**The date matters.** Over the season ({_day(f['window'][0])} to {_day(f['window'][1])}), "
        f"{_pct(f['date_dependent'])} of daylight point-slots (one point at one {f['envelope_slot_min']}-minute "
        "time of day) are sunny on some days and shaded on others. "
        f"Only {_pct(f['always_sunlit'])} are always sunny and {_pct(f['always_shaded'])} always shaded. "
        f"The {len(dates)} campaign dates are known from the device files, so use the per-date results.\n",
        f"**The clock matters more.** If the loggers recorded UTC rather than Rio local time ({utc}), "
        f"only {_pct(f['clock_agree'])} of daylight point-slots on the campaign dates keep the same sun or "
        f"shade state ({_pct(f['clock_agree_min'])} to {_pct(f['clock_agree_max'])} by date). "
        "Confirming the clock is the single most useful thing the team can send.\n",
    ], "Left: for each time of day (Rio local time), the share of route points that are always shaded, "
       "date-dependent or always sunny over the season. Right: where along the route the date matters most "
       "(darker = more date-dependent).")

    lo, hi = f["dose_lo_date"], f["dose_hi_date"]
    out += _section("sun_dose.png", [
        f"The dose is the direct sunlight energy a point receives over the past hour on a clear day. "
        f"In the hour up to {f['dose_slot']}, a point in full sun gets up to {_n(f['dose_hi'])} Wh/m² on "
        f"{_day(hi)} and {_n(f['dose_lo'])} Wh/m² on {_day(lo)}. "
        f"On a typical day of the season, {_pct(f['dose_zero_share'])} of points get no direct sun in that "
        f"hour. The package also gives {_join([f'{h}-hour' for h in f['dose_hours'] if h != 1])} doses.\n",
    ], f"Direct sun in the past hour along the route, at {count_word(len(f['dose_slots']))} times of day "
       "(Rio local time). "
       "Coloured lines: the campaign dates. Grey band: lowest to highest over the season. "
       "Drops to zero are building shade.")

    obs_b, clim_b = f["wind_obs_bearing"], f["wind_clim_bearing"]
    out += _section("map_vent_shelter.png", [
        f"The ventilation measures are computed from the building geometry for wind from the "
        f"{_compass(clim_b)} ({clim_b:.0f}°), the long-term prevailing direction at Galeão airport. "
        "None of them is a measured or simulated wind.\n",
        "**Shelter angle** is how high the buildings rise above the horizon when you look into the wind. "
        f"A high angle means the wind is blocked close by. The median is {f['shelter_median']:.0f}°, and half "
        f"of the points lie between {f['shelter_q25']:.0f}° and {f['shelter_q75']:.0f}°.\n",
    ], f"Shelter angle toward the prevailing wind ({clim_b:.0f}°, arrow). Dark: buildings rise steeply "
       "toward the wind, so the point is sheltered. Light: open toward the wind.")

    third = f["align_third_deg"]
    out += _section("profiles_vent.png", [
        "**Canyon alignment** is the angle between the street and the wind: 0° means the wind blows along "
        f"the street, 90° across it. {_pct(f['align_along'])} of points are within {third:.0f}° of along the "
        f"wind and {_pct(f['align_across'])} are within {third:.0f}° of across it, so the route alternates "
        "between the two.\n",
        "**Roughness length** (z0, Macdonald method) describes how much the buildings slow the wind above "
        f"them. Along most of the route it is near zero: median {f['z0_median']:.3f} m, "
        f"{100 * f['z0_over_h_median']:.1f}% of the mean building height. "
        f"Maré is densely built (median plan density {f['lp_median']:.2f} within {f['lp_buffer_m']} m), "
        "beyond the range the method was calibrated on (regular arrays of blocks). Read these values as "
        "outside the method's calibrated range, not as a smooth surface.\n",
    ], f"Ventilation measures along the route for wind from {clim_b:.0f}°. Grey: every metre. Blue: "
       f"{SEGMENT_M} m means. From top: frontal density facing the wind, canyon alignment, shelter angle, "
       "roughness length.")

    walk = []
    for clock, label in (("utc", "UTC"), ("local", "local time")):
        if f[f"walk_n_{clock}"]:
            b = f[f"walk_bearing_{clock}"]
            walk.append(f"{_compass(b)} ({b:.0f}°) if the clock recorded {label}")
    out += _section("wind_rose_compare.png", [
        f"Over the campaign season ({f['wind_window'][0]} to {f['wind_window'][1]}), the observed wind at "
        f"Galeão came on average from the {_compass(obs_b)} ({obs_b:.0f}°), close to the long-term "
        f"{clim_b:.0f}°. "
        + (f"During the walks themselves it came from the {_join(walk)}. " if walk else "")
        + "The airport is a regional reference: wind in the streets is weaker and follows the street.\n",
    ], f"Wind at Galeão airport (10 m): the campaign season (left) against {CLIM_YEAR_START} to {CLIM_YEAR_END} "
       "(right). "
       "Bars point to where the wind comes from; longer bars are more frequent, colour is mean speed.")

    # --- using the data ----------------------------------------------
    out.append("::: keep")
    out.append("## Using the data\n")
    out.append(
        f"**Choose the segment length from the sensor.** The points are {f['spacing_m']:g} m apart, but a sensor "
        "carried at walking speed responds slowly: each reading blends the last stretch walked. Neighbouring "
        f"{f['spacing_m']:g} m points are therefore not independent. A good segment length is L = v × k × τ, with v the walking speed, "
        f"τ the sensor's time constant and k = {f['k_tau']} (about {_pct(f['k_tau_response'])} of a step "
        "change). Match each reading to the segment that ends at that point, not one centred on it. "
        "The segment script in the package re-aggregates the points to any L.\n"
    )
    out.append(
        "**We need τ to set L.** What is the time constant of your air-temperature sensor as mounted "
        "(with its housing), and is it the 63% or the 90% response time?\n"
    )
    out.append(
        "**Robust subset.** Until the clock is confirmed, the points and times that are always sunny or "
        "always shaded give the same answer under any date and either clock.\n"
    )
    out.append(":::\n")

    # --- read with care ------------------------------------------------
    out.append("::: care")
    out.append("## Read with care\n")
    out.append(
        "- **Proxies, not measurements.** Sun, shade and ventilation values come from building geometry. "
        "None is a measured air temperature, sunlight or wind.\n"
        "- **Building shade only.** Trees are not included.\n"
        "- **Clear sky.** The sun dose assumes no cloud, so it is an upper bound.\n"
        f"- **Time.** The season results use Rio local time ({utc}, no daylight saving). The per-date shade "
        "reads the device clock as UTC until the team confirms it.\n"
        f"- **{year} geometry.** Buildings and terrain come from {year} data; 2024 airborne data will "
        "replace them in a later version.\n"
        f"- **Route.** {_n(f['n_flagged'])} of {_n(f['n_points'])} points "
        f"({100 * f['n_flagged'] / f['n_points']:.1f}%) sit inside a building outline or more than "
        f"{f['flag_dist_m']:g} m from a street centre line. The route was traced from a street map, so these "
        "points possibly lie off the walked street.\n"
    )
    out.append(":::\n")

    out.append("## Files and contact\n")
    out.append(
        "The data files sit in the package folder. The technical README gives the full method, the sources, "
        "every column and the spec table. Contact: Théo Hermann.\n"
    )
    return "\n".join(out)


def write_report(package_dir: Path) -> tuple[Path, Path]:
    """Write package_dir/report.md and render package_dir/report.pdf."""
    package_dir = Path(package_dir)
    md = package_dir / "report.md"
    text = render_report_markdown(package_dir)
    md.write_text(text, encoding="utf-8")
    version = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))["package_version"]
    pdf = render_markdown_pdf(md, package_dir / "report.pdf", css=report_css(version),
                              title="Octopus OM2 report", md_format="markdown")
    return md, pdf
