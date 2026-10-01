"""The short human report for the Octopus OM2 package (PI, 2026-10-01):
plain words, the results and the context needed to read them, the four
key figures. The README stays the technical document; this is what the
page's "Download report (PDF)" button gives.

Every number in the text is read from the built package (manifest.json,
p00_spec_conformance.json, OM2/p07_quality_report.json, OM2/points.parquet,
p05_building_shade.parquet) and formatted here, never typed. A statement
whose number cannot be computed from those files is left out.

The text follows the research-writing style contract (no em dashes, no
bare hedges), checked with style_lint.py.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.om_package.report_pdf import REPORT_CSS, render_markdown_pdf
from src.om_package.routes import ROUTE_FLAG_MAX_STREET_DIST_M
from src.om_package.shade import daylight_rows, daylight_shade_fraction_pct

PROJECT_FORM = "Brisa+ (MorphoFavela)"
STUDY_TITLE = (
    "Street by street: explaining air temperature differences across streets "
    "and over time in Complexo da Maré"
)
SEGMENT_M = 10

#: Plain words for what an unfinished spec item waits on.
_WAITS_ON = {
    "OCTOPUS_CSV": "more campaign files",
    "OCTOPUS_TZ": "the campaign timezone",
}
#: Plain words for what the deliberate cut leaves out.
_DESCOPED_PLAIN = {
    "sky_view_factor_terrestrial": "terrestrial LiDAR (laser scans from street level)",
    "airborne_vs_terrestrial_comparison": "terrestrial LiDAR (laser scans from street level)",
    "height_change_2024_2026": "terrestrial LiDAR (laser scans from street level)",
    "tree_shade": "tree shade",
}
#: What the report asks the team for, keyed by the same pending tokens.
_TEAM_ASK = {
    "OCTOPUS_CSV": "the raw campaign files for the route",
    "OCTOPUS_TZ": "the timezone the device clocks used",
}

#: (file, heading, size class): the class caps the figure's height in
#: REPORT_CSS so each heading, its text and its figure share one page.
FIGURES = [
    ("map_form.png", "How much sky the street sees", "map"),
    ("map_shade.png", "Where the route is in building shade", "map"),
    ("profiles.png", "Street form and shade, metre by metre", "tall"),
    ("shade_calendar.png", "Building shade by date and time of day", "wide"),
]


def _n(x: int) -> str:
    return f"{x:,}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _day(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {d.strftime('%B %Y')}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _is_are(n: int) -> str:
    return "is" if n == 1 else "are"


def _load(package_dir: Path) -> dict:
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    conformance = json.loads((package_dir / "p00_spec_conformance.json").read_text(encoding="utf-8"))
    quality = json.loads((package_dir / "OM2" / "p07_quality_report.json").read_text(encoding="utf-8"))
    points = pd.read_parquet(package_dir / "OM2" / "points.parquet")
    shade = pd.read_parquet(package_dir / "p05_building_shade.parquet",
                            columns=["point_id", "timestamp", "date", "sun_altitude_deg", "shaded"])
    return dict(manifest=manifest, conformance=conformance, quality=quality, points=points, shade=shade)


def _segment_means(points: pd.DataFrame, col: str) -> pd.Series:
    """Mean of col per SEGMENT_M stretch, keyed by the stretch's start (m),
    over points NOT flagged as possibly off the walked street."""
    on_street = points[~points["route_geometry_flag"].astype(bool)]
    seg = (on_street["distance_along_m"] // SEGMENT_M * SEGMENT_M).astype(int)
    return on_street.groupby(seg)[col].mean().dropna()


def compute_facts(package_dir: Path) -> dict:
    """Every number the report states, computed from the package files."""
    d = _load(Path(package_dir))
    m, pts, sh = d["manifest"], d["points"], d["shade"]
    om2 = next(r for r in m["routes"] if r["route_id"] == "OM_2")
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
    }

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

    n_flag = d["quality"]["route_geometry_flagged_points"]
    assert n_flag == int(pts["route_geometry_flag"].sum()), "p07 flag count disagrees with points table"
    f["n_flagged"] = n_flag
    f["flag_dist_m"] = ROUTE_FLAG_MAX_STREET_DIST_M

    f["shade_rows"] = len(sh)
    if len(sh):
        step = sh.sort_values(["point_id", "timestamp"]).groupby("point_id")["timestamp"].diff().dropna()
        f["step_min"] = int(step.median() / pd.Timedelta(minutes=1))
        day = daylight_rows(sh).assign(hour=lambda x: x["timestamp"].dt.hour)
        f["day_share"] = float(day["shaded"].mean())
        if daylight_shade_fraction_pct(sh) != p05["shade_fraction_daylight_pct"]:
            raise ValueError("manifest daylight shade share disagrees with p05_building_shade")
        per_point = day.groupby("point_id")["shaded"].mean()
        f["pt_q25"], f["pt_q75"] = (float(v) for v in per_point.quantile([0.25, 0.75]))
        hour_sets = [set(g["hour"]) for _, g in day.groupby("date")]
        common = sorted(set.intersection(*hour_sets))
        f["common_hours"] = common
        cc = day[day["hour"].isin(common)]
        per_date = cc.groupby("date")["shaded"].mean()
        f["date_min"], f["date_min_val"] = str(per_date.idxmin()), float(per_date.min())
        f["date_max"], f["date_max_val"] = str(per_date.idxmax()), float(per_date.max())
        per_hour = cc.groupby("hour")["shaded"].mean()
        f["hour_min"], f["hour_min_val"] = int(per_hour.idxmin()), float(per_hour.min())
        f["hour_max"], f["hour_max_val"] = int(per_hour.idxmax()), float(per_hour.max())
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


def render_report_markdown(package_dir: Path) -> str:
    package_dir = Path(package_dir)
    f = compute_facts(package_dir)
    if not f["use_terms"].upper().startswith("INTERNAL REVIEW DRAFT") or "redistribution" not in f["use_terms"]:
        raise ValueError(f"use terms changed; update the report's wording to match: {f['use_terms']!r}")

    dates = f["campaign_dates"]
    out: list[str] = []
    out.append("# Octopus OM2: street form and shade along the route\n")
    out.append(f"Version {f['version']}, built {f['built']}. Internal review draft.\n")

    out.append("## What this is\n")
    out.append(
        "This report shows street form and building shade along the OM2 walking route in Maré. "
        f"The data support the Octopus team's study \"{STUDY_TITLE}\" "
        "(Octopus LRP #2, lead Jingxue, PI Simone). "
        f"Théo Hermann produced them for the {PROJECT_FORM} research project. "
        "They describe the street only. The temperature analysis belongs to the Octopus team. "
        "This is an internal review draft for the named Octopus team members. "
        "Please do not share it further or cite it.\n"
    )

    out.append("## What is in it\n")
    out.append(
        f"The package holds {_n(f['n_points'])} points along the {_n(round(f['length_m']))} m route, "
        f"one every {f['spacing_m']:g} m at {f['height_m']:g} m above the ground. "
        f"The route crosses {_join(f['communities'])}. "
        "Each point carries the building height, the street width and their ratio, the sky view, "
        "the built density, the street direction and simple ventilation measures. "
    )
    if f["shade_rows"]:
        out[-1] += (
            f"Building shade is given every {f['step_min']} minutes on {len(dates)} campaign dates, "
            f"from {_day(dates[0])} to {_day(dates[-1])}.\n"
        )
    else:
        out[-1] += "Building shade waits on the campaign dates.\n"
    out.append(_spec_sentence(f) + "\n")
    if f["descoped_plain"]:
        cut = _join(f["descoped_plain"])
        out.append(f"{cut[0].upper() + cut[1:]} {_is_are(len(f['descoped_plain']))} not in this version, by decision.\n")
    if f["waits_on"]:
        asks = _join([_TEAM_ASK.get(t, t) for t in f["waits_on"]])
        out.append(f"From the team we need {asks}.\n")

    out.append("## Results\n")
    (fig1, h1, c1), (fig2, h2, c2), (fig3, h3, c3), (fig4, h4, c4) = FIGURES

    out.append("::: figsec")
    out.append(f"### {h1}\n")
    out.append(
        "Sky view is the share of the sky open above a point, from 0 (no sky) to 1 (open sky). "
        f"Along the route the median is {f['svf_median']:.2f}, and half of the {_n(f['svf_n'])} points "
        f"with a value lie between {f['svf_q25']:.2f} and {f['svf_q75']:.2f}. "
        f"The most enclosed {SEGMENT_M} m stretch starts {_n(f['svf_min_seg'])} m from the route start "
        f"(mean {f['svf_min_val']:.2f}). The most open one starts at {_n(f['svf_max_seg'])} m "
        f"(mean {f['svf_max_val']:.2f}). Both leave out the points flagged as possibly off the walked street.\n"
    )
    out.append(f"![Sky view along the route, from 2019 airborne data. Labels give metres from the route start.](OM2/{fig1}){{.{c1}}}\n")
    out.append(":::\n")

    out.append("::: figsec")
    out.append(f"### {h2}\n")
    if f["shade_rows"]:
        out.append(
            f"The colour gives the share of daylight {f['step_min']}-minute steps in building shade at each point. "
            f"Over all dates, the route is in building shade {_pct(f['day_share'])} of daylight time. "
            f"Half of the points are in shade for {_pct(f['pt_q25'])} to {_pct(f['pt_q75'])} of daylight.\n"
        )
    else:
        out.append("No campaign dates are in this version, so the map has no shade values.\n")
    out.append(f"![Share of daylight in building shade per point, over all campaign dates.](OM2/{fig2}){{.{c2}}}\n")
    out.append(":::\n")

    out.append("::: figsec")
    out.append(f"### {h3}\n")
    out.append(
        f"Thin grey lines show every metre and thick blue lines show {SEGMENT_M} m means. "
        f"The median building height is {f['bh_median']:.1f} m and the median street width is "
        f"{f['sw_median']:.1f} m. The median height-to-width ratio is {f['hw_median']:.1f}. "
        f"The street is deepest in the {SEGMENT_M} m stretch that starts at {_n(f['hw_max_seg'])} m, "
        f"where the buildings are {f['hw_max_val']:.1f} times as tall as the street is wide.\n"
    )
    out.append(f"![Building height, height-to-width ratio, sky view, built density, a ventilation measure and shade along the route.](OM2/{fig3}){{.{c3}}}\n")
    out.append(":::\n")

    out.append("::: figsec")
    out.append(f"### {h4}\n")
    if f["shade_rows"]:
        h0, h_end = f["common_hours"][0], f["common_hours"][-1]
        out.append(
            "Each panel is one campaign date. Dark is building shade, light is sun and grey is night. "
            f"To compare dates on equal terms we keep the daylight hours all {len(dates)} dates share, "
            f"{h0:02d}:00 to {h_end:02d}:59 UTC. In those hours the shaded share goes from "
            f"{_pct(f['date_min_val'])} on {_day(f['date_min'])} to {_pct(f['date_max_val'])} on "
            f"{_day(f['date_max'])}. Over all dates, shade is lowest in the {f['hour_min']:02d}:00 UTC hour "
            f"({_pct(f['hour_min_val'])}) and highest in the {f['hour_max']:02d}:00 UTC hour "
            f"({_pct(f['hour_max_val'])}).\n"
        )
    else:
        out.append("No campaign dates are in this version, so the calendar is empty.\n")
    out.append(f"![One panel per date: distance along the route against time of day (UTC). The red bracket marks the walk window.](OM2/{fig4}){{.{c4}}}\n")
    out.append(":::\n")

    out.append("## Read with care\n")
    out.append("::: care")
    out.append(
        "- Shade comes from buildings only. Trees are not included.\n"
        "- Times are labelled UTC because the campaign clock is not yet confirmed. "
        "If the devices used local time, all times shift by the offset.\n"
        "- The values are proxies for street form. None is a measured air temperature or wind.\n"
        "- Buildings and sky view come from 2019 data, so later changes are missing.\n"
        f"- {_n(f['n_flagged'])} of the {_n(f['n_points'])} points ({100 * f['n_flagged'] / f['n_points']:.1f}%) "
        f"sit inside a building outline or more than {f['flag_dist_m']:g} m from the nearest street centre line. "
        "The route was traced from a street map, so these points possibly lie off the walked street "
        "until the team's own route replaces it.\n"
    )
    out.append(":::\n")

    out.append("## Files and contact\n")
    out.append(
        "The data files sit in the package folder. The technical README "
        "has the full method, the sources and the spec table. Contact: Théo Hermann.\n"
    )
    return "\n".join(out)


def write_report(package_dir: Path) -> tuple[Path, Path]:
    """Write package_dir/report.md and render package_dir/report.pdf."""
    package_dir = Path(package_dir)
    md = package_dir / "report.md"
    md.write_text(render_report_markdown(package_dir), encoding="utf-8")
    pdf = render_markdown_pdf(md, package_dir / "report.pdf", css=REPORT_CSS,
                              title="Octopus OM2 report", md_format="markdown")
    return md, pdf
