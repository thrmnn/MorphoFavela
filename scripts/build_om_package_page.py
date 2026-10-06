#!/usr/bin/env python3
"""The dedicated Octopus package page (figure_organization_spec.md §4).

Builds outputs/_packages/mare_om2/index.html — a stable URL across
versions. Reads the newest version directory under mare_om2/ that is not
internal (name not starting with '_') and renders, from files already on
disk (never recomputed, never guessed), in reading order (PI, 2026-10-01:
"clear hierarchy"):

  - header: version + build time; one action row (Download full package
    (ZIP) primary; report PDF, results slides, slide preview and technical
    README secondary);
    use-terms callout with the team-release badge (independent of BRISA
    release_class)
  - what's new in this version, its numbers read from manifest.json
  - spec: status counts, the full conformance table behind a toggle
  - figure gallery: every figures/*.png with a caption, click to enlarge
  - files: data files and documents
  - for the PI and the technical reader: the open release decision (linked
    to brisaverse's /ops), what the team owes, quality, campaign windows,
    data dictionary, version history, glossary, /paper/x1

Hooked into scripts/build_om_package.py — every OM2 build regenerates
this page from whatever that build just wrote.

Run:
    python scripts/build_om_package_page.py --root /home/theo/SCL/SCR/MorphoFavela
    python scripts/build_om_package_page.py --check   # validate, never write
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import hubkit  # noqa: E402
from src.om_package import layout  # noqa: E402
from src.om_package.figures import SEGMENT_LENGTH_M as SEGMENT_M  # noqa: E402
from src.om_package.routes import ROUTE_FLAG_MAX_STREET_DIST_M  # noqa: E402
from src.om_package.spec import internal_dir_for  # noqa: E402

DEFAULT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

BRISA_HUB = "https://brisa.theoalessandro.com"
OPS_DECISION_ID = "om_release_v1_0_0"
OPS_LINK = f"{BRISA_HUB}/ops#dec-{OPS_DECISION_ID}"
PAPER_LINK = f"{BRISA_HUB}/paper/x1"
# The results deck lives on the hub origin that serves this page (under
# /morphofavela-dash/), regenerated from this package by brisaverse
# (slides/gen_om_pk_spec.py). Root-absolute on purpose so it always reaches
# the current deck; check() cannot resolve these in this checkout, so they
# are its one declared exception, exact paths only.
DECK_PDF = "/decks/brisa_om_pk.pdf"
DECK_PREVIEW = "/decks/preview_om_pk.html"
HUB_ORIGIN_LINKS = frozenset({DECK_PDF, DECK_PREVIEW})
PANEL_DOC_REL = "docs/critic/octopus_package_panel_2026-09-24.md"  # repo-root relative
# Rendered inside outputs/_packages/mare_om2/ (stable across versions, like
# index.html) so the "Panel ruling" link never points outside outputs/ — the
# live VPS hub only ever rsyncs outputs/ subtrees (cockpit_bridge_tick.sh),
# never docs/, so a direct docs/ link 404s on the real deployment.
PANEL_PAGE_NAME = "panel_review.html"

NAMED_TEAM = ["Jingxue", "Vincent", "Simone"]  # PI ruling 2026-09-24, Q6a — must match the /ops decision card

# Independent of BRISA release_class (this package belongs to Octopus LRP
# #2, not the P1 figure lifecycle, so it is never in that scheme at all).
# Bump by hand once the PI's om_release_v0_1_1 decision resolves and the
# package is actually sent to the team.
TEAM_RELEASE_STATUS = "draft"  # draft | sent | superseded



#: Page-local layout on top of hubkit.CSS; colours come from its tokens only,
#: so light and dark both follow the hub theme.
PAGE_CSS = """<style>
.top{margin-top:18px}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin:4px 0 16px}
.actions .btn a{display:inline-block;padding:9px 15px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--ink);text-decoration:none;font-weight:600;font-size:14px}
.actions .btn a:hover{border-color:var(--accent)}
.actions .primary a{background:var(--accent);border-color:var(--accent);color:var(--accent-ink);font-size:15px;padding:10px 18px}
.lede{font-size:17px;color:var(--lede);max-width:760px}
.terms p{margin:4px 0}
ul.new{padding-left:20px;max-width:820px}ul.new li{margin:6px 0}
.counts{display:flex;flex-wrap:wrap;gap:14px}.count strong{font-size:18px;margin-left:4px}
details summary{cursor:pointer;color:var(--accent);font-weight:600;margin:6px 0}
.scroll{overflow:auto;border:1px solid var(--line);border-radius:8px}.scroll.tall{max-height:480px}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--bg-soft)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.tile{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tile img{width:100%;height:220px;object-fit:contain;background:#fff;display:block;cursor:zoom-in;
border-bottom:1px solid var(--line)}
.tile figcaption{padding:10px 12px;font-size:13px;color:var(--mut);line-height:1.45}
.tile figcaption strong{display:block;color:var(--ink);font-size:14px;margin-bottom:2px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px 28px}
.cols h3{font-size:15px;margin:4px 0 6px}
ul.files a{text-decoration:none}ul.files a:hover{text-decoration:underline}
ul.files{list-style:none;padding:0;margin:0}ul.files li{padding:5px 0;border-bottom:1px solid var(--line);font-size:14px}
.divider{margin:44px 0 0;border-top:2px solid var(--ink);padding-top:8px}
.divider span{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);font-weight:700}
a.ops{display:inline-block;padding:8px 14px;background:var(--accent);color:var(--accent-ink);border-radius:8px;
text-decoration:none;font-weight:600}a.ops code{background:none;color:inherit}
</style>"""


# ---------------------------------------------------------------- sources --

def _version_sort_key(name: str) -> tuple:
    m = re.match(r"^v(\d+)\.(\d+)(?:\.(\d+))?$", name)
    if not m:
        return (-1, -1, -1, name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), "")


def discover_versions(package_root: Path) -> list[str]:
    """Version directory names under mare_om2/, oldest first. A directory
    starting with '_' is internal-only and never a version."""
    if not package_root.is_dir():
        return []
    names = [p.name for p in package_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    return sorted(names, key=_version_sort_key)


def latest_version(package_root: Path) -> str | None:
    versions = discover_versions(package_root)
    return versions[-1] if versions else None


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_dictionary_rows(version_dir: Path) -> list[dict]:
    csv_path = layout.table_path(version_dir, "data_dictionary", "csv")
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_CHANGELOG_HEADER = re.compile(r"^##\s+(v\d+\.\d+(?:\.\d+)?)\s+—\s+(\d{4}-\d{2}-\d{2})\s*$")


def parse_changelog(changelog_path: Path) -> list[dict]:
    """One entry per '## vX.Y — YYYY-MM-DD' section, bullets in order
    (wrapped continuation lines are joined onto the bullet they belong to)."""
    if not changelog_path.exists():
        return []
    entries: list[dict] = []
    current: dict | None = None
    for raw in changelog_path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        m = _CHANGELOG_HEADER.match(ln)
        if m:
            if current:
                entries.append(current)
            current = {"version": m.group(1), "date": m.group(2), "bullets": []}
            continue
        if current is None:
            continue
        if ln.startswith("- "):
            current["bullets"].append(ln[2:].strip())
        elif ln and current["bullets"] and not ln.startswith("#"):
            current["bullets"][-1] += " " + ln
    if current:
        entries.append(current)
    return entries


def version_history(package_root: Path, latest_dir: Path) -> list[dict]:
    """Version history from each version's CHANGELOG (read off the newest
    version's cumulative CHANGELOG.md) cross-checked against each version's
    own manifest.json for build time and point count."""
    entries = parse_changelog(internal_dir_for(latest_dir) / "CHANGELOG.md")
    for e in entries:
        vdir = package_root / e["version"]
        e["on_disk"] = vdir.is_dir()
        manifest = load_json(vdir / "manifest.json") if e["on_disk"] else None
        e["built_at_utc"] = (manifest or {}).get("built_at_utc")
        routes = (manifest or {}).get("routes") or []
        om2 = next((r for r in routes if r.get("route_id") == "OM_2"), None)
        e["n_points"] = (om2 or {}).get("n_points")
    return entries


def quality_summary(quality: dict) -> dict:
    n_points = quality.get("n_points", 0)
    flagged = quality.get("route_geometry_flagged_points")
    pct = round(100 * flagged / n_points, 1) if flagged is not None and n_points else None
    below_100 = []
    for col, info in quality.get("columns", {}).items():
        cov = info.get("coverage_fraction")
        if cov is not None and cov < 1.0:
            below_100.append({
                "column": col,
                "pct": round(cov * 100, 1),
                "n_valid": info.get("n_valid"),
                "n_total": info.get("n_total"),
            })
    below_100.sort(key=lambda r: r["pct"])
    return {
        "n_points": n_points,
        "flagged": flagged,
        "flagged_pct": pct,
        "below_100": below_100,
        "pending_items": list(quality.get("pending_items", [])),
        "descoped_items": list(quality.get("descoped_items", [])),
        "descoped_by": quality.get("descoped_by", ""),
    }


# ---------------------------------------------------------------- render --

def _shade_step_min(version_dir: Path) -> int:
    """Time step of the P-05 shade table, read from its timestamps."""
    import pyarrow.parquet as pq

    t = pq.ParquetFile(layout.table_path(version_dir, "building_shade", "parquet")).read_row_group(0, columns=["timestamp_utc"]).to_pandas()["timestamp_utc"]
    return int(pd.Series(sorted(t.unique())).diff().min() / pd.Timedelta(minutes=1))


def _rel_to(from_dir: Path, target: Path) -> str:
    """POSIX relative href from from_dir to target, for a page served from
    the repo root (this project's hub convention — see build_project_hub.py)."""
    import os
    return os.path.relpath(target, from_dir).replace("\\", "/")


def _table(headers: list[str], rows: list[list[str]], *, escape_cols: set[int] | None = None) -> str:
    escape_cols = escape_cols if escape_cols is not None else set(range(len(headers)))
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = []
        for i, cell in enumerate(row):
            cell = "" if cell is None else str(cell)
            tds.append(f"<td>{html.escape(cell) if i in escape_cols else cell}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def render_page(root: Path) -> str:
    package_root = root / "outputs" / "_packages" / "mare_om2"
    version = latest_version(package_root)
    if version is None:
        raise SystemExit(f"no version directory under {package_root} — run scripts/build_om_package.py first")
    version_dir = package_root / version
    manifest = load_json(version_dir / "manifest.json") or {}
    quality_path = layout.table_path(version_dir, "quality_report", "json")
    quality = load_json(quality_path) or {}
    q = quality_summary(quality)
    dict_rows = load_dictionary_rows(version_dir)
    history = version_history(package_root, version_dir)

    routes = manifest.get("routes") or []
    om2_route = next((r for r in routes if r.get("route_id") == "OM_2"), {})
    communities = om2_route.get("communities_crossed") or []

    pdf_path = version_dir / "report.pdf"
    pdf_rel = _rel_to(package_root, pdf_path)
    pdf_download = f"octopus_om2_{version}_report.pdf"
    readme_pdf_path = version_dir / "README.pdf"
    readme_pdf_rel = _rel_to(package_root, readme_pdf_path)
    p05 = manifest.get("p05_shade") or {}
    p10, p11 = manifest["p10"], manifest["p11"]
    built = manifest.get("built_at_utc", "?")
    built_label = built[:16].replace("T", " ") + " UTC" if len(built) >= 16 else built

    # --- one action row: the report first, everything else secondary -----
    actions = []
    zip_path = package_zip_path(version_dir)
    if zip_path.exists():
        zip_mb = zip_path.stat().st_size / 1e6
        actions.append(f'<span class="btn primary"><a href="{_rel_to(package_root, zip_path)}" '
                       f'download="{html.escape(zip_path.name)}">Download full package (ZIP, {zip_mb:.0f} MB)</a></span>')
    if pdf_path.exists():
        actions.append(f'<span class="btn{"" if zip_path.exists() else " primary"}"><a href="{pdf_rel}" '
                       f'download="{html.escape(pdf_download)}">Download report (PDF)</a></span>')
    actions.append(f'<span class="btn"><a href="{DECK_PDF}">Results slides (PDF)</a></span>')
    actions.append(f'<span class="btn"><a href="{DECK_PREVIEW}" target="_blank" rel="noopener">View slides</a></span>')
    if readme_pdf_path.exists():
        actions.append(f'<span class="btn"><a href="{readme_pdf_rel}" '
                       f'download="octopus_om2_{html.escape(version)}_README.pdf">Technical README (PDF)</a></span>')
    status_badges = (
        hubkit.badge("info", f"version {version}")
        + hubkit.badge("warn", "INTERNAL REVIEW DRAFT")
        + hubkit.badge("amber", f"team release: {TEAM_RELEASE_STATUS}")
    )
    report_note = ("" if pdf_path.exists() else
                   '<p class="sub">Report PDF not built for this version: rebuild with scripts/build_om_package.py.</p>')
    status_html = f"""
<section id="status" class="top">
  <div class="actions">{"".join(actions)}</div>
  {report_note}
  <p class="lede">Street form, sun and ventilation proxies for the {om2_route.get("n_points", 0):,} points of the
  OM2 walking route in Maré ({", ".join(html.escape(c) for c in communities) or "?"}), for the Octopus LRP #2 team.
  Start with the report: it gives the results and the context needed to read them in a few pages.</p>
  <div class="callout terms"><p>{status_badges}</p>
  <p>{html.escape(manifest.get("use_terms", "?"))}</p>
  <p class="gloss">These badges describe this Octopus package only. They are independent of BRISA's own
  <code>release_class</code> lifecycle for P1 figures.</p></div>
</section>"""

    # --- what's new: the headline numbers of this version, from the manifest --
    new_items = [
        f"<strong>Sun exposure for every time of day</strong> over {html.escape(' to '.join(p10['window']))}: "
        f"{100 * p10['date_dependent_share']:.0f}% of daylight point-slots change with the date, so use the "
        "per-date results for the campaign dates.",
        "<strong>Two wind regimes</strong> from the Galeão airport reports: "
        + " and ".join(f"{html.escape(g['name'])} ({g['mean_direction_deg']:.0f}°)" for g in p11["campaign_regimes"])
        + ". Ventilation proxies (shelter angle, canyon alignment, frontal density, roughness) are given for each regime. "
        "Geometry-derived proxies, not measured airflow.",
        f"<strong>Walks</strong>: {manifest['walks']['n_walks']} logger walks on {manifest['walks']['n_dates']} dates, with an arrival "
        "time at every route point and sensor-matched values.",
    ]
    n_figs = len(sorted((version_dir / layout.FIGURES_DIR).glob("*.png")))
    new_items.append(f"<strong>A shorter report</strong> with all {n_figs} figures and a README reorganised for scanning.")
    new_html = f"""
<section id="new">
  <h2>What's new in {html.escape(version)}</h2>
  <ul class="new">{"".join(f"<li>{i}</li>" for i in new_items)}</ul>
</section>"""

    # --- spec: counts up front, the table behind a toggle ------------------
    conformance = load_json(internal_dir_for(version_dir) / "p00_spec_conformance.json")
    if conformance is None:
        conformance_html = """
<section id="conformance">
  <h2>Spec</h2>
  <p class="sub">No internal p00_spec_conformance.json found for this version: rebuild with
  scripts/build_om_package.py.</p>
</section>"""
    else:
        _STATUS_BADGE = {"delivered": "ok", "delivered (scoped)": "ok", "descoped": "info",
                         "partial": "amber", "pending": "warn"}
        items = conformance.get("items", [])
        conf_rows = [
            [
                it["id"],
                it["requirement"],
                hubkit.badge(_STATUS_BADGE.get(it["status"], "info"), it["status"]),
                it["evidence"],
                ", ".join(
                    list(it.get("pending_on") or [])
                    + [f"descoped — {d}" for d in it.get("decisions") or []]
                ) or "—",
            ]
            for it in items
        ]
        counts = []
        for status in ("delivered", "delivered (scoped)", "partial", "pending", "descoped"):
            n = sum(1 for it in items if it["status"] == status)
            if n:
                counts.append(f'<span class="count">{hubkit.badge(_STATUS_BADGE[status], status)} <strong>{n}</strong></span>')
        conformance_html = f"""
<section id="conformance">
  <h2>Spec</h2>
  <p>The team's package spec has {len(items)} items ({html.escape(items[0]["id"]) if items else ""} to
  {html.escape(items[-1]["id"]) if items else ""}). Status computed from this build:</p>
  <p class="counts">{"".join(counts)}</p>
  <p class="sub">A <em>scoped</em> or <em>descoped</em> part is a deliberate cut by PI decision, not a gap.</p>
  <details class="spec"><summary>Show the full spec table</summary>
  <p class="sub">Computed by <code>src/om_package/spec.py</code>.</p>
  <div class="scroll">{_table(["id", "requirement", "status", "evidence", "pending on / descoped"], conf_rows, escape_cols={0, 1, 3, 4})}</div>
  </details>
</section>"""

    # --- figure gallery: every figure, caption says what to look at ---------
    n_dates = p05.get("n_campaign_dates") or 0
    gallery_spec = [
        (layout.FIG["route"], "The route", "The OM2 route over the Maré buildings, with distance marks every 250 m and the neighbourhoods it crosses."),
        (layout.FIG["form"], "Street form along the route",
         f"Building height, height-to-width ratio, sky view factor and plan area density; grey = every metre, black = {SEGMENT_M} m means."),
        (layout.FIG["shade_map"], "Building shade on the walk dates",
         f"Share of daylight each point spends in direct sun over the {n_dates} walk dates (lighter = more sun)."),
        (layout.FIG["shade_calendar"], "Shade by date and time of day",
         "Share of route points in direct sun, one row per walk date, by time of day (Rio local time)."),
        (layout.FIG["sun_dose"], "Direct sun before each walk",
         "Clear-sky direct sun in the 1 and 3 hours before each walk reached each point, one row per walk."),
        (layout.FIG["wind"], "Wind regimes", "Wind direction at Galeão airport for the campaign season and 2015 to 2024, and each regime by hour of day."),
        (layout.FIG["vent_profiles"], "Ventilation along the route", "Windward frontal area density, canyon alignment and upwind shelter angle for both wind regimes."),
        (layout.FIG["shelter_maps"], "Upwind shelter angle maps", "Upwind shelter angle per point for each wind regime, on one colour scale."),
        (layout.FIG["svf_sensor"], "Sensor-matched sky view factor", "Sky view factor at 1 m and as a slow sensor on one walk would see it."),
        (layout.FIG["vent_schematic"], "How the ventilation measures are drawn", "Frontal area density, canyon alignment and upwind shelter angle, and the three flow regimes across a street."),
        (layout.FIG["flags"], "Flagged points", "Route points inside building outlines or away from a mapped street, by class, with the repaired positions."),
    ]
    tiles = []
    for name, title, caption in gallery_spec:
        path = version_dir / layout.FIGURES_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"gallery figure missing from {version_dir.name}: {path}")
        rel = _rel_to(package_root, path)
        cap = f"{title}{'' if title.endswith(('?', '.')) else '.'} {caption}"
        tiles.append(
            f'<figure class="tile"><a href="{rel}" target="_blank" rel="noopener" '
            f'onclick="event.preventDefault();zoom(\'{rel}\',\'{hubkit._js_attr(cap)}\')">'
            f'<img src="{rel}" alt="{html.escape(cap)}" loading="lazy"></a>'
            f'<figcaption><strong>{html.escape(title)}</strong> {html.escape(caption)}</figcaption></figure>'
        )
    contact_html = f"""
<section id="figures">
  <h2>Figures</h2>
  <p class="sub">Click a figure to enlarge it.</p>
  <div class="gallery">{"".join(tiles)}</div>
</section>"""

    # --- files ------------------------------------------------------------
    readme_rel = _rel_to(package_root, version_dir / "README.md")
    # Served as text/markdown under the hub's MorphoFavela mount, a .md file
    # reaches the tablet as raw markdown (PI, 2026-09-27: "impossible to
    # read"); the hub's /doc viewer renders any same-origin src through md.js.
    dash_version = "/morphofavela-dash/" + version_dir.relative_to(root).as_posix()
    readme_view = f"/doc?src={dash_version}/README.md"
    manifest_rel = _rel_to(package_root, version_dir / "manifest.json")
    dict_rel = _rel_to(package_root, layout.table_path(version_dir, "data_dictionary", "csv"))
    panel_rel = _rel_to(package_root, package_root / PANEL_PAGE_NAME)

    data_files = [
        (layout.table("route_points", "parquet"), "route points, one row per metre (GeoParquet)"),
        (layout.table("route_points", "gpkg"), "route points (GeoPackage)"),
        (layout.table("route_points", "csv"), "route points (CSV)"),
        (layout.table("building_shade", "parquet"), f"building shade per point and {_shade_step_min(version_dir)}-min step, campaign dates"),
        (layout.table("walks", "csv"), "one row per logger walk: timing, coverage, wind regime"),
        (layout.table("walk_points", "parquet"), "arrival time, shade, dose and sensor-matched values per walk and point"),
        (layout.table("sun_envelope", "parquet"), "sun class per point and local time of day over the season"),
        (layout.table("sun_envelope", "csv"), "sun envelope (CSV)"),
        (layout.table("sun_dose", "parquet"), f"clear-sky direct-sun dose, {'/'.join(map(str, p10['dose_hours']))} h"),
        (layout.table("horizon_profiles", "parquet"), "horizon angle per point and azimuth"),
        (layout.table("wind_regimes", "csv"), "the two wind regimes, campaign season and climatology"),
        (layout.table("wind_regime_by_hour", "csv"), "regime share by local hour"),
        (layout.table("data_dictionary", "csv"), "data dictionary"),
        (layout.SCRIPTS["aggregate_to_segments"], "re-aggregate the points to any segment length"),
        (layout.SCRIPTS["join_shade_example"], "example join of device data to the shade table"),
    ]
    data_files_html = "".join(
        f'<li><a href="{_rel_to(package_root, version_dir / label)}"><code>{html.escape(label)}</code></a>'
        f' <span class="sub">{html.escape(desc)}</span></li>'
        for label, desc in data_files if (version_dir / label).exists()
    )
    files_html = f"""
<section id="files">
  <h2>Files</h2>
  <div class="cols">
  <div><h3>Data</h3><ul class="files">{data_files_html or "<li class='sub'>None found on disk for this version.</li>"}</ul></div>
  <div><h3>Documents</h3><ul class="files">
    <li><a href="{pdf_rel}" download="{html.escape(pdf_download)}">report.pdf</a> <span class="sub">the short report</span></li>
    <li><a href="{readme_view}">README</a> <span class="sub">technical document (<a href="{readme_rel}">raw .md</a>, <a href="{readme_pdf_rel}">PDF</a>)</span></li>
    <li><a href="{manifest_rel}">manifest.json</a> <span class="sub">sha256 per file, version, CRS, use terms</span></li>
    <li><a href="{panel_rel}">Panel ruling</a> <span class="sub">the expert-panel review behind v0.1's must-fix list</span></li>
  </ul></div>
  </div>
</section>"""

    # --- for the PI: decision, what the team owes, quality ------------------
    named_team_str = ", ".join(NAMED_TEAM)
    decision_html = f"""
<section id="decision">
  <h2>What the PI decides</h2>
  <p>Release <strong>{html.escape(version)}</strong> to the named Octopus team
  ({html.escape(named_team_str)}) now? The PI rules on it in the brisaverse cockpit; this page has no tap of its own.</p>
  <p><a class="ops" href="{OPS_LINK}" target="_blank" rel="noopener">Rule on <code>{OPS_DECISION_ID}</code> at /ops</a></p>
</section>"""

    owes = [
        ("Sensor time constant",
         "The air-temperature sensor's response time as mounted (63% or 90%), to set the analysis segment length "
         "(README: Using the data)."),
        ("om_routes.gpkg",
         "The team's own walked route: makes point_id final and removes the route_geometry_flag defect."),
        ("2024 airborne LiDAR + footprints",
         "Replaces the 2019 geometry; every geometry input is already a build parameter."),
    ]
    owes_html = f"""
<section id="team-owes">
  <h2>What the team owes</h2>
  {_table(["Item", "Why it matters"], [[a, b] for a, b in owes])}
</section>"""

    below_rows = [[r["column"], f'{r["pct"]}%', f'{r["n_valid"]}/{r["n_total"]}'] for r in q["below_100"]]
    pending_html = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in q["pending_items"])
    descoped_html = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in q["descoped_items"])
    quality_html = f"""
<section id="quality">
  <h2>Quality</h2>
  <p>{q["n_points"]} OM2 points. <strong>{q["flagged"]}</strong> flagged by
  <code>route_geometry_flag</code>
  ({q["flagged_pct"]}%: inside a building footprint or &gt;{ROUTE_FLAG_MAX_STREET_DIST_M:g} m from the
  nearest street centreline).</p>
  <details><summary>{len(below_rows)} column(s) below 100% coverage; pending and descoped items</summary>
  {_table(["Column", "Coverage", "Valid / total"], below_rows) if below_rows else "<p class='sub'>None.</p>"}
  <p>Pending items (from <code>quality_report.json</code>):</p>
  <ul>{pending_html or "<li class='sub'>None.</li>"}</ul>
  <p>Descoped items (deliberate cut by decision <code>{html.escape(q["descoped_by"])}</code>, not gaps):</p>
  <ul>{descoped_html or "<li class='sub'>None.</li>"}</ul>
  </details>
</section>"""

    walks_csv = layout.table_path(version_dir, "walks", "csv")
    windows_rows = []
    if walks_csv.exists():
        with walks_csv.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                windows_rows.append([r.get("walk_id"), r.get("date"), r.get("start_local"), r.get("end_local"),
                                     r.get("coverage_share"), r.get("partial"), r.get("wind_regime")])
    if p05.get("n_rows"):
        shade_html = f"""
<section id="shade">
  <h2>Walks</h2>
  <p>{p05.get("n_walks")} walks on {p05.get("n_campaign_dates")} dates.
  {p05.get("shade_fraction_daylight_pct")}% of daylight rows are in building shade.
  Shade timestamps: <code>{html.escape(str(p05.get("tz")))}</code> local time, with a UTC column.</p>
  <details><summary>Walks (Rio local time)</summary>
  <div class="scroll">{_table(["walk", "date", "start_local", "end_local", "coverage_share", "partial", "wind regime"], windows_rows) if windows_rows else "<p class='sub'>No walks table found.</p>"}</div>
  </details>
</section>"""
    else:
        shade_html = """
<section id="shade">
  <h2>Walks</h2>
  <p class="sub">No shade rows in this build.</p>
</section>"""

    dict_table_rows = [[r.get("id", ""), r.get("definition", ""), r.get("unit", ""), r.get("status", "")]
                       for r in dict_rows]
    dict_html = f"""
<section id="dictionary">
  <h2>Data dictionary</h2>
  <p>{len(dict_rows)} variables (<a href="{dict_rel}">full dictionary as CSV</a>, with source, method and limits).</p>
  <details><summary>Show the dictionary</summary>
  <div class="scroll tall">{_table(["id", "definition", "unit", "status"], dict_table_rows)}</div>
  </details>
</section>"""

    history_rows = [
        [e["version"], e["date"], "yes" if e["on_disk"] else "superseded / not on disk",
         (e.get("built_at_utc") or "—")[:16].replace("T", " "), str(e.get("n_points")) if e.get("n_points") is not None else "—"]
        for e in history
    ]
    changelog_items = "".join(
        f"<li><strong>{html.escape(e['version'])}</strong> ({html.escape(e['date'])})<ul>"
        + "".join(f"<li>{html.escape(b)}</li>" for b in e["bullets"])
        + "</ul></li>"
        for e in history
    )
    history_html = f"""
<section id="history">
  <h2>Version history</h2>
  {_table(["Version", "Date", "On disk", "Built (UTC)", "OM2 points"], history_rows)}
  <details><summary>Changelog detail</summary><ul>{changelog_items}</ul></details>
</section>"""

    # Same one-line <details class="glossary"> convention as the project hub
    # (build_project_hub.py::_recent_results_section).
    glossary_html = (
        '<section id="glossary"><div class="callout">'
        '<details class="glossary"><summary>Glossary</summary>'
        '<span class="gloss">'
        "P-02..P-11 this package's spec items (route points, aggregation, form variables, building shade, "
        "ventilation proxies, quality report, data dictionary, changelog, sun exposure, ventilation indices) · "
        "WP-02 the Brisa+ (MorphoFavela) shared horizon/sky-obstruction engine, reused unmodified "
        "for shade · "
        "lambda_p (plan_density_lambda_p) building plan-area fraction, 0-1, 1.0 = fully built · "
        "Tregenza sky the 145-patch sky-hemisphere subdivision the sky-view and shade computations sample · "
        "proxy a value derived from building geometry, never a measured air temperature, sunlight or wind"
        '</span></details></div></section>'
    )

    paper_html = f"""
<section id="paper">
  <h2>Paper link</h2>
  <p><a href="{PAPER_LINK}" target="_blank" rel="noopener">/paper/x1</a>: our role in Octopus LRP #2, the
  package spec, and what the next version waits on.</p>
</section>"""

    pi_html = '<div class="divider"><span>For the PI and the technical reader</span></div>'
    body = (
        PAGE_CSS + status_html + new_html + conformance_html + contact_html + files_html
        + pi_html + decision_html + owes_html + quality_html + shade_html + dict_html + history_html
        + glossary_html + paper_html
    )
    prov = hubkit.git_provenance(root, "scripts/build_om_package_page.py")
    return hubkit.page(
        "Octopus OM2 data package",
        f"Version <strong>{html.escape(version)}</strong> · built {html.escape(built_label)} · "
        "Maré, Rio de Janeiro · Brisa+ (MorphoFavela) for Octopus LRP #2",
        body,
        provenance=prov,
    )


def package_zip_path(version_dir: Path) -> Path:
    """The whole version directory as one download, next to it (outside the
    directory, so manifest.json never has to hash the archive of itself)."""
    return version_dir.parent / f"octopus_om2_{version_dir.name}.zip"


def write_package_zip(version_dir: Path) -> Path:
    """Zip every shipped file of version_dir under a top folder
    octopus_om2_<version>/; build temporaries (names starting with '_') stay out."""
    import zipfile

    out = package_zip_path(version_dir)
    tmp = out.with_suffix(".zip.part")
    top = out.stem
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(version_dir.rglob("*")):
            rel = f.relative_to(version_dir)
            if f.is_file() and not any(part.startswith("_") for part in rel.parts):
                zf.write(f, f"{top}/{rel.as_posix()}")
    tmp.replace(out)
    return out


def build_panel_page(root: Path) -> Path | None:
    """Render the panel-ruling markdown (repo docs/, never synced to the VPS)
    into a standalone page inside outputs/_packages/mare_om2/, so the package
    page's own link to it never points outside outputs/. Returns None if the
    source doc is missing (rendered page then also absent — check() reports
    the resulting dangling link rather than silently linking nothing)."""
    package_root = root / "outputs" / "_packages" / "mare_om2"
    src = root / PANEL_DOC_REL
    if not src.exists():
        return None
    dest = package_root / PANEL_PAGE_NAME
    hubkit.render_doc_page(src, dest, root=root, mirror_dir=package_root)
    return dest


def build_page(root: Path = DEFAULT_ROOT) -> Path:
    build_panel_page(root)
    out = root / "outputs" / "_packages" / "mare_om2" / "index.html"
    out.write_text(render_page(root), encoding="utf-8")
    return out


# ----------------------------------------------------------------- check --

_LOCAL_LINK = re.compile(r'(?:href|src)="([^"]+)"')
# git_provenance() bakes in datetime.now(); strip it before diffing two
# renders so --check compares content, not the wall-clock second it ran.
_PROVENANCE_TS = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")
# The footer's commit sha changes with every commit, so a page built one commit
# earlier failed --check and froze the tick (2026-10-02); provenance, not content.
_PROVENANCE_SHA = re.compile(r"(branch \S+ · )[0-9a-f]{7,40}( · )")


def _normalize(html_str: str) -> str:
    return _PROVENANCE_SHA.sub(r"\1<SHA>\2", _PROVENANCE_TS.sub("<TS>", html_str))


def check(root: Path = DEFAULT_ROOT) -> int:
    package_root = root / "outputs" / "_packages" / "mare_om2"
    out = package_root / "index.html"
    fails: list[str] = []

    try:
        fresh = render_page(root)
    except SystemExit as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    if not out.exists():
        fails.append(f"{out} does not exist — run scripts/build_om_package_page.py first")
    else:
        existing = out.read_text(encoding="utf-8")
        if _normalize(existing) != _normalize(fresh):
            fails.append(
                f"{out} differs from what the generator produces now from the on-disk "
                "sources — hand-edited or stale (never hand-edit a generated file; "
                "regenerate it instead)"
            )

    for m in _LOCAL_LINK.finditer(fresh):
        target = m.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path or target_path in HUB_ORIGIN_LINKS:
            continue
        if target_path.startswith("/doc?src=/morphofavela-dash/"):
            # hub markdown viewer over a file under the MorphoFavela mount:
            # the file it will fetch must exist in this checkout
            target_path = target_path[len("/doc?src=/morphofavela-dash/"):]
            resolved = (root / target_path).resolve()
        else:
            resolved = (package_root / target_path).resolve()
        if not resolved.exists():
            fails.append(f"link does not resolve: {target} -> {resolved}")

    if OPS_LINK not in fresh:
        fails.append(f"missing expected PI-decision link {OPS_LINK}")
    for deck_link in sorted(HUB_ORIGIN_LINKS):
        if f'href="{deck_link}"' not in fresh:
            fails.append(f"missing expected results-deck link {deck_link}")
    if PAPER_LINK not in fresh:
        fails.append(f"missing expected /paper/x1 link {PAPER_LINK}")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print(f"om_package_page --check OK ({out})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--check", action="store_true", help="validate the existing page, never write")
    args = ap.parse_args()
    root = Path(args.root)
    if args.check:
        return check(root)
    out = build_page(root)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
