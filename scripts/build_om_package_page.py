#!/usr/bin/env python3
"""The dedicated Octopus package page (figure_organization_spec.md §4).

Builds outputs/_packages/mare_om2/index.html — a stable URL across
versions. Reads the newest version directory under mare_om2/ that is not
internal (name not starting with '_') and renders, from files already on
disk (never recomputed, never guessed):

  - status: version, build time, use_terms banner, and a team-release
    badge (draft/sent/superseded) that is visually and textually distinct
    from BRISA release_class — this package is not part of that scheme
  - what the PI decides (the open om_release_v0_1_1 decision, linked to
    brisaverse's /ops — this page carries no tap of its own)
  - what the team owes (raw OM2 CSVs, clock/timezone, om_routes.gpkg,
    LiDAR)
  - the documents: README, data dictionary (as a table), changelog
    (version history), a quality summary read from p07_quality_report.json,
    the contact sheet, and the panel ruling
  - a link to brisaverse's /paper/x1

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
import hubkit  # noqa: E402

DEFAULT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")

BRISA_HUB = "https://brisa.theoalessandro.com"
OPS_DECISION_ID = "om_release_v0_1_2"
OPS_LINK = f"{BRISA_HUB}/ops#dec-{OPS_DECISION_ID}"
PAPER_LINK = f"{BRISA_HUB}/paper/x1"
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

TEAM_OWES = [
    ("More raw OM2 CSVs", "A 5-file, one-per-device pilot was pulled 2026-09-25 from Zenodo_release/fixed_data/ (which holds far more files than the pilot downloaded) — those 5 files have NO Latitude/Longitude column (I_1/I_3/I_4/O_3/O_4 schema), so whether they ARE the OM2 device is UNVERIFIED; a confirmed GPS-track CSV is still needed to exercise the spatial half of the join example."),
    ("Clock / timezone confirmation", "GPS-fix rows are UTC per firmware; RTC-fallback rows are unconfirmed. v0.1.2's P-05 table is computed with tz=\"UTC\" as a stated labelling choice, not a resolution (Q1)."),
    ("om_routes.gpkg", "The team's own walked route — repairs point_id from PROVISIONAL to final and removes the route_geometry_flag defect (Q2)."),
    ("2024 airborne LiDAR + 2026 terrestrial OM2 scan", "The PI's Drive LiDAR_DSM_DTM/2024/ folders (Maré, Rio das Pedras, Rocinha_Vidigal) are scaffolded but EMPTY — asked of Carlo Moroz (Q3); see docs/research/octopus_lidar_sources.md."),
]


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
    csv_path = version_dir / "p08_data_dictionary.csv"
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
    entries = parse_changelog(latest_dir / "CHANGELOG.md")
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
    }


# ---------------------------------------------------------------- render --

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
    quality_path = version_dir / "OM2" / "p07_quality_report.json"
    quality = load_json(quality_path) or {}
    q = quality_summary(quality)
    dict_rows = load_dictionary_rows(version_dir)
    history = version_history(package_root, version_dir)

    routes = manifest.get("routes") or []
    om2_route = next((r for r in routes if r.get("route_id") == "OM_2"), {})
    communities = om2_route.get("communities_crossed") or []

    # --- status -------------------------------------------------------
    status_badges = (
        hubkit.badge("info", f"version {version}")
        + hubkit.badge("warn", "INTERNAL REVIEW DRAFT")
        + hubkit.badge("amber", f"team release: {TEAM_RELEASE_STATUS}")
    )
    status_html = f"""
<section id="status">
  <h2>Status</h2>
  <p>{status_badges}</p>
  <p class="sub">Built {html.escape(manifest.get("built_at_utc", "?"))} ·
  CRS {html.escape(manifest.get("crs", "?"))} ·
  {om2_route.get("n_points", "?")} OM2 points ·
  crosses {", ".join(html.escape(c) for c in communities) or "?"}.</p>
  <div class="callout"><p class="lead">Use terms</p>
  <p>{html.escape(manifest.get("use_terms", "?"))}</p>
  <p class="gloss">The INTERNAL REVIEW DRAFT and team-release badges above describe
  <em>this Octopus package</em> only. They are independent of BRISA's own
  <code>release_class</code> lifecycle for P1 figures — this package is not
  part of that scheme at all, so the two badge systems must never be read
  as equivalent.</p></div>
</section>"""

    # --- PI decision ----------------------------------------------------
    named_team_str = ", ".join(NAMED_TEAM)
    decision_html = f"""
<section id="decision">
  <h2>What the PI decides</h2>
  <p>Release <strong>{html.escape(version)}</strong> to the named Octopus team
  — {html.escape(named_team_str)} — now? The panel's must-fix list (see the
  panel ruling below) has been applied. This page states the decision; it
  carries no tap of its own — the PI rules on it in the brisaverse cockpit.</p>
  <p><a href="{OPS_LINK}" target="_blank" rel="noopener"
  style="display:inline-block;padding:8px 14px;background:var(--accent);color:#fff;
  border-radius:8px;text-decoration:none;font-weight:600">
  → Rule on <code style="background:none;color:inherit">{OPS_DECISION_ID}</code> at /ops</a></p>
</section>"""

    # --- what the team owes ---------------------------------------------
    owes_rows = [[item, note] for item, note in TEAM_OWES]
    owes_html = f"""
<section id="team-owes">
  <h2>What the team owes</h2>
  {_table(["Item", "Why it's blocking"], owes_rows)}
</section>"""

    # --- quality ----------------------------------------------------------
    below_rows = [[r["column"], f'{r["pct"]}%', f'{r["n_valid"]}/{r["n_total"]}'] for r in q["below_100"]]
    pending_html = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in q["pending_items"])
    quality_html = f"""
<section id="quality">
  <h2>Quality (P-07)</h2>
  <p>{q["n_points"]} OM2 points. <strong>{q["flagged"]}</strong> flagged by
  <code>route_geometry_flag</code>
  ({q["flagged_pct"]}% — inside a building footprint or &gt;10 m from the
  nearest street centreline).</p>
  <p>{len(below_rows)} column(s) below 100% coverage:</p>
  {_table(["Column", "Coverage", "Valid / total"], below_rows) if below_rows else "<p class='sub'>None.</p>"}
  <p>Pending items (quoted exactly from <code>p07_quality_report.json</code>):</p>
  <ul>{pending_html or "<li class='sub'>None.</li>"}</ul>
</section>"""

    # --- P-05 shade / campaign windows -----------------------------------
    p05 = manifest.get("p05_shade") or {}
    windows_csv = version_dir / "p05b_campaign_windows.csv"
    windows_rows = []
    if windows_csv.exists():
        with windows_csv.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                windows_rows.append(
                    [
                        Path(r.get("csv_path", "")).name,
                        r.get("date"),
                        r.get("first_timestamp"),
                        r.get("last_timestamp"),
                        r.get("n_rows"),
                        r.get("has_gps"),
                        r.get("n_epoch_reset"),
                    ]
                )
    if p05.get("n_rows"):
        shade_html = f"""
<section id="shade">
  <h2>P-05 building shade — campaign windows</h2>
  <p><strong>{p05.get("n_rows")}</strong> (point x 5-min-timestamp) rows across
  <strong>{p05.get("n_campaign_dates")}</strong> campaign dates, from a
  <strong>{p05.get("n_csv_pilot")}</strong>-file pilot pull (one CSV per device).
  <strong>{p05.get("shade_fraction_pct")}%</strong> of rows shaded.
  Computed <code>tz={html.escape(str(p05.get("tz")))}</code> —
  a stated labelling choice, the campaign timezone stays UNRESOLVED.
  Horizon march <code>max_dist_m={p05.get("max_dist_m")}</code> m (not
  WP-04's 500 m citywide default — the extended-300m DTM/footprints layer
  has real nodata gaps closer than that; see the package README's Known
  limits).</p>
  {_table(["CSV", "date", "first_timestamp", "last_timestamp", "n_rows", "has_gps", "n_epoch_reset"], windows_rows) if windows_rows else "<p class='sub'>No campaign-windows table found.</p>"}
  <p class="sub">Walk windows above are as read off the raw CSVs by
  <code>infer_campaign_windows()</code>; the shade table's own windows are
  each padded to the enclosing hour before the 5-min sweep.</p>
</section>"""
    else:
        shade_html = """
<section id="shade">
  <h2>P-05 building shade — campaign windows</h2>
  <p class="sub">No campaign CSVs found at build time — this version ships the
  empty-schema P-05 table (see the package README's P-05 section).</p>
</section>"""

    # --- contact sheet ------------------------------------------------
    contact_rel = _rel_to(package_root, version_dir / "OM2" / "contact_sheet.png")
    contact_html = f"""
<section id="contact-sheet">
  <h2>Contact sheet</h2>
  <a href="{contact_rel}" target="_blank" rel="noopener">
  <img src="{contact_rel}" alt="OM2 contact sheet {html.escape(version)}" style="max-width:100%;border:1px solid var(--line);border-radius:8px"></a>
</section>"""

    # --- documents ------------------------------------------------------
    readme_rel = _rel_to(package_root, version_dir / "README.md")
    changelog_rel = _rel_to(package_root, version_dir / "CHANGELOG.md")
    manifest_rel = _rel_to(package_root, version_dir / "manifest.json")
    dict_rel = _rel_to(package_root, version_dir / "p08_data_dictionary.csv")
    panel_rel = _rel_to(package_root, package_root / PANEL_PAGE_NAME)

    # Deliverable data files this release exists to ship — linked directly so
    # a recipient never has to reverse-engineer paths out of manifest.json's
    # files map to reach them.
    data_files = [
        (version_dir / "OM2" / "points.parquet", "OM2/points.parquet", "route-point table (GeoParquet)"),
        (version_dir / "OM2" / "points.gpkg", "OM2/points.gpkg", "route-point table (GeoPackage)"),
        (version_dir / "OM2" / "points.csv", "OM2/points.csv", "route-point table (plain CSV)"),
        (version_dir / "p05_building_shade.parquet", "p05_building_shade.parquet", "P-05 shade (point x 5-min timestamp)"),
        (version_dir / "p05_building_shade.csv", "p05_building_shade.csv", "P-05 shade (CSV)"),
        (version_dir / "p05b_campaign_windows.parquet", "p05b_campaign_windows.parquet", "P-05 campaign walk windows"),
        (version_dir / "p05b_campaign_windows.csv", "p05b_campaign_windows.csv", "P-05 campaign walk windows (CSV)"),
    ]
    data_files_html = "".join(
        f'<li><a href="{_rel_to(package_root, p)}"><code>{html.escape(label)}</code></a> — {html.escape(desc)}</li>'
        for p, label, desc in data_files if p.exists()
    )

    dict_table_rows = [
        [r.get("id", ""), r.get("definition", ""), r.get("unit", ""), r.get("status", "")]
        for r in dict_rows
    ]
    dict_html = f"""
<section id="dictionary">
  <h2>Data dictionary (P-08)</h2>
  <p>{len(dict_rows)} variables (<a href="{dict_rel}">full dictionary, incl. source/method/limits, as CSV</a>).</p>
  <div style="max-height:480px;overflow:auto;border:1px solid var(--line);border-radius:8px">
  {_table(["id", "definition", "unit", "status"], dict_table_rows)}
  </div>
</section>"""

    history_rows = [
        [
            e["version"],
            e["date"],
            "yes" if e["on_disk"] else "superseded / not on disk",
            e.get("built_at_utc") or "—",
            str(e.get("n_points")) if e.get("n_points") is not None else "—",
        ]
        for e in history
    ]
    changelog_items = "".join(
        f"<li><strong>{html.escape(e['version'])}</strong> — {html.escape(e['date'])}<ul>"
        + "".join(f"<li>{html.escape(b)}</li>" for b in e["bullets"])
        + "</ul></li>"
        for e in history
    )
    docs_html = f"""
<section id="documents">
  <h2>Documents</h2>
  <ul>
    <li><a href="{readme_rel}">README.md</a> — release scope, coverage vs Table 1, sources, methods, known limits.</li>
    <li><a href="{changelog_rel}">CHANGELOG.md</a></li>
    <li><a href="{manifest_rel}">manifest.json</a> — per-file sha256, package_version, crs, use_terms.</li>
    <li><a href="{panel_rel}">Panel ruling</a> — the expert-panel review v0.1's must-fix list came from.</li>
  </ul>
  <h3>Data files</h3>
  <p class="sub">The actual deliverable — linked directly, not just via manifest.json's files map.</p>
  <ul>{data_files_html or "<li class='sub'>None found on disk for this version.</li>"}</ul>
  <h3>Version history</h3>
  {_table(["Version", "Date", "On disk", "Built (UTC)", "OM2 points"], history_rows)}
  <details><summary>Changelog detail</summary><ul>{changelog_items}</ul></details>
</section>"""

    # --- glossary ---------------------------------------------------------
    # Same one-line <details class="glossary"> convention as the project hub
    # (build_project_hub.py::_recent_results_section) — for a reader outside
    # MorphoFavela (e.g. Jingxue) meeting this package's shorthand cold.
    glossary_html = (
        '<section id="glossary"><div class="callout">'
        '<details class="glossary"><summary>Glossary</summary>'
        '<span class="gloss">'
        "P-02..P-08 this package's own pipeline steps (route points, buffer/segment "
        "aggregation, airborne form variables, building shade, ventilation proxies, "
        "quality report, data dictionary) · "
        "WP-02 MorphoFavela's shared horizon/sky-obstruction engine, reused unmodified "
        "for P-05 shade · "
        "lambda_p (plan_density_lambda_p) building plan-area fraction of a 10&nbsp;m grid "
        "cell, 0-1, 1.0 = fully built · "
        "Tregenza sky the 145-patch sky-hemisphere subdivision the sky-view-factor / "
        "shade computations sample directions from"
        '</span></details></div></section>'
    )

    paper_html = f"""
<section id="paper">
  <h2>Paper link</h2>
  <p><a href="{PAPER_LINK}" target="_blank" rel="noopener">→ /paper/x1</a> —
  our role in Octopus LRP #2, the package spec, and what v0.2 is waiting on.</p>
</section>"""

    body = (
        status_html + decision_html + glossary_html + owes_html + contact_html
        + quality_html + shade_html + dict_html + docs_html + paper_html
    )
    prov = hubkit.git_provenance(root, "scripts/build_om_package_page.py")
    return hubkit.page(
        "Maré morphology, OM2 — package",
        "Octopus LRP #2 contributor package · figure_organization_spec.md §4",
        body,
        provenance=prov,
        doc=True,
    )


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


def _normalize(html_str: str) -> str:
    return _PROVENANCE_TS.sub("<TS>", html_str)


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
        if not target_path:
            continue
        resolved = (package_root / target_path).resolve()
        if not resolved.exists():
            fails.append(f"link does not resolve: {target} -> {resolved}")

    if OPS_LINK not in fresh:
        fails.append(f"missing expected PI-decision link {OPS_LINK}")
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
