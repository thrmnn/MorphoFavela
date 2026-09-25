#!/usr/bin/env python3
"""Per-site pages — figure_organization_spec.md §3 (phase O6, after O2).

Writes `outputs/_hub/sites/<site>.html` for each `config/sites.yaml` key,
plus a small `outputs/_hub/sites/index.html`. The site set is **exactly**
the keys of `config/sites.yaml` — the legacy calibration sites in
`SITE_NAMES` (borel, jacarezinho, morro_do_juramento) and the boundary-less
`ipanema` calibration site are never emitted, even though registry rows and
open decisions can mention them.

Each page, in spec order:
    1. Header — display name, study area (km²), rotation badge.
    2. Decisions touching this site (open `tasks.json` decisions whose text
       names the site) — each linked to `/ops#<id>`.
    3. Product slots — dashboard, territory map, A3 sheet, brief, deck. A
       slot with nothing on disk shows "not built for this site" rather
       than being omitted, so a reviewer sees the gap, not silence.
    4. One row per `work_packages.yaml` key — the count of `current`-
       lifecycle figures the registry tags with this site, up to 4
       thumbnails, and "+N in tree →" when there are more than shown.
    5. Caveats — `sites.yaml`'s optional per-site `caveats:
       [{id, affects, decision}]` list, rendered next to the row/slot named
       in `affects`, never as a page-level banner.

Maré is the one site with two declared boundary definitions (PI ruling
2026-09-24, `mare_citywide_definition` / `mare_site_study_area` in
brisaverse `shared/facts/tasks.json`): its page adds a "both definitions"
panel read straight from the newest `runs/mare_definitions_*/summary.json`
(never hand-typed — see MAREDEF, tasks.json), and links the MAREDEF note
when one exists on disk.

Usage:
    python scripts/build_site_pages.py [--root PATH]
    python scripts/build_site_pages.py --check [--root PATH]

--check does not write anything: it asserts the site set equals
config/sites.yaml's keys and that every one of the 5 product slots is
resolved (built or explicitly flagged "not built"), exiting 1 otherwise.

Zero third-party deps beyond PyYAML (already a repo dependency) — same
budget as build_results_registry.py / check_registry.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import shutil
import sys
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from hubkit import badge, breadcrumb, card, git_provenance, page, relativize_page, section  # noqa: E402
import build_results_registry as brr  # noqa: E402
import registry_join  # noqa: E402
from src.sites.territory import normalize_site_key  # noqa: E402

CONFIG = ROOT / "config"
BRISAVERSE_TASKS = Path.home() / "SCL" / "SCR" / "brisaverse" / "shared" / "facts" / "tasks.json"
# gen_dossiers.py's own generated "sites" field per dossier (shared/facts/
# gen_dossiers.py, _ledger_site_tokens) — the mechanical, ledger/claim-text
# -derived list of which of the fixed five sites a decision's own figures
# touch. Read alongside tasks.json (never instead of it): a decision whose
# id/prose IS the site-scoping (e.g. mare_citywide_definition) still matches
# through _site_variants below; this covers the citywide-figure class that
# prose alone under-reports (2026-09-25 live-round-1 finding 5 — a WP-07
# figure plotting all five favelas read as "touches Maré only" because only
# Maré happened to be named in the decision's own worked-example prose).
BRISAVERSE_DOSSIERS = Path.home() / "SCL" / "SCR" / "brisaverse" / "shared" / "facts" / "dossiers.json"
# Full external URL, not a root-relative "/ops#..." href: /ops is brisaverse's
# own hub route (a different repo/server), so it must never pass through this
# script's relativize_page (which treats a bare "/..." href as a path inside
# THIS repo's outputs/ tree and would mangle it) — same convention as
# build_om_package_page.py's BRISA_HUB/OPS_LINK. Anchor is "dec-<id>" (the
# literal id ops.html's own card sets, `id="dec-${esc(domId(d.id))}"`), which
# brisaverse's check_hub.py (15) also expects.
BRISA_HUB = "https://brisa.theoalessandro.com"

PRODUCT_SLOTS = ["dashboard", "territory_map", "a3_sheet", "brief", "deck"]
SLOT_LABEL = {
    "dashboard": "Dashboard", "territory_map": "Territory map",
    "a3_sheet": "A3 sheet", "brief": "Brief", "deck": "Deck",
}


# --------------------------------------------------------------------- config

def load_sites_yaml() -> dict:
    """Raw config/sites.yaml `sites:` mapping — always from THIS checkout
    (config is tracked; identical across worktrees), same convention as
    src.sites.territory.load_sites_config."""
    data = yaml.safe_load((CONFIG / "sites.yaml").read_text(encoding="utf-8"))
    return data.get("sites") or {}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _ascii_slug(site: str) -> str:
    return _strip_accents(site).lower().replace(" ", "_")


def _site_variants(site_key: str, display_name: str) -> list[str]:
    """Lowercased substrings that count as "this decision names this site" —
    both the accented registry key and its ASCII alias (site producers write
    both, e.g. data/maré/... vs. the "mare" slug in run/package names)."""
    return sorted({site_key.lower(), _ascii_slug(site_key), display_name.lower(),
                   _strip_accents(display_name).lower()})


# ------------------------------------------------------------------- tasks.json

def _load_open_decisions() -> list[dict]:
    if not BRISAVERSE_TASKS.exists():
        return []
    try:
        data = json.loads(BRISAVERSE_TASKS.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data.get("open_decisions") or []


def _load_dossier_sites() -> dict[str, list[str]]:
    """{decision_id: [ascii site slugs]} from brisaverse's generated
    dossiers.json, or {} if it's missing/unparseable — this is an ADDITION
    to the tasks.json prose match below, never a replacement, so a missing/
    stale dossiers.json degrades this to the old prose-only behaviour
    instead of hiding a decision entirely."""
    if not BRISAVERSE_DOSSIERS.exists():
        return {}
    try:
        data = json.loads(BRISAVERSE_DOSSIERS.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for ds in data.get("dossiers") or []:
        did = ds.get("id")
        sites = ds.get("sites") or []
        if did and sites:
            out[did] = [str(s).lower() for s in sites]
    return out


def decisions_for_site(site_key: str, display_name: str, open_decisions: list[dict],
                        dossier_sites: dict[str, list[str]] | None = None) -> list[dict]:
    variants = _site_variants(site_key, display_name)
    ascii_site = _ascii_slug(site_key)
    dossier_sites = dossier_sites or {}
    hits = []
    for d in open_decisions:
        text = " ".join(str(d.get(k, "")) for k in ("id", "question", "plain_summary", "context")).lower()
        text_norm = _strip_accents(text)
        if any(v in text or _strip_accents(v) in text_norm for v in variants):
            hits.append(d)
            continue
        if ascii_site in dossier_sites.get(d.get("id"), []):
            hits.append(d)
    return hits


# ---------------------------------------------------------------- product slots

def _first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        matches = sorted(glob.glob(str(root / pat)))
        if matches:
            return Path(matches[0])
    return None


def _url(root: Path, p: Path) -> str:
    return "/" + str(p.relative_to(root))


def resolve_slots(root: Path, site_key: str) -> dict[str, tuple[str | None, str]]:
    """{slot: (root-absolute url or None, human note)} for the 5 product
    slots. `root` is the checkout that owns data/outputs (gitignored, so it
    is only ever the main checkout — pass --root from a worktree)."""
    ascii_site = _ascii_slug(site_key)
    slots: dict[str, tuple[str | None, str]] = {}

    dash = root / "outputs" / "_distribution" / "html_dashboards" / site_key / "index.html"
    slots["dashboard"] = ((_url(root, dash), "Interactive per-favela dashboard")
                          if dash.exists() else (None, "not built for this site"))

    tmap_name = "mare_territory_map" if site_key == "maré" else f"{ascii_site}_territory_map"
    tmap = root / "outputs" / site_key / "territory" / f"{tmap_name}.png"
    slots["territory_map"] = ((_url(root, tmap), "Study-area / data-extent / citywide territory map")
                              if tmap.exists() else (None, "not built for this site"))

    a3 = root / "outputs" / "_distribution" / "site_dashboards" / site_key / f"folha_{site_key}_A3.png"
    slots["a3_sheet"] = ((_url(root, a3), "Folha de rua A3 print sheet")
                         if a3.exists() else (None, "not built for this site"))

    brief = _first_existing(root, [f"docs/briefs/{site_key}/*.pdf", f"docs/briefs/{ascii_site}/*.pdf"])
    slots["brief"] = ((_url(root, brief), brief.name) if brief else (None, "not built for this site"))

    deck = _first_existing(root, [f"outputs/_hub/{ascii_site}_review/*.pptx",
                                  f"outputs/_hub/{site_key}_review/*.pptx",
                                  f"outputs/_hub/{ascii_site}_review/*.pdf"])
    slots["deck"] = ((_url(root, deck), deck.name) if deck else (None, "not built for this site"))

    return slots


# ------------------------------------------------------------------ registry

def _registry_nodes_for_site(nodes: dict, site_key: str) -> list[dict]:
    out = []
    for node_id, n in nodes.items():
        if n.get("kind") != "figure" or n.get("lifecycle") != "current":
            continue
        site = n.get("site")
        if not site or normalize_site_key(site) != site_key:
            continue
        out.append({**n, "id": node_id})
    return out


def _is_withheld_layer(root_relative_path: str) -> bool:
    """True if `root_relative_path` (a registry node's `path`, repo-root-
    relative — i.e. "outputs/<site>/...", NOT stripped of that prefix) is a
    per-cell layer the hub must never inline a preview of — kept in sync
    with build_project_hub.py's `_is_withheld`, which takes the SAME rule
    but on a path already relative to outputs/ (no leading "outputs/"
    segment; see its own docstring): <site>/morphometrics/grid,
    <site>/svf_v2/*.gpkg, <site>/cfd*, and the whole runs/ tree, which is
    blanket-forbidden regardless of what a given run actually contains,
    per that function's own docstring and the WP_REPORTS_DIR comment next
    to it. A withheld node still counts toward the WP row's total; it is
    only excluded from the 4 thumbnails, so nothing is hidden — see it in
    the tree via "+N in tree →" instead of inlined here unreviewed."""
    parts = root_relative_path.split("/")
    if parts and parts[0] == "outputs":
        parts = parts[1:]
    if "runs" in parts:
        return True
    if len(parts) >= 3 and parts[1] == "morphometrics" and parts[2] == "grid":
        return True
    if len(parts) >= 2 and parts[1] == "svf_v2" and root_relative_path.endswith(".gpkg"):
        return True
    if len(parts) >= 2 and parts[1].startswith("cfd"):
        return True
    return False


def wp_rows_for_site(root: Path, work_packages: dict, nodes: dict, site_key: str) -> list[dict]:
    by_wp: dict[str, list[dict]] = {}
    for n in _registry_nodes_for_site(nodes, site_key):
        by_wp.setdefault(n["wp"], []).append(n)
    rows = []
    for wp_key, wp_cfg in work_packages.items():
        fam = by_wp.get(wp_key, [])
        fam.sort(key=lambda n: n.get("produced_utc") or "", reverse=True)
        previewable = [n for n in fam if not _is_withheld_layer(n["path"])]
        rows.append({
            "wp": wp_key, "title": wp_cfg.get("title", wp_key),
            "count": len(fam), "thumbs": previewable[:4],
        })
    return rows


# ----------------------------------------------------------------- rendering

def _slot_html(root: Path, slot: str, url_note: tuple[str | None, str], caveats: list[dict]) -> str:
    url, note = url_note
    label = SLOT_LABEL[slot]
    cav_html = "".join(f'<div class="pill amber" style="display:inline-block;margin-left:6px" '
                       f'title="{_esc(c["decision"])}">{_esc(c["id"])}</div>' for c in caveats)
    if url:
        p = root / url.lstrip("/")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
            body = f'<a href="{url}"><img src="{url}" style="max-width:260px;width:100%;height:auto;border:1px solid #ddd" loading="lazy" alt="{_esc(label)}"></a>'
        else:
            body = f'<a href="{url}">{_esc(note)}</a>'
        return f'<div class="cap" style="flex:1 1 200px;min-width:180px"><h3>{_esc(label)}</h3>{body}{cav_html}</div>'
    return (f'<div class="cap" style="flex:1 1 200px;min-width:180px;opacity:.65">'
           f'<h3>{_esc(label)}</h3><p>not built for this site</p>{cav_html}</div>')


def _wp_row_html(row: dict, caveats: list[dict], ascii_site: str, all_html_exists: bool) -> str:
    # Corrective step 3 (charter phase D): the badge under each thumbnail is
    # registry_join.badge_text() applied to the SAME registry node the
    # thumbnail's src comes from — never a locally reimplemented rule. This
    # is what makes the badge comparable, word for word, to the one the
    # review folder shows for the identical artefact (same content_hash).
    thumbs = "".join(
        f'<span style="display:inline-block;text-align:center;margin-right:4px">'
        f'<a href="{"/" + t["path"]}"><img src="{"/" + t["path"]}" loading="lazy" '
        f'style="width:90px;height:60px;object-fit:cover;border:1px solid #ddd" '
        f'alt="{_esc(t.get("id", ""))}"></a>'
        f'<div class="pill doc" style="font-size:.7em;margin-top:2px" '
        f'data-registry-badge="{_esc(t.get("id", ""))}">{_esc(registry_join.badge_text(t))}</div>'
        f'</span>'
        for t in row["thumbs"])
    more = row["count"] - len(row["thumbs"])
    # By work package (all.html, the O7 "charter tree" deliverable) does not
    # exist yet in this cycle — audit_hub_graph.py hard-fails any dangling
    # href, so the "+N in tree" pointer only renders once the target is
    # real; until then the count above is still honest, just not a link.
    if more > 0 and all_html_exists:
        more_html = f'<a href="/outputs/_hub/all.html?site={ascii_site}#wp-{row["wp"]}">+{more} in tree →</a>'
    elif more > 0:
        more_html = f'<span class="pill doc">+{more} more</span>'
    else:
        more_html = ""
    cav_html = "".join(f' <span class="pill amber" title="{_esc(c["decision"])}">{_esc(c["id"])}</span>'
                       for c in caveats)
    muted = ' style="opacity:.55"' if row["count"] == 0 else ""
    return (f'<div class="wprow"{muted}><strong>{_esc(row["title"])}</strong> '
           f'<span class="pill {"ok" if row["count"] else "doc"}">{row["count"]} current</span>'
           f'{cav_html}<div style="margin-top:4px">{thumbs}{more_html}</div></div>')


def _esc(s) -> str:
    return html.escape(str(s))


def _copy_run_asset(root: Path, sites_out: Path, src_rel: Path) -> str:
    """Copy a file that lives under runs/ into this page's own _assets/ dir
    and return its outputs/-relative served URL. A raw '/runs/...' href/src
    must never appear in the hub: build_project_hub.py's `_is_withheld`
    blanket-forbids the 'runs' path segment anywhere under outputs/_hub
    (its own docstring: "every run report is copied ... into this repo-
    owned ... directory instead", next to WP_REPORTS_DIR) — the same reason
    the WP-row thumbnails above skip runs/-sourced nodes. Prefixing the
    copy's name with the run id keeps two runs' same-named files apart."""
    assets = sites_out / "_assets"
    assets.mkdir(parents=True, exist_ok=True)
    dst = assets / f"{src_rel.parent.name}__{src_rel.name}"
    if not dst.exists() or dst.stat().st_mtime < src_rel.stat().st_mtime:
        shutil.copy2(src_rel, dst)
    return "/" + str(dst.relative_to(root))


def _mare_definitions_panel(root: Path, sites_out: Path, sites_cfg: dict) -> str:
    """Both P1 Maré definitions (PI ruling 2026-09-24, mare_citywide_definition
    — BOTH_investigate): A, the 6 IPP favela polygons (P1 run of record), and
    E, the IPP Territórios Sociais complex outline. Every number read fresh
    from the newest runs/mare_definitions_*/summary.json — never hand-typed.
    """
    runs = sorted(root.glob("runs/mare_definitions_*/summary.json"))
    if not runs:
        return '<p class="pill doc">Both definitions declared (PI ruling 2026-09-24) — no runs/mare_definitions_*/summary.json on this checkout yet.</p>'
    summary = json.loads(runs[-1].read_text(encoding="utf-8"))
    defs = summary.get("definitions", {})
    rows = []
    for key in ("A_ipp_complexo_mare", "E_ipp_complex_outline"):
        d = defs.get(key)
        if not d:
            continue
        svf_pct = d.get("svf", {}).get("citywide_percentile_position")
        kwh_pct = d.get("kwh_m2", {}).get("citywide_percentile_position")
        rows.append(f'<tr><td>{_esc(key)}</td><td>{_esc(d.get("label", ""))}</td>'
                   f'<td>{svf_pct:.1f}</td><td>{kwh_pct:.1f}</td></tr>' if svf_pct is not None
                   else f'<tr><td>{_esc(key)}</td><td>{_esc(d.get("label", ""))}</td><td>n/a</td><td>n/a</td></tr>')
    table = (f'<table><thead><tr><th>Definition</th><th>Boundary</th>'
            f'<th>SVF citywide percentile</th><th>kWh/m² citywide percentile</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')
    note = sites_cfg.get("maré", {}).get("definition_note", "")
    note_html = f'<p class="pill" style="display:block;white-space:normal">{_esc(note)}</p>' if note else ""
    maredef_note = _first_existing(root, ["docs/research/mare_definitions_note.md",
                                         "docs/research/MAREDEF*.md",
                                         "runs/mare_definitions_*/note.md",
                                         "runs/mare_definitions_*/MAREDEF*.md"])
    if maredef_note:
        note_url = (_copy_run_asset(root, sites_out, maredef_note) if "runs" in maredef_note.parts
                   else _url(root, maredef_note))
        maredef_html = f'<p><a href="{note_url}">MAREDEF note — why the gap →</a></p>'
    else:
        maredef_html = '<p class="pill doc">MAREDEF (decompose the A-vs-E gap) is open — no note on disk yet.</p>'
    return (f'<h2>Both definitions</h2><p class="lead">P1 run of record stays on A; '
           f'E is reported beside it (source: <code>{_esc(runs[-1].relative_to(root))}</code>).</p>'
           f'{table}{note_html}{maredef_html}')


_SUBUNIT_TABLE_COLUMNS = [
    ("name", "Subunit"), ("n_cells", "Cells"), ("n_built_cells", "Built cells"),
    ("lambda_p_median", "λp (median)"), ("H_mean_median", "H mean (median, m)"),
    ("far_median", "FAR (median)"), ("svf_median", "SVF (median)"),
    ("sun_winter_median_h", "Winter sun (median, h)"),
    ("kwh_m2_median", "Annual irradiation (median, kWh/m²)"),
]


def _subunit_morphology_panel(root: Path, site_key: str, sites_cfg: dict) -> str:
    """Per-subunit density/height/footprint/sky-view/winter-sun/annual-
    irradiation (src.sites.subunit_morphology, scripts/build_subunit_morphology.py)
    for a site that declares subunits in config/sites.yaml but is not Maré
    (Maré keeps its own dedicated per-neighbourhood brief and the "Both
    definitions" panel above; this function is never called for it — see
    build_site_page). Every number is read fresh from the newest
    runs/subunit_morphology_*/<site>/summary.json + subunit_morphology.csv
    — never hand-typed. Descriptive only: rows are geographic order (north
    to south, by mean grid-cell y), never ranked; a site whose subunit
    polygons tile its study area exactly reports zero "between communities"
    ground honestly rather than omitting the mechanism.
    Vidigal and Rocinha declare no subunits at all (single polygon,
    "Isolada") — that is reported plainly, not silently skipped, so a
    reviewer sees the gap rather than a missing section."""
    cfg = sites_cfg.get(site_key, {})
    if cfg.get("subunits") is None:
        return ('<h2>Subunits</h2><p class="pill doc">No subunits declared for this site '
               '(config/sites.yaml) — reported as a single polygon.</p>')
    runs = sorted(root.glob(f"runs/subunit_morphology_*/{site_key}/summary.json"))
    if not runs:
        return (f'<h2>Subunits</h2><p class="pill doc">Subunits declared — no '
               f'runs/subunit_morphology_*/{_esc(site_key)}/summary.json on this checkout yet.</p>')
    summary_path = runs[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_path = root / summary["csv"]
    if not csv_path.exists():
        return f'<h2>Subunits</h2><p class="pill doc">summary.json found but {_esc(str(csv_path))} is missing.</p>'
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    thead = "".join(f"<th>{_esc(label)}</th>" for _, label in _SUBUNIT_TABLE_COLUMNS)
    trs = []
    for r in rows:
        tds = []
        for key, _ in _SUBUNIT_TABLE_COLUMNS:
            v = r.get(key, "")
            if key == "name":
                tds.append(f"<td>{_esc(v)}</td>")
                continue
            try:
                tds.append(f"<td>{float(v):.2f}</td>")
            except (TypeError, ValueError):
                tds.append(f"<td>{_esc(v)}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    table_html = f'<table><thead><tr>{thead}</tr></thead><tbody>{"".join(trs)}</tbody></table>'
    between_note = (" No study-area ground fell outside a named subunit (0 &quot;between&quot; cells)."
                    if not summary.get("has_between_subunits") else "")
    note = (f'<p class="lead">Per-subunit density, height, footprint, sky view, winter sun and '
           f'annual irradiation — geographic order (north to south), never ranked.{between_note} '
           f'{summary["n_subunits"]} subunits, source: '
           f'<code>{_esc(str(summary_path.relative_to(root)))}</code>, WP-04 run '
           f'<code>{_esc(summary.get("wp04_run", ""))}</code>.</p>')
    return f'<h2>Subunits</h2>{note}{table_html}'


def build_site_page(root: Path, sites_out: Path, site_key: str, cfg: dict, sites_cfg: dict,
                    work_packages: dict, nodes: dict, open_decisions: list[dict], prov: str,
                    dossier_sites: dict[str, list[str]] | None = None) -> str:
    display = cfg["display_name"]
    tp_path = root / "data" / site_key / "territory_provenance.json"
    tp = json.loads(tp_path.read_text(encoding="utf-8")) if tp_path.exists() else None

    badges = []
    if tp:
        sa_km2 = tp["study_area"]["area_m2"] / 1e6
        badges.append(badge("ok", f"study area {sa_km2:.2f} km²"))
        badges.append(badge("info", f"rotation {tp.get('display_rotation_deg', 0)}°"))
    else:
        badges.append(badge("doc", "territory not built on this checkout"))

    caveats = cfg.get("caveats") or []
    caveats_by_target: dict[str, list[dict]] = {}
    for c in caveats:
        caveats_by_target.setdefault(c.get("affects", ""), []).append(c)

    dec_hits = decisions_for_site(site_key, display, open_decisions, dossier_sites)
    dec_html = ""
    if dec_hits:
        items = "".join(f'<li><a href="{BRISA_HUB}/ops#dec-{_esc(d["id"])}">{_esc(d.get("question", d["id"]))}</a></li>'
                        for d in dec_hits)
        dec_html = f'<h2>Decisions touching this site</h2><ul>{items}</ul>'
    else:
        dec_html = '<h2>Decisions touching this site</h2><p class="pill doc">None open right now.</p>'

    slots = resolve_slots(root, site_key)
    slots_html = ('<h2>Deliverables</h2><div style="display:flex;flex-wrap:wrap;gap:16px">' +
                 "".join(_slot_html(root, s, slots[s], caveats_by_target.get(s, [])) for s in PRODUCT_SLOTS) +
                 "</div>")

    rows = wp_rows_for_site(root, work_packages, nodes, site_key)
    ascii_site = _ascii_slug(site_key)
    all_html_exists = (root / "outputs" / "_hub" / "all.html").exists()
    rows_html = ('<h2>Work packages</h2>' +
                "".join(_wp_row_html(r, caveats_by_target.get(r["wp"], []), ascii_site, all_html_exists)
                        for r in rows))

    mare_html = _mare_definitions_panel(root, sites_out, sites_cfg) if site_key == "maré" else ""
    subunit_html = "" if site_key == "maré" else _subunit_morphology_panel(root, site_key, sites_cfg)

    body = dec_html + slots_html + mare_html + subunit_html + rows_html
    crumb = breadcrumb([("← Project hub", "../index.html"), (display, None)])
    return page(display, " ".join(badges), body, crumb=crumb, provenance=prov)


def build_index_page(sites_cfg: dict, prov: str) -> str:
    cards = [card(cfg["display_name"], f"Deliverables, decisions and current artifacts for {cfg['display_name']}.",
                  f"{site_key}.html", kind="ok")
            for site_key, cfg in sites_cfg.items()]
    crumb = breadcrumb([("← Project hub", "../index.html"), ("Sites", None)])
    return page("Sites", badge("ok", f"{len(sites_cfg)} sites"), section("Sites", cards), crumb=crumb, provenance=prov)


# --------------------------------------------------------------------- main

def check(root: Path) -> int:
    sites_out = root / "outputs" / "_hub" / "sites"
    sites_cfg = load_sites_yaml()
    expected = set(sites_cfg)
    built = {p.stem for p in sites_out.glob("*.html") if p.stem != "index"}
    ok = True
    if built != expected:
        missing, extra = expected - built, built - expected
        print(f"FAIL: site set mismatch. missing={sorted(missing)} extra={sorted(extra)}")
        ok = False
    for site_key in expected:
        slots = resolve_slots(root, site_key)
        unresolved = [s for s in PRODUCT_SLOTS if s not in slots]
        if unresolved:
            print(f"FAIL: {site_key} missing slot resolution for {unresolved}")
            ok = False
        else:
            summary = ", ".join(f'{s}={"built" if slots[s][0] else "not built"}' for s in PRODUCT_SLOTS)
            print(f"  {site_key}: {summary}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def main(root: Path | None = None, do_check: bool = False) -> int:
    # Output, like build_project_hub.py, is written under the DATA root
    # (--root), not the invoking checkout: every worktree building against
    # the same main checkout converges on the one outputs/_hub tree there,
    # rather than each worktree growing its own disconnected copy (outputs/
    # is gitignored, so there is nothing to merge back if they diverged).
    data_root = Path(root).resolve() if root is not None else ROOT
    if do_check:
        return check(data_root)

    out = data_root / "outputs" / "_hub"
    sites_out = out / "sites"
    docs_out = out / "docs"
    sites_out.mkdir(parents=True, exist_ok=True)
    sites_cfg = load_sites_yaml()
    work_packages = brr.load_work_packages().get("work_packages", {})
    registry = brr.build()
    nodes = registry["nodes"]
    open_decisions = _load_open_decisions()
    dossier_sites = _load_dossier_sites()
    prov = git_provenance(ROOT, "scripts/build_site_pages.py")

    for site_key, cfg in sites_cfg.items():
        html_out = build_site_page(data_root, sites_out, site_key, cfg, sites_cfg, work_packages, nodes, open_decisions, prov,
                                    dossier_sites)
        html_out = relativize_page(html_out, sites_out, data_root, mirror_dir=docs_out)
        (sites_out / f"{site_key}.html").write_text(html_out, encoding="utf-8")

    index_html = build_index_page(sites_cfg, prov)
    index_html = relativize_page(index_html, sites_out, data_root, mirror_dir=docs_out)
    (sites_out / "index.html").write_text(index_html, encoding="utf-8")

    print(f"wrote {len(sites_cfg)} site pages + index -> {sites_out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="checkout that owns data/ and outputs/ (default: this script's own repo)")
    parser.add_argument("--check", action="store_true",
                        help="assert the site set equals sites.yaml and every slot resolves; no write")
    args = parser.parse_args()
    sys.exit(main(root=args.root, do_check=args.check))
