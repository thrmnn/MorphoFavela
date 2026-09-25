"""Assemble one folder the PI can review every figure from, in one place.

Release class is read from each run's own manifest, never typed here: the
citywide and zoom families are withheld (red line L1), so the folder is a
*viewing* surface only — it never writes into shared/figures or papers/.
Seeing an artefact and releasing it are different acts (2026-09-17).

Navigation council ruling, 2026-09-24 (docs/critic/navigation_council_2026-09-24.md),
Phase 1: one ordered `RECORDS` list (no more SECTIONS/EXTRA split) whose
explicit `order:int` drives both the TOC and the body — asserted, not just
intended (G1). The page opens on a freshness stamp and "New this cycle"
(id="s-new"), then the curated records in order, then a link to `all.html`,
which now holds the multi-hundred-file sweep so the main page stays a few
tablet screens long.

Phase 4: this generator stays the only authority for "what exists on disk
this cycle" (MANIFEST.json) but JOINS `release_class` from brisaverse's
`shared/facts/p1_artifacts.json` by (run_of_record, filename) — it never
re-derives release_class itself (that stays gen_p1_artifacts.py's job,
ethics-critical). A figure with no register row renders with the
`unclassified` badge; it is still shown (release class never hides anything
from the PI — ruling §2). The staged rows are pulled into a dedicated
"Awaiting your call" block (id="s-awaiting", G3), each badge linking the
`/ops` promotion card.

Charter phase D / figure_organization_spec.md §7 O7 ("WP chips"): the sweep
(everything not in a curated RECORDS section) is grouped on `all.html` by
the results registry's own WP -> family -> run, one `id="wp-<KEY>"` node per
`config/work_packages.yaml` key — including keys with nothing swept this
cycle, so a chip never links to a missing anchor. `index.html` gets a "By
work package" chip row above the curated TOC, one chip per key, muted when
empty, each linking `all.html#wp-<KEY>`. The registry is rebuilt in-process
from the SAME disk state this cycle sweeps (never a possibly-stale
`outputs/_registry/results.json` — measure at point of use); a run that is
not the family's `current` head is collapsed under a `<details>`, never
dropped. A swept file with no registry row (not yet declared in any family,
or the registry generator is unavailable) renders in a final "Not yet in
the registry" section, grouped by the folder it sits in as before — release
class never hides anything from the PI (ruling §2) applies here too.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html as _html
import json
import re
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the citywide pair is 21298x6211 by design

ROOT = Path(__file__).resolve().parents[1]
BRISAVERSE_ROOT = Path.home() / "SCL" / "SCR" / "brisaverse"
THUMB_W = 1100
KEEP_DATED_FOLDERS = 3  # PI decision, ruling §7.1: 3 dated review folders
NEW_SINCE_CAP = 30  # "New this cycle" thumbnail cap before "see sections below"
OPS_PROMOTION_ANCHOR = "/ops#dec-wp07_figure_promotion"  # hub/gallery/paper.html's own link


def _image_dir(run_dir: Path) -> Path:
    """Where a run keeps its PNGs. Most write them beside the manifest; some use
    a figures/ subdirectory. Resolve it rather than assuming either."""
    return run_dir / "figures" if (run_dir / "figures").is_dir() else run_dir


def _latest(glob: str) -> Path | None:
    """Newest run holding images. Returns None rather than raising: a family
    still being produced should leave its section out, not break the folder."""
    hits = sorted(d for d in (ROOT / "runs").glob(glob)
                  if d.is_dir() and (d / "figure_manifest.json").is_file()
                  and list(_image_dir(d).glob("*.png")))
    return hits[-1] if hits else None


def _wp_by_figure(run_dir: Path) -> dict[str, str]:
    """Which work package each figure's numbers come from — derived, never typed:
    a figure cites ledger ids, each ledger entry names its source run, and the
    ledger's own runs_of_record maps a run back to its WP. The PI asked "what
    about WP4 and WP6 figures?" precisely because f2 and f4 never said so."""
    manifest = run_dir / "figure_manifest.json"
    if not manifest.exists():
        return {}
    man = json.loads(manifest.read_text())
    ledger_rel = man.get("ledger_source")
    if not ledger_rel:
        return {}
    ledger_path = ROOT / ledger_rel
    if not ledger_path.exists():
        return {}
    ledger = json.loads(ledger_path.read_text())
    entries = ledger.get("entries", {})
    run_to_wp = {run: wp.upper() for wp, run in
                 ledger.get("_meta", {}).get("runs_of_record", {}).items()}
    out = {}
    for fig in man.get("figures", {}).values():
        if fig.get("status") != "produced" or not fig.get("png_path"):
            continue
        wps = sorted({run_to_wp[r] for r in
                      (entries.get(i, {}).get("source", {}).get("run_id") for i in fig.get("ledger_ids_used", []))
                      if r in run_to_wp})
        if wps:
            out[Path(fig["png_path"]).name] = " + ".join(wps)
    return out


def _manifest_classes(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "figure_manifest.json"
    if not path.exists():
        return {}
    figs = json.loads(path.read_text()).get("figures", {})
    return {
        Path(f["png_path"]).name: {
            "release_class": f.get("release_class"),
            "red_line": f.get("red_line"),
            "pixel_m": (f.get("aggregation") or {}).get("pixel_m"),
            "n_cells": (f.get("aggregation") or {}).get("n_cells_aggregated"),
        }
        for f in figs.values() if f.get("status") == "produced" and f.get("png_path")
    }


# --------------------------------------------------------------------------
# Phase 4 join: read-only lookup into brisaverse's release-class register.
# gen_p1_artifacts.py stays the only authority for release_class (it parses
# red_lines.md §5 and cross-checks run manifests, ethics-critical); this
# generator only joins by (run_of_record, filename), never re-derives.
# --------------------------------------------------------------------------

def _load_p1_register() -> list[dict]:
    path = BRISAVERSE_ROOT / "shared" / "facts" / "p1_artifacts.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("artifacts", [])
    except (json.JSONDecodeError, OSError):
        return []


def _register_index(register: list[dict]) -> dict[tuple[str, str], dict]:
    """(run_of_record, filename) -> register row. Filename alone collides
    across runs (every WP-07 figure family reuses f1_/f2_/f3_/f4_): the run
    each figure actually came from is what disambiguates it."""
    idx: dict[tuple[str, str], dict] = {}
    for row in register:
        run = row.get("run_of_record")
        name = Path(row.get("image_url") or "").name
        if run and name:
            idx[(run, name)] = row
    return idx


def _register_hash_index(register: list[dict]) -> dict[tuple[str, str], dict]:
    """(filename, md5) -> register row, resolved against the register's own
    run_of_record on THIS disk. A fallback for the case the primary
    (run, filename) key misses because the review folder picked a
    differently-timestamped run of the same family — content hash still
    proves it is the figure the register describes, not a same-named one."""
    idx: dict[tuple[str, str], dict] = {}
    for row in register:
        run = row.get("run_of_record")
        name = Path(row.get("image_url") or "").name
        if not (run and name):
            continue
        run_dir = ROOT / "runs" / run
        if not run_dir.is_dir():
            continue
        src = _image_dir(run_dir) / name
        if not src.exists():
            continue
        try:
            h = hashlib.md5(src.read_bytes()).hexdigest()
        except OSError:
            continue
        idx[(name, h)] = row
    return idx


def _release_badge(row: dict | None) -> str:
    """withheld / staged / publishable / unclassified — the only four badges
    the ruling allows (§2). `state` decides first (staged is a state, not a
    class); release_class text decides the rest."""
    if row is None:
        return "unclassified"
    if row.get("state") == "staged":
        return "staged"
    rc = (row.get("release_class") or "").lower()
    if row.get("state") == "withheld" or "withheld" in rc:
        return "withheld"
    if "publishable" in rc:
        return "publishable"
    return "unclassified"


def _join_release(name: str, src: Path, run_name: str | None,
                   by_run_file: dict, by_hash: dict, hash_names: set) -> dict:
    """The joined fields folded into a figure's row: release_badge always
    present; register_id/register_state only when a row matched. Hashing is
    skipped unless the filename is one the register could plausibly know
    (hash_names) — the 628-file sweep must never pay for 628 reads."""
    row = by_run_file.get((run_name, name)) if run_name else None
    if row is None and name in hash_names and src.exists():
        try:
            h = hashlib.md5(src.read_bytes()).hexdigest()
        except OSError:
            h = None
        if h is not None:
            row = by_hash.get((name, h))
    out = {"release_badge": _release_badge(row)}
    if row is not None:
        out["register_id"] = row.get("id")
        out["register_state"] = row.get("state")
    return out


_RUN_SOURCE_RE = re.compile(r"^runs/([^/]+)/")


def _apply_release_badges(entries: list[dict], register: list[dict]) -> None:
    """Phase 4 join (G2/G3's counterpart on the MorphoFavela side): stamp
    every 'ok' entry with a release_badge from brisaverse's register, keyed
    off the same (run, filename) each entry was already copied under — no
    entry is re-classified by anything other than the register."""
    by_run_file = _register_index(register)
    by_hash = _register_hash_index(register)
    hash_names = {name for (name, _h) in by_hash}
    for e in entries:
        if e.get("status") != "ok":
            continue
        name = e["file"]
        m = _RUN_SOURCE_RE.match(e.get("source", ""))
        run_name = m.group(1) if m else None
        src_abs = ROOT / e["source"]
        e.update(_join_release(name, src_abs, run_name, by_run_file, by_hash, hash_names))


# --------------------------------------------------------------------------
# Record-specific path/render helpers. Each RECORDS entry below is either
# "resolve" (a run directory, old SECTIONS style) or "paths" (an explicit
# file list, old EXTRA style), plus an optional "extra" renderer for content
# that is not a figure grid.
# --------------------------------------------------------------------------

def _mare_territory_paths() -> list[Path]:
    """Maré's own territory pair first, then every other site's territory map
    for comparison, then the site sheet and brief. Deduplicated by resolved
    path so Maré's map (named explicitly, so it is guaranteed present even if
    a future site glob changes) never appears twice."""
    paths = [
        ROOT / "outputs/maré/territory/mare_territory_map.png",
        ROOT / "outputs/maré/territory/mare_irradiation_distributions.png",
    ]
    seen = {p.resolve() for p in paths if p.exists()}
    for p in sorted(ROOT.glob("outputs/*/territory/*_territory_map.png")):
        if p.resolve() not in seen:
            paths.append(p)
            seen.add(p.resolve())
    paths += [
        ROOT / "outputs/_distribution/site_dashboards/maré/folha_maré_A3.png",
        ROOT / "outputs/_distribution/site_dashboards/maré/folha_maré.pdf",
        ROOT / "docs/briefs/mare/mare_morphology_brief.pdf",
    ]
    return paths


def _newest_om2_package() -> Path | None:
    versions = sorted((ROOT / "outputs/_packages/mare_om2").glob("v*"),
                       key=lambda p: p.name)
    return versions[-1] if versions else None


def _octopus_contact_sheet() -> list[Path]:
    pkg = _newest_om2_package()
    return [pkg / "OM2" / "contact_sheet.png"] if pkg else []


def _has_pandoc() -> bool:
    return shutil.which("pandoc") is not None


def _render_markdown_file(src: Path, dest: Path, title: str) -> bool:
    """Render a markdown file to a standalone HTML page at dest. Prefers
    pandoc; falls back to an escaped <pre> block so a missing binary never
    drops the PI's only copy of the text (checked at call time, not cached,
    since the sandbox that built this may differ from the one that reads it)."""
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _has_pandoc():
        try:
            subprocess.run(
                ["pandoc", "-f", "gfm", "-t", "html", "--standalone",
                 "--metadata", f"title={title}", "-o", str(dest), str(src)],
                check=True, capture_output=True, timeout=30,
            )
            return True
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
            pass
    dest.write_text(
        f'<!doctype html><meta charset="utf-8"><title>{_html.escape(title)}</title>'
        '<body style="max-width:70ch;margin:32px auto;font:15px/1.6 -apple-system,'
        'BlinkMacSystemFont,\'Segoe UI\',sans-serif;padding:0 16px">'
        f'<h1>{_html.escape(title)}</h1>'
        f'<pre style="white-space:pre-wrap;word-break:break-word">{_html.escape(src.read_text())}</pre>'
        '</body>'
    )
    return True


def _render_csv_table(src: Path, dest: Path, title: str) -> bool:
    if not src.exists():
        return False
    with src.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    head, body_rows = rows[0], rows[1:]
    thead = "".join(f"<th>{_html.escape(c)}</th>" for c in head)
    trs = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(c)}</td>" for c in r) + "</tr>"
        for r in body_rows
    )
    dest.write_text(
        f'<!doctype html><meta charset="utf-8"><title>{_html.escape(title)}</title>'
        '<body style="font:14px/1.5 -apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;padding:16px">'
        f'<h1>{_html.escape(title)}</h1>'
        '<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 9px;'
        'text-align:left;vertical-align:top;font-size:13px}th{background:#f0f0f0}</style>'
        f'<table><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table></body>'
    )
    return True


def _summarize_quality_report(src: Path) -> str:
    """A short table from OM2/p07_quality_report.json: n_points plus per-column
    coverage. Not a copy of the JSON — a PI-readable digest of it."""
    if not src.exists():
        return ""
    data = json.loads(src.read_text())
    n = data.get("n_points")
    cols = data.get("columns", {})
    rows = []
    for name, info in cols.items():
        if not isinstance(info, dict):
            continue
        cov = info.get("coverage_fraction")
        cov_s = f"{cov * 100:.1f}%" if isinstance(cov, (int, float)) else "-"
        rows.append(f'<tr><td>{_html.escape(str(name))}</td>'
                    f'<td>{info.get("n_valid", "-")}/{info.get("n_total", "-")}</td>'
                    f'<td>{cov_s}</td></tr>')
    if not rows:
        return ""
    return (f'<p>{n} OM2 points. Per-column coverage from p07_quality_report.json:</p>'
            '<table style="border-collapse:collapse;font-size:13px;margin-bottom:8px">'
            '<tr><th style="text-align:left;padding:2px 10px">column</th>'
            '<th style="text-align:left;padding:2px 10px">valid/total</th>'
            '<th style="text-align:left;padding:2px 10px">coverage</th></tr>'
            + "".join(rows) + '</table>')


def _octopus_extra(section_out: Path) -> str | None:
    """The package's docs rendered as pages inside the review folder, plus a
    quality summary and a link to /paper/x1. Returns None (no extra block) if
    no package build exists on disk yet."""
    pkg = _newest_om2_package()
    if pkg is None:
        return None
    parts = [f'<p class="prov">outputs/_packages/mare_om2/{pkg.name}/</p>']

    docs = [
        (pkg / "README.md", section_out / "readme.html", "OM2 package README", "README"),
        (pkg / "CHANGELOG.md", section_out / "changelog.html", "OM2 CHANGELOG", "CHANGELOG"),
        (ROOT / "docs/critic/octopus_package_panel_2026-09-24.md",
         section_out / "panel_review.html", "Octopus package panel ruling", "panel ruling (2026-09-24)"),
    ]
    links = []
    for src, dest, title, label in docs:
        if _render_markdown_file(src, dest, title):
            links.append((f"{section_out.name}/{dest.name}", label))
    if _render_csv_table(pkg / "p08_data_dictionary.csv", section_out / "data_dictionary.html",
                          "OM2 data dictionary"):
        links.append((f"{section_out.name}/data_dictionary.html", "data dictionary (p08)"))

    if links:
        parts.append('<p>' + " · ".join(f'<a href="{href}">{label}</a>' for href, label in links) + '</p>')

    quality_html = _summarize_quality_report(pkg / "OM2" / "p07_quality_report.json")
    if quality_html:
        parts.append('<h3 style="font-size:15px;margin:18px 0 4px">P-07 quality report</h3>' + quality_html)

    parts.append('<p style="margin-top:14px"><a href="/paper/x1">/paper/x1 →</a></p>')
    return "\n".join(parts)


# --------------------------------------------------------------------------
# RECORDS — one ordered list, replacing the old SECTIONS list + EXTRA dict.
# Each record's explicit "order" is the ONLY thing that decides where it
# renders (G1) — never the slug, never dict/list position.
# --------------------------------------------------------------------------

RECORDS: list[dict] = [
    dict(
        order=1, slug="mare_territory", title="Maré — territory and site deliverables",
        blurb=(
            "What \"Maré\" means in each product: the data extent, the 16-community study area and "
            "the citywide definition on one map, then the site sheet and brief rebuilt on the study "
            "area — alongside every other site's territory map, for comparison. The citywide choice is "
            "the open card <a href=\"/ops\">mare_citywide_definition</a>; the interactive twin is "
            "<a href=\"/morphofavela-dash/outputs/_distribution/html_dashboards/maré/index.html\">here</a>; "
            "the full territory index is "
            "<a href=\"/morphofavela-dash/outputs/_hub/territory.html\">/_hub/territory.html</a>; "
            "the brief, deck and Folha de Rua refresher for your review are at "
            "<a href=\"/morphofavela-dash/outputs/_hub/mare_review/index.html\">_hub/mare_review</a>."
        ),
        paths=_mare_territory_paths,
    ),
    dict(
        order=2, slug="octopus_om2", title="Octopus OM2 morphology package (X1, contributor)",
        blurb=(
            "Street-form variables MorphoFavela contributed to Octopus LRP #2 (\"Street by street\", "
            "lead Jingxue, PI Simone) — built by <code>scripts/build_om_package.py</code>. No "
            "temperature analysis, no conclusions: that is the Octopus team's work. Panel-reviewed "
            "2026-09-24; the ranked must-fix list is still open."
        ),
        badge="internal review draft — Octopus team only",
        paths=_octopus_contact_sheet,
        extra=_octopus_extra,
    ),
    dict(
        order=3, slug="p1_solar_figures", title="P1 solar figures (f1-f4)",
        blurb=(
            "The four figures staged for the paper, each tagged with the work package its numbers come "
            "from. Your promotion ruling is the only thing between these and shared/figures. The "
            "numbers behind them, per work package, are at "
            "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_wp04.html\">WP04</a>, "
            "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_wp05.html\">WP05</a>, "
            "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_wp06.html\">WP06</a> and "
            "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_g3.html\">G3</a>."
        ),
        resolve=lambda: _latest("wp07_figures_*"),
    ),
    dict(
        order=4, slug="citywide_maps", title="Citywide maps (f5, f5b, f6)",
        blurb="Sky-view and irradiation across the whole 8.4 M-cell domain. Withheld under red line "
              "L1 — yours to read, not to circulate.",
        resolve=lambda: _latest("wp07_map_*"),
    ),
    dict(
        order=5, slug="zoom_favelas", title="Per-favela zoom extracts",
        blurb="Each study favela at the run's sampling pitch, sharing the citywide colour limits. "
              "Ipanema is absent: no bairro boundary exists on disk.",
        resolve=lambda: _latest("wp07_zoom_*"),
    ),
    dict(
        order=6, slug="terrain_vs_buildings", title="Terrain versus buildings",
        blurb="How much of the sun lost to an open flat horizon is the hill, and how much is what was "
              "built on it. The maps put terrain-only beside terrain-with-buildings on one colour scale.",
        resolve=lambda: _latest("terrain_split_*"),
    ),
    dict(
        order=7, slug="method_schematics", title="How the method works",
        blurb="The obstruction surface built from terrain and building tops, the ray march that decides "
              "whether a sky patch is blocked, and the matrix step that turns visibility into irradiation.",
        resolve=lambda: _latest("wp07_method_*"),
    ),
    dict(
        order=8, slug="morphotypes", title="Morphotypes and morphotopes",
        blurb="The cross-site signature work the weekly deck draws on.",
        paths=lambda: [ROOT / "outputs/cross_site/signature/figures_v2" / n for n in (
            "morphotype_schematics.png", "maps_morphotypes.png", "morphotope_maps.png",
            "morphotope_maps_repartition.png", "morphotope_profile.png",
            "morphotope_recurrence.png", "morphotope_stability.png",
            "fingerprint_heatmap.png", "dendrogram.png", "composition_by_site.png",
            "k_selection_rigor.png", "experience_dotplots.png",
        )] + [ROOT / "outputs/cross_site/presentation_figures/fig_morpho_violins.png"],
    ),
    dict(
        order=9, slug="folha_de_rua", title="Folha de Rua site sheets",
        blurb="One A3 sheet per site: grid, terrain, density, then sky view and sunlight.",
        paths=lambda: sorted(ROOT.glob("outputs/_distribution/site_dashboards/*/folha_*_A3.png"))
                      + sorted(ROOT.glob("outputs/_distribution/site_dashboards/*/folha_*.pdf")),
    ),
    dict(
        order=10, slug="weekly_deck", title="Weekly update deck (W39)",
        blurb="Tomorrow's deck and its contact sheet.",
        paths=lambda: [Path.home() / "SCL/SCR/brisaverse/slides/brisa_wk39_update.pdf",
                        Path.home() / "SCL/SCR/brisaverse/slides/contact_wk39_update.png"],
    ),
]


def _assert_ordered(sections: list[dict]) -> None:
    """G1: the rendered order (TOC and body) must equal sorted(order). This is
    what would have caught the 2026-09-24 bug where a "00_" slug prefix
    silently won over the section's real place in the list."""
    orders = [s["order"] for s in sections]
    if orders != sorted(orders):
        raise AssertionError(f"sections rendered out of declared order: {orders}")


def _copy(src: Path, dest_dir: Path, meta: dict, entries: list, section: str) -> None:
    if not src.exists():
        entries.append({"section": section, "file": src.name, "status": "MISSING", "source": str(src)})
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    row = {"section": section, "file": src.name, "status": "ok",
           "source": str(src.relative_to(ROOT)) if ROOT in src.parents else str(src)}
    row.update({k: v for k, v in meta.items() if v is not None})
    if src.suffix.lower() == ".png":
        try:
            with Image.open(src) as im:
                row["pixels"] = list(im.size)
                im.draft("RGB", (THUMB_W, THUMB_W))
                thumb = im.copy()
                thumb.thumbnail((THUMB_W, THUMB_W * 4))
                tdir = dest_dir / "_thumbs"
                tdir.mkdir(exist_ok=True)
                thumb.convert("RGB").save(tdir / (src.stem + ".jpg"), quality=84)
            row["thumb"] = f"_thumbs/{src.stem}.jpg"
        except Exception as exc:
            row["thumb_error"] = type(exc).__name__
    st = src.stat()
    row["bytes"] = st.st_size
    # "New this cycle" (build()'s _compute_new_since) diffs this against the
    # previous dated folder's _utc — source mtime, not copy time, so a
    # re-run of this generator on an unchanged figure never re-flags it.
    row["src_mtime_utc"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries.append(row)


# Everything else that already exists under outputs/. Archived snapshots and the
# audit dashboards are excluded: they are stale copies, and showing the PI the
# same figure three times is worse than not showing it. This sweep now lands
# on all.html (ruling §2, PI decision §7.3), not the main page.
SWEEP_EXCLUDE = ("_review", "_thumbs", "/thumbs/", "_archive_",
                 "_distribution/audit", "_hub/wp07_staged", "_hub/thumbs")
SWEEP_SUFFIXES = (".png", ".svg", ".pdf")


def sweep_remaining(out_root: Path, already: set, entries: list) -> list:
    """Hardlink every other figure on disk, deduplicated by content. Grouped by
    the directory it came from, because that is what says which analysis it
    belongs to. These sections carry no "order" key: they render on all.html,
    never the main page, and _assert_ordered ignores them."""
    seen_hashes = set()
    groups = defaultdict(list)
    for src in sorted((ROOT / "outputs").rglob("*")):
        if src.suffix.lower() not in SWEEP_SUFFIXES or not src.is_file():
            continue
        t = str(src)
        if any(x in t for x in SWEEP_EXCLUDE) or src.name in already:
            continue
        h = hashlib.md5(src.read_bytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        groups[src.parent.relative_to(ROOT / "outputs").as_posix()].append(src)

    sections = []
    for rel, srcs in sorted(groups.items()):
        slug = "sweep/" + rel.replace("/", "__")
        for src in srcs:
            _copy(src, out_root / slug, {}, entries, slug)
        sections.append({"slug": slug, "title": rel, "blurb": "",
                         "provenance": f"outputs/{rel}", "group": "other"})
    return sections


# --------------------------------------------------------------------------
# Charter phase D / figure_organization_spec.md §7 O7 ("WP chips"): group the
# sweep by the results registry's WP -> family -> run, instead of by the
# folder it happens to sit in. This is a read-only join, same shape as the
# Phase 4 release-badge join above: the registry stays the only authority for
# `wp`/`family`/`lifecycle`, this generator only looks each swept path up.
# --------------------------------------------------------------------------

def _wp_key_order() -> list[tuple[str, str]]:
    """(key, title) pairs in config/work_packages.yaml's own declared order —
    the chip row's order and the full set of chips, including a key with
    nothing swept this cycle (shown muted, never omitted, so a chip never
    links to a missing anchor). Appends the registry's own synthetic
    UNASSIGNED bucket when the yaml declares any unassigned family, matching
    build_results_registry.py's wp:UNASSIGNED node."""
    path = ROOT / "config" / "work_packages.yaml"
    if not path.exists():
        return []
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return []
    out = [(k, (v or {}).get("title") or k) for k, v in (cfg.get("work_packages") or {}).items()]
    if cfg.get("unassigned"):
        out.append(("UNASSIGNED", "Unassigned"))
    return out


def _load_fresh_registry() -> dict:
    """A fresh results registry (organization_charter.md §2), rebuilt
    in-process from the SAME disk state this cycle is about to sweep —
    never a possibly-stale outputs/_registry/results.json (defended-number
    drift: measure at point of use, never cache). Returns {} if the
    registry generator is unavailable or errors, so a broken registry
    degrades the WP tree to one "Not yet in the registry" bucket rather
    than breaking the whole review-folder build."""
    try:
        import build_results_registry as brr
    except ImportError:
        return {}
    try:
        return brr.build()
    except Exception:
        return {}


def _group_sweep_by_wp(entries: list[dict], registry: dict) -> dict:
    """`{"wps": [...], "unmatched": [...]}`. `wps` covers every declared
    work_packages.yaml key in order (§ _wp_key_order), each with its
    families (declared order) and, per family, its runs — the `current`
    run's items open, every other lifecycle (superseded/draft/archived)
    collapsed under its own run node, rows kept, never deleted. A static
    family (no run axis) is emitted as a single `run_id: None` bucket.
    `unmatched` is every swept entry whose source path has no figure row in
    the registry at all — shown, never hidden (ruling §2)."""
    nodes = registry.get("nodes", {})
    fig_by_path = {n["path"]: n for n in nodes.values()
                   if n.get("kind") == "figure" and n.get("path")}

    wp_order = _wp_key_order()
    wp_titles = dict(wp_order)
    fam_order: dict[str, list[str]] = defaultdict(list)
    for node_id, n in nodes.items():
        if n.get("kind") != "family":
            continue
        wp_key = (n.get("parent") or "").removeprefix("wp:")
        fam_key = node_id.removeprefix("fam:")
        if fam_key not in fam_order[wp_key]:
            fam_order[wp_key].append(fam_key)

    by_wp: dict[str, dict] = {}
    unmatched: list[dict] = []
    for e in entries:
        if e.get("status") != "ok":
            continue
        fig = fig_by_path.get(e.get("source"))
        if fig is None:
            unmatched.append(e)
            continue
        wp_key, fam_key = fig.get("wp"), fig.get("family")
        parent = nodes.get(fig.get("parent"), {})
        if parent.get("kind") == "run":
            run_id = fig["parent"].removeprefix("run:")
            run_lifecycle = parent.get("lifecycle") or "current"
            run_utc = parent.get("run_utc")
        else:
            run_id, run_lifecycle, run_utc = None, "current", None
        fam_bucket = by_wp.setdefault(wp_key, {}).setdefault(fam_key, {})
        run_bucket = fam_bucket.setdefault(
            run_id, {"run_id": run_id, "lifecycle": run_lifecycle, "run_utc": run_utc, "items": []})
        run_bucket["items"].append(e)

    wps = []
    for wp_key, title in wp_order:
        families = []
        n_wp = 0
        for fam_key in fam_order.get(wp_key, []):
            fam_runs = by_wp.get(wp_key, {}).get(fam_key)
            if not fam_runs:
                continue
            runs = sorted(fam_runs.values(),
                          key=lambda r: (r["lifecycle"] != "current", r["run_utc"] or "", r["run_id"] or ""))
            n_fam = sum(len(r["items"]) for r in runs)
            if n_fam == 0:
                continue
            families.append({"key": fam_key, "runs": runs, "n": n_fam})
            n_wp += n_fam
        wps.append({"key": wp_key, "title": title, "families": families, "n": n_wp})

    return {"wps": wps, "unmatched": unmatched}


def _previous_cycle_utc(out_root: Path) -> str | None:
    """The prior dated folder's own MANIFEST._utc — the diff base for 'new
    this cycle'. None on the very first recorded cycle: nothing to compare
    against, so nothing is flagged new (never the whole folder)."""
    base = out_root.parent
    if not base.is_dir():
        return None
    siblings = sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name != out_root.name and (d / "MANIFEST.json").is_file()
    )
    if not siblings:
        return None
    try:
        return json.loads((siblings[-1] / "MANIFEST.json").read_text()).get("_utc")
    except (json.JSONDecodeError, OSError):
        return None


def _compute_new_since(entries: list[dict], curated_sections: list[dict], prev_utc: str | None) -> dict:
    """Every curated (non-sweep) figure whose source mtime is newer than the
    previous cycle, grouped by the section (family) it belongs to, in the
    same order those sections render in. Capped at NEW_SINCE_CAP total."""
    if prev_utc is None:
        return {"cutoff_utc": None, "families": [], "total": 0, "shown": 0}
    titles = {s["slug"]: s["title"] for s in curated_sections}
    by_slug: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("status") != "ok" or not e.get("src_mtime_utc"):
            continue
        if e["section"] not in titles:
            continue  # sweep entries never count toward "new this cycle"
        if e["src_mtime_utc"] <= prev_utc:
            continue
        by_slug.setdefault(e["section"], []).append(e)

    total = sum(len(v) for v in by_slug.values())
    shown = 0
    families = []
    for slug, items in by_slug.items():
        remaining = NEW_SINCE_CAP - shown
        take = items[:remaining] if remaining > 0 else []
        shown += len(take)
        families.append({"slug": slug, "title": titles[slug], "items": take, "n": len(items)})
    return {"cutoff_utc": prev_utc, "families": families, "total": total, "shown": shown}


def _compute_awaiting(entries: list[dict]) -> list[dict]:
    """G3: the staged rows — every entry the register join marked
    register_state == 'staged', in section then filename order. This is the
    page-side count that must equal the register's own staged count and the
    /paper/p1 staged strip (checked cross-repo by check_review_surface.py)."""
    return sorted(
        (e for e in entries if e.get("status") == "ok" and e.get("register_state") == "staged"),
        key=lambda e: (e["section"], e["file"]),
    )


def _prune_dated_folders(base: Path, keep: int) -> list[Path]:
    """Delete all but the newest `keep` dated folders — but only a folder
    whose own MANIFEST.json names this script as its generator. A directory
    here from anything else (or from an older, differently-shaped generator)
    is left alone. Ruling §7.1 (PI decision, laptop) plus §4: the hard-linked
    'weekly RM' twin follows the same rule, since it carries the same
    MANIFEST.json."""
    if not base.is_dir():
        return []
    candidates = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        man = child / "MANIFEST.json"
        if not man.is_file():
            continue
        try:
            gen = json.loads(man.read_text()).get("generator")
        except (json.JSONDecodeError, OSError):
            continue
        if gen == "scripts/build_pi_review_folder.py":
            candidates.append(child)
    candidates.sort(key=lambda p: p.name)
    stale = candidates[:-keep] if keep > 0 else candidates
    for child in stale:
        shutil.rmtree(child)
    return stale


def build(out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    sections: list[dict] = []

    for record in sorted(RECORDS, key=lambda r: r["order"]):
        slug, title, blurb = record["slug"], record["title"], record["blurb"]
        section_out = out_root / slug

        if "resolve" in record:
            run_dir = record["resolve"]()
            if run_dir is None:
                continue
            classes = _manifest_classes(run_dir)
            wps = _wp_by_figure(run_dir)
            img_dir = _image_dir(run_dir)
            for name in sorted(classes):
                meta = dict(classes[name])
                if name in wps:
                    meta["work_package"] = wps[name]
                _copy(img_dir / name, section_out, meta, entries, slug)
            provenance = str(run_dir.relative_to(ROOT))
        else:
            paths = record["paths"]() if callable(record["paths"]) else record["paths"]
            for src in paths:
                _copy(src, section_out, {}, entries, slug)
            provenance = "existing outputs/ and slides/ products"

        sec = {"slug": slug, "title": title, "blurb": blurb, "provenance": provenance,
               "order": record["order"]}
        if record.get("badge"):
            sec["badge"] = record["badge"]
        if record.get("extra"):
            extra_html = record["extra"](section_out)
            if extra_html:
                sec["extra_html"] = extra_html
        sections.append(sec)

    _assert_ordered(sections)

    # Drop section directories this build did not write. Renaming a section
    # otherwise leaves its old copy behind and the PI sees it twice.
    written = {s["slug"].split("/")[0] for s in sections}
    written.add("sweep")
    for child in out_root.iterdir():
        if child.is_dir() and child.name not in written:
            shutil.rmtree(child)

    already = {e["file"] for e in entries if e["status"] == "ok"}
    other_sections = sweep_remaining(out_root, already, entries)

    # Phase 4 join: stamp every entry (curated and swept) with a release_badge
    # from brisaverse's register, then pull the staged rows into their own
    # "Awaiting your call" list — G3's page-side half.
    _apply_release_badges(entries, _load_p1_register())
    awaiting = _compute_awaiting(entries)

    prev_utc = _previous_cycle_utc(out_root)
    new_since = _compute_new_since(entries, sections, prev_utc)

    # O7 (WP chips): group the sweep by the registry's WP -> family -> run,
    # for all.html's tree and index.html's chip row.
    sweep_entries = [e for e in entries if e.get("section", "").startswith("sweep/")]
    wp_tree = _group_sweep_by_wp(sweep_entries, _load_fresh_registry())

    manifest = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/build_pi_review_folder.py",
        "cycle_date": out_root.name,
        "sections": sections + other_sections,
        "files": entries,
        "new_since": new_since,
        "awaiting": awaiting,
        "wp_tree": wp_tree,
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out_root / "index.html").write_text(_render_index(manifest))
    (out_root / "all.html").write_text(_render_all(manifest))
    broken = dangling_relative_links(out_root)
    if broken:
        raise SystemExit(f"review folder links to files that do not exist: {broken[:10]}")
    return manifest


def dangling_relative_links(out_root: Path) -> list[str]:
    """Relative href/src targets on the folder's pages that resolve to nothing —
    the 2026-09-24 class where a section's documents were written one directory
    below the links pointing at them."""
    broken = []
    for page in (out_root / "index.html", out_root / "all.html"):
        for target in set(re.findall(r'(?:href|src)="([^"#?:]+)"', page.read_text())):
            if not target.startswith("/") and not (out_root / target).exists():
                broken.append(f"{page.name}: {target}")
    return sorted(broken)


_STYLE = """<style>
:root{--bg:#faf8f5;--ink:#1c1a17;--dim:#6b6560;--line:#ddd6cd;--warn:#b6482b;--draft:#c99a2e;--ok:#2f7d4f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:0 20px 80px}
header{padding:32px 0 8px;border-bottom:1px solid var(--line);margin-bottom:24px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:40px 0 2px}h3{font-size:15px;margin:22px 0 2px}
.stamp{color:var(--dim);font-size:13px;margin:0 0 20px}
.blurb{color:var(--dim);margin:0 0 4px;max-width:62ch}
.prov{color:var(--dim);font-size:12.5px;font-family:ui-monospace,monospace;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}
figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}
figure img{width:100%;display:block;background:#fff}
figcaption{padding:9px 11px;font-size:13px}
.name{font-family:ui-monospace,monospace;word-break:break-all}
.meta{color:var(--dim);font-size:12px;margin-top:3px}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;
border:1px solid var(--line);margin-right:5px}
.withheld{color:var(--warn);border-color:var(--warn)}
.wp{background:var(--ink);color:var(--bg);border-color:var(--ink);font-weight:600}
.draft{color:var(--draft);border-color:var(--draft);font-weight:600}
.reg{font-weight:600}
.reg-staged{color:var(--draft);border-color:var(--draft)}
.reg-withheld{color:var(--warn);border-color:var(--warn)}
.reg-publishable{color:var(--ok);border-color:var(--ok)}
.reg-unclassified{color:var(--dim);border-color:var(--line)}
a{color:inherit}.nolink{padding:9px 11px;font-size:13px}
#s-awaiting{margin-bottom:16px}
.toc{border:1px solid var(--line);border-radius:8px;padding:16px 18px;background:#fff;margin-bottom:8px}
.toc ul{list-style:none;margin:8px 0 16px;padding:0;display:grid;
grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:2px 18px}
.toc ul.cols{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
.toc li{font-size:13px;padding:2px 0;min-width:0;overflow-wrap:anywhere}
.toc li .dim{color:var(--dim)}
.toc .n{color:var(--dim);font-variant-numeric:tabular-nums}
#s-new{margin-bottom:16px}
.all-link{margin:48px 0 12px;font-size:14px}
.mark-row{margin:6px 0 14px}
#mark-reviewed{font:inherit;font-weight:600;padding:8px 14px;border-radius:6px;
border:1px solid var(--ink);background:var(--ink);color:var(--bg);cursor:pointer}
#mark-reviewed:disabled{opacity:.6;cursor:default}
.mr-msg{margin-left:10px;color:var(--dim);font-size:13px}
.mr-msg.mr-err{color:var(--warn)}
.mr-msg.mr-ok{color:var(--ok)}
.wp-chips{margin:6px 0 20px}
.wp-chips strong{font-size:13px;color:var(--dim);display:block;margin-bottom:7px}
.chip-row{display:flex;flex-wrap:wrap;gap:7px}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:13px;padding:5px 11px;
border-radius:99px;border:1px solid var(--line);background:#fff;text-decoration:none}
.chip .n{color:var(--dim);font-variant-numeric:tabular-nums}
.chip-muted{opacity:.42}
details.run-details{margin:6px 0 14px;border:1px solid var(--line);border-radius:8px;padding:2px 12px;background:#fff}
details.run-details summary{cursor:pointer;padding:8px 0;font-size:13px;color:var(--dim)}
details.run-details .grid{padding-bottom:14px}
</style>"""


# "Mark this cycle reviewed" (navigation council ruling §2 "Seen and
# decided", §5 phase 5): POSTs the brisaverse hub's /api/review-mark as a
# RELATIVE url — this page is only ever meaningfully live when served
# through the hub's /morphofavela-dash/ mirror (same origin as the API), so
# no base URL is hardcoded here. Opened as a bare file:// (the PI's own
# file-manager / weekly-RM copy, see mirror_to_weekly_rm above) the fetch
# simply fails and the button says so — never a silent no-op.
_MARK_REVIEWED_SCRIPT = """<script>
(function(){
  var btn=document.getElementById('mark-reviewed'), msg=document.getElementById('mark-reviewed-msg');
  if(!btn) return;
  btn.addEventListener('click', function(){
    var label=btn.textContent;
    btn.disabled=true; btn.textContent='Marking…';
    msg.className='mr-msg'; msg.textContent='';
    fetch('/api/review-mark', {method:'POST'})
      .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
      .then(function(d){
        btn.textContent='Reviewed \\u2713';
        msg.className='mr-msg mr-ok';
        msg.textContent='as of '+(d.last_reviewed_utc||'now')+' \\u2014 /now\\u2019s \\u201cNew this cycle\\u201d count will read 0.';
      })
      .catch(function(){
        btn.disabled=false; btn.textContent=label;
        msg.className='mr-msg mr-err';
        msg.textContent='\\u26a0 could not reach the hub \\u2014 this only works served through the hub (not a bare file:// open).';
      });
  });
})();
</script>"""


def _anchor(sl: str) -> str:
    return "s-" + sl.replace("/", "-").replace("__", "-")


def _release_badge_tag(e: dict) -> str:
    """The Phase-4 register badge — withheld / staged / publishable /
    unclassified, joined from brisaverse's p1_artifacts.json. Distinct from
    the .withheld tag above, which is the RUN's own self-declared proposal;
    this one is the PI-facing, ethics-reviewed classification."""
    badge = e.get("release_badge")
    if not badge:
        return ""
    label = f"register: {badge}" if badge != "unclassified" else "unclassified"
    if badge == "staged":
        return f'<a class="tag reg reg-staged" href="{OPS_PROMOTION_ANCHOR}">{label} → rule on this</a>'
    return f'<span class="tag reg reg-{badge}">{label}</span>'


def _figure_card(e: dict, slug: str) -> str:
    if e["status"] == "MISSING":
        return f'<figure data-file="{e["file"]}"><div class="nolink">missing: <span class="name">{e["file"]}</span></div></figure>'
    tags = _release_badge_tag(e)
    if e.get("work_package"):
        tags += f'<span class="tag wp">{e["work_package"]}</span>'
    if e.get("release_class") == "withheld":
        tags += f'<span class="tag withheld">withheld · {e.get("red_line", "L1")}</span>'
    elif e.get("release_class"):
        tags += f'<span class="tag">{e["release_class"]}</span>'
    bits = []
    if e.get("pixels"):
        bits.append(f'{e["pixels"][0]}×{e["pixels"][1]} px')
    if e.get("pixel_m"):
        bits.append(f'{e["pixel_m"]:g} m pitch')
    if e.get("n_cells"):
        bits.append(f'{e["n_cells"]:,} cells')
    bits.append(f'{e["bytes"] / 1e6:.1f} MB')
    img = (f'<a href="{slug}/{e["file"]}"><img src="{slug}/{e["thumb"]}" loading="lazy" alt=""></a>'
           if e.get("thumb") else "")
    link = f'<a href="{slug}/{e["file"]}">{e["file"]}</a>'
    return (f'<figure data-file="{e["file"]}">{img}<figcaption>{tags}<div class="name">{link}</div>'
            f'<div class="meta">{" · ".join(bits)}</div></figcaption></figure>')


def _render_new_since(new_since: dict) -> str:
    total = new_since.get("total", 0)
    if not new_since.get("cutoff_utc"):
        return ('<section id="s-new"><h2>New this cycle <span class="n">(n/a)</span></h2>'
                '<p class="blurb">First recorded cycle — no previous folder to compare against.</p>'
                '</section>')
    if total == 0:
        return ('<section id="s-new"><h2>New this cycle <span class="n">(0)</span></h2>'
                f'<p class="blurb">Nothing changed since {new_since["cutoff_utc"]}.</p></section>')
    parts = [f'<section id="s-new"><h2>New this cycle <span class="n">({total})</span></h2>'
             f'<p class="blurb">Every curated figure produced after the previous cycle '
             f'({new_since["cutoff_utc"]}), grouped by section.</p>']
    shown = 0
    for fam in new_since["families"]:
        if not fam["items"]:
            continue
        parts.append(f'<h3>{fam["title"]} <span class="n">{fam["n"]}</span></h3><div class="grid">')
        for e in fam["items"]:
            parts.append(_figure_card(e, fam["slug"]))
            shown += 1
        parts.append("</div>")
    if shown < total:
        parts.append(f'<p><a href="#s-toc">+{total - shown} more this cycle — see sections below ↓</a></p>')
    parts.append("</section>")
    return "\n".join(parts)


def _render_awaiting(awaiting: list[dict]) -> str:
    """Ruling §2 page structure item 3, G3: every staged row, generated from
    release_class, badge linking the /ops promotion card. Always rendered
    (even at zero) so the section's presence itself is not a signal."""
    if not awaiting:
        return ('<section id="s-awaiting"><h2>Awaiting your call <span class="n">(0)</span></h2>'
                '<p class="blurb">Nothing staged this cycle.</p></section>')
    parts = [f'<section id="s-awaiting"><h2>Awaiting your call <span class="n">({len(awaiting)})</span></h2>'
             f'<p class="blurb">Every figure the register (brisaverse '
             f'<code>shared/facts/p1_artifacts.json</code>) marks <code>staged</code> — '
             f'promoting or holding each one is your tap on '
             f'<a href="{OPS_PROMOTION_ANCHOR}">/ops</a>, not an agent\'s.</p><div class="grid">']
    for e in awaiting:
        parts.append(_figure_card(e, e["section"]))
    parts.append("</div></section>")
    return "\n".join(parts)


def _render_wp_chips(wp_tree: dict) -> str:
    """O7: one chip per config/work_packages.yaml key, always all of them —
    a WP with nothing swept this cycle still gets a chip (muted) and a
    matching all.html anchor, so a chip never opens onto nothing."""
    wps = wp_tree.get("wps") or []
    if not wps:
        return ""
    parts = ['<nav class="wp-chips" aria-label="By work package"><strong>By work package</strong>'
             '<div class="chip-row">']
    for wp in wps:
        cls = "chip chip-muted" if wp["n"] == 0 else "chip"
        parts.append(f'<a class="{cls}" href="all.html#wp-{_html.escape(wp["key"])}">'
                     f'{_html.escape(wp["title"])} <span class="n">{wp["n"]}</span></a>')
    parts.append("</div></nav>")
    return "\n".join(parts)


def _render_index(m: dict) -> str:
    # Deliberately NOT re-sorted here: the render step must prove the order it
    # was handed is already correct, not silently repair it (G1).
    curated = [s for s in m["sections"] if "order" in s]
    _assert_ordered(curated)

    by_section: dict[str, list] = {}
    for e in m["files"]:
        by_section.setdefault(e["section"], []).append(e)
    curated = [s for s in curated if by_section.get(s["slug"])]
    other_sections = [s for s in m["sections"] if s.get("group") == "other" and by_section.get(s["slug"])]

    new_since = m.get("new_since", {"cutoff_utc": None, "total": 0})
    stamp_bits = [f'AS OF {m["_utc"]}', f'cycle {m.get("cycle_date", "?")}']
    stamp_bits.append(
        f'{new_since.get("total", 0)} new since last cycle ({new_since["cutoff_utc"]})'
        if new_since.get("cutoff_utc") else "first recorded cycle"
    )

    parts = [f"""<!doctype html><meta charset="utf-8"><title>Figure review</title>
{_STYLE}
<header><h1>Figure review</h1>
<p class="stamp">{" · ".join(stamp_bits)}</p>
<p class="mark-row"><button id="mark-reviewed" type="button">Mark this cycle reviewed</button>
<span id="mark-reviewed-msg" class="mr-msg"></span></p>
<p class="blurb">Every figure this cycle produced, in one place. Cards marked
<span class="tag withheld">withheld · L1</span> are yours to read; they do not travel into
the paper or shared figures without your own tap.</p></header>"""]

    parts.append(_render_new_since(new_since))
    parts.append(_render_awaiting(m.get("awaiting", [])))
    parts.append(_render_wp_chips(m.get("wp_tree", {})))

    parts.append('<nav class="toc" id="s-toc"><strong>This review</strong><ul>')
    for s in curated:
        parts.append(f'<li><a href="#{_anchor(s["slug"])}">{s["title"]}</a> '
                     f'<span class="n">{len(by_section[s["slug"]])}</span></li>')
    parts.append("</ul></nav>")

    for s in curated:
        badge_html = f' <span class="tag draft">{_html.escape(s["badge"])}</span>' if s.get("badge") else ""
        parts.append(f'<h2 id="{_anchor(s["slug"])}">{s["title"]}{badge_html}</h2>'
                     + (f'<p class="blurb">{s["blurb"]}</p>' if s["blurb"] else "")
                     + f'<p class="prov">{s["provenance"]}</p>')
        if s.get("extra_html"):
            parts.append(s["extra_html"])
        parts.append('<div class="grid">')
        for e in by_section[s["slug"]]:
            parts.append(_figure_card(e, s["slug"]))
        parts.append("</div>")

    if other_sections:
        n_other = sum(len(by_section[s["slug"]]) for s in other_sections)
        parts.append(f'<p class="all-link">Everything else on disk: {n_other} figures in '
                     f'{len(other_sections)} folders — earlier and ongoing analyses, not curated. '
                     f'<a href="all.html">Open all.html →</a></p>')

    parts.append(_MARK_REVIEWED_SCRIPT)
    return "\n".join(parts)


def _render_all(m: dict) -> str:
    """Charter phase D / O7: the sweep, grouped by the registry's WP ->
    family -> run. Every declared work_packages.yaml key gets an
    `id="wp-<KEY>"` node — even an empty one — so index.html's chip row
    never links to a missing anchor (checked by a dedicated test, not the
    dangling-link guard, which only follows file targets, never `#...`
    fragments). Runs that are not their family's `current` head render
    collapsed under `<details>`: rows stay, they are just not open by
    default (organization_charter.md §4 lifecycle table)."""
    wp_tree = m.get("wp_tree") or {"wps": [], "unmatched": []}
    folder_title = {s["slug"]: s["title"] for s in m["sections"] if s.get("group") == "other"}

    parts = [f"""<!doctype html><meta charset="utf-8"><title>Everything else — figure review</title>
{_STYLE}
<header><h1>Everything else on disk</h1>
<p class="blurb">Every other figure under outputs/, deduplicated by content, grouped by the results
registry's work package → family → run (organization_charter.md §2). Superseded, draft and archived
runs are collapsed under their family, never deleted — open the run to see them.
<a href="index.html">← back to the review</a></p></header>"""]

    parts.append('<nav class="toc" id="tree"><ul class="cols">')
    for wp in wp_tree["wps"]:
        parts.append(f'<li><a href="#wp-{_html.escape(wp["key"])}">{_html.escape(wp["title"])} '
                     f'<span class="dim">{_html.escape(wp["key"])}</span></a> '
                     f'<span class="n">{wp["n"]}</span></li>')
    if wp_tree["unmatched"]:
        parts.append(f'<li><a href="#wp-UNCLASSIFIED">Not yet in the registry</a> '
                     f'<span class="n">{len(wp_tree["unmatched"])}</span></li>')
    parts.append("</ul></nav>")

    for wp in wp_tree["wps"]:
        parts.append(f'<h2 id="wp-{_html.escape(wp["key"])}">{_html.escape(wp["title"])} '
                     f'<span class="dim">{_html.escape(wp["key"])}</span> '
                     f'<span class="n">{wp["n"]}</span></h2>')
        if not wp["families"]:
            parts.append('<p class="blurb">Nothing swept for this work package this cycle — every '
                         'current figure it has is either curated above or has yet to run.</p>')
            continue
        for fam in wp["families"]:
            parts.append(f'<h3>{_html.escape(fam["key"])} <span class="n">{fam["n"]}</span></h3>')
            # O8 cleanup (organization_charter.md §4 lifecycle table +
            # figure_organization_spec.md §7 O8): a family with many
            # non-current runs (e.g. wp02_horizon's ~95 superseded runs)
            # used to render one <details> PER RUN — dozens of collapsed
            # widgets stacked in a row, which is itself noise the charter's
            # rubric criterion 4 flags ("no more than 4 cards before '+N
            # more'"). Group by lifecycle instead: ONE collapsed entry per
            # lifecycle bucket ("N earlier runs — superseded" / "draft" /
            # "archived"), with each run still nested inside so nothing is
            # lost — open the group, then open a run, to see its figures.
            non_current_by_lifecycle: dict[str, list[dict]] = defaultdict(list)
            for run in fam["runs"]:
                label = run["run_id"] or "static family (no run axis)"
                if run["run_id"] is None or run["lifecycle"] == "current":
                    parts.append(f'<p class="prov">{_html.escape(label)} · {run["lifecycle"]}</p>'
                                 '<div class="grid">')
                    for e in run["items"]:
                        parts.append(_figure_card(e, e["section"]))
                    parts.append("</div>")
                else:
                    non_current_by_lifecycle[run["lifecycle"]].append(run)

            # Fixed order matches the charter §4 table (superseded, draft,
            # archived); any future lifecycle value still renders, appended.
            lifecycle_order = ["superseded", "draft", "archived"]
            for lifecycle in sorted(non_current_by_lifecycle,
                                     key=lambda lc: (lifecycle_order.index(lc)
                                                      if lc in lifecycle_order else len(lifecycle_order), lc)):
                runs_in_group = non_current_by_lifecycle[lifecycle]
                n_runs = len(runs_in_group)
                n_figs = sum(len(r["items"]) for r in runs_in_group)
                noun = "earlier run" if lifecycle == "superseded" else "run"
                parts.append(f'<details class="run-details"><summary>{n_runs} {noun}'
                             f'{"s" if n_runs != 1 else ""} — {lifecycle} ({n_figs} figure'
                             f'{"s" if n_figs != 1 else ""})</summary>')
                for run in runs_in_group:
                    label = run["run_id"] or "static family (no run axis)"
                    parts.append(f'<details class="run-details"><summary>{len(run["items"])} figure(s) '
                                 f'— {_html.escape(label)} ({run["lifecycle"]})</summary><div class="grid">')
                    for e in run["items"]:
                        parts.append(_figure_card(e, e["section"]))
                    parts.append("</div></details>")
                parts.append("</details>")

    if wp_tree["unmatched"]:
        parts.append(f'<h2 id="wp-UNCLASSIFIED">Not yet in the registry '
                     f'<span class="n">{len(wp_tree["unmatched"])}</span></h2>'
                     '<p class="blurb">Swept figures with no work-package registry row yet — not '
                     'declared in config/work_packages.yaml, or the registry could not be built this '
                     'run. Shown, never hidden (organization_charter.md ruling §2), grouped by the '
                     'folder they sit in.</p>')
        by_folder: dict[str, list] = defaultdict(list)
        for e in wp_tree["unmatched"]:
            by_folder[e["section"]].append(e)
        for slug in sorted(by_folder, key=lambda s: folder_title.get(s, s)):
            items = by_folder[slug]
            parts.append(f'<h3>{_html.escape(folder_title.get(slug, slug))} '
                         f'<span class="n">{len(items)}</span></h3><div class="grid">')
            for e in items:
                parts.append(_figure_card(e, slug))
            parts.append("</div>")

    return "\n".join(parts)


# The PI browses this from a file manager, where a symlink shows up as a single
# file rather than a folder. The repo copy is what the hub mirrors; this one is
# a hard-linked twin, so it is a real directory that costs no extra disk.
WEEKLY_RM = Path.home() / "SCL" / "SCR" / "weekly RM"


def mirror_to_weekly_rm(built: Path, date_slug: str) -> Path:
    dest = WEEKLY_RM / date_slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(built, dest, copy_function=os.link)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", type=Path, default=None,
                        help="where to build this cycle's folder "
                             "(default: <root>/outputs/_review/<today>)")
    parser.add_argument("--root", type=Path, default=None,
                        help="repo root to read runs/ and outputs/ from, and to "
                             "build the default out_dir under (default: this "
                             "script's own repo — pass this when running from a "
                             "worktree, whose gitignored data/outputs live only "
                             "in the main checkout)")
    args = parser.parse_args()
    if args.root is not None:
        ROOT = Path(args.root).resolve()

    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = args.out_dir if args.out_dir is not None else ROOT / "outputs" / "_review" / date_slug
    man = build(out)
    twin = mirror_to_weekly_rm(out, out.name)
    pruned_review = _prune_dated_folders(out.parent, KEEP_DATED_FOLDERS)
    pruned_weekly = _prune_dated_folders(WEEKLY_RM, KEEP_DATED_FOLDERS)

    ok = sum(1 for e in man["files"] if e["status"] == "ok")
    missing = [e["file"] for e in man["files"] if e["status"] == "MISSING"]
    print(f"{out}: {ok} files in {len(man['sections'])} sections")
    print(f"also at {twin}")
    if missing:
        print("MISSING:", ", ".join(missing))
    if pruned_review:
        print(f"pruned {len(pruned_review)} older dated folder(s): "
              + ", ".join(p.name for p in pruned_review))
    if pruned_weekly:
        print(f"pruned {len(pruned_weekly)} older weekly-RM folder(s): "
              + ", ".join(p.name for p in pruned_weekly))
    print(f"served at: /morphofavela-dash/outputs/_review/{out.name}/index.html")
    print("stable:    /figures")
