"""Tests for the Octopus package page builder (scripts/build_om_package_page.py).

The pure parsing/formatting functions are tested against synthetic fixtures
(no data dependency). render_page/check() need the real built package
(outputs/_packages/mare_om2/, gitignored) and are skipped when it's absent —
same convention as tests/test_om_package.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "build_om_package_page", repo_root / "scripts" / "build_om_package_page.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()
DEFAULT_ROOT = M.DEFAULT_ROOT
PACKAGE_ROOT = DEFAULT_ROOT / "outputs" / "_packages" / "mare_om2"

pytestmark_real = pytest.mark.skipif(
    not PACKAGE_ROOT.is_dir(), reason="mare_om2 package not built at the default root"
)


# --- version discovery ---------------------------------------------------

def test_version_sort_key_orders_semver():
    names = ["v0.2", "v0.1.1", "v0.1", "v0.10.0"]
    assert sorted(names, key=M._version_sort_key) == ["v0.1", "v0.1.1", "v0.2", "v0.10.0"]


def test_discover_versions_excludes_internal_dirs(tmp_path):
    for name in ["v0.1", "v0.1.1", "_internal", "_review"]:
        (tmp_path / name).mkdir()
    assert M.discover_versions(tmp_path) == ["v0.1", "v0.1.1"]


def test_latest_version_picks_highest_semver(tmp_path):
    for name in ["v0.1", "v0.1.1", "v0.2"]:
        (tmp_path / name).mkdir()
    assert M.latest_version(tmp_path) == "v0.2"


def test_discover_versions_empty_when_root_missing(tmp_path):
    assert M.discover_versions(tmp_path / "does_not_exist") == []
    assert M.latest_version(tmp_path / "does_not_exist") is None


# --- changelog parsing -----------------------------------------------------

_SAMPLE_CHANGELOG = """\
# Changelog — mare_om2

## v0.1.1 — 2026-09-24

Some intro text, not a bullet.

- First bullet with a
  wrapped continuation line.
- Second bullet.

## v0.1 — 2026-09-24

- Initial release bullet.
"""


def test_parse_changelog_entries_and_wrapped_bullets(tmp_path):
    p = tmp_path / "CHANGELOG.md"
    p.write_text(_SAMPLE_CHANGELOG)
    entries = M.parse_changelog(p)
    assert [e["version"] for e in entries] == ["v0.1.1", "v0.1"]
    assert entries[0]["date"] == "2026-09-24"
    assert entries[0]["bullets"][0] == "First bullet with a wrapped continuation line."
    assert entries[0]["bullets"][1] == "Second bullet."
    assert entries[1]["bullets"] == ["Initial release bullet."]


def test_parse_changelog_missing_file_returns_empty(tmp_path):
    assert M.parse_changelog(tmp_path / "nope.md") == []


# --- quality summary ---------------------------------------------------

def test_quality_summary_computes_pct_and_below_100():
    quality = {
        "n_points": 200,
        "route_geometry_flagged_points": 20,
        "columns": {
            "a": {"coverage_fraction": 1.0, "n_valid": 200, "n_total": 200},
            "b": {"coverage_fraction": 0.9, "n_valid": 180, "n_total": 200},
        },
        "pending_items": ["tree_shade", "height_change_2024_2026"],
    }
    s = M.quality_summary(quality)
    assert s["n_points"] == 200
    assert s["flagged"] == 20
    assert s["flagged_pct"] == 10.0
    assert [r["column"] for r in s["below_100"]] == ["b"]
    assert s["below_100"][0]["pct"] == 90.0
    assert s["pending_items"] == ["tree_shade", "height_change_2024_2026"]


def test_quality_summary_handles_empty_report():
    s = M.quality_summary({})
    assert s["n_points"] == 0
    assert s["flagged"] is None
    assert s["flagged_pct"] is None
    assert s["below_100"] == []
    assert s["pending_items"] == []


# --- real package (skipped if not built) -----------------------------------

@pytestmark_real
def test_render_page_contains_expected_links_and_numbers():
    html_str = M.render_page(DEFAULT_ROOT)
    assert M.OPS_LINK in html_str
    assert M.PAPER_LINK in html_str
    version = M.latest_version(PACKAGE_ROOT)
    quality = M.load_json(PACKAGE_ROOT / version / "OM2" / "p07_quality_report.json")
    assert f'<strong>{quality["route_geometry_flagged_points"]}</strong>' in html_str
    for item in quality["pending_items"]:
        assert item in html_str


@pytestmark_real
def test_check_passes_on_freshly_built_page():
    M.build_page(DEFAULT_ROOT)
    assert M.check(DEFAULT_ROOT) == 0


@pytestmark_real
def test_check_fails_on_hand_edited_page():
    M.build_page(DEFAULT_ROOT)
    out = PACKAGE_ROOT / "index.html"
    original = out.read_text(encoding="utf-8")
    try:
        out.write_text(original.replace("<h1>", "<h1>SABOTAGED "), encoding="utf-8")
        assert M.check(DEFAULT_ROOT) == 1
    finally:
        out.write_text(original, encoding="utf-8")


# --- report.pdf is the main download; README.pdf and the deck sit beside it --

@pytestmark_real
def test_report_pdf_is_main_download_and_readme_pdf_secondary():
    version = M.latest_version(PACKAGE_ROOT)
    version_dir = PACKAGE_ROOT / version
    for name in ("report.pdf", "README.pdf"):
        data = (version_dir / name).read_bytes()
        assert data.startswith(b"%PDF"), name
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "report.pdf" in manifest["files"] and "README.pdf" in manifest["files"]
    page = M.render_page(DEFAULT_ROOT)
    main = re.search(r'<a href="([^"]+)" download="([^"]+)">Download report \(PDF\)</a>', page)
    assert main, "no Download report (PDF) link"
    assert main.group(1).endswith(f"{version}/report.pdf")
    assert main.group(2) == f"octopus_om2_{version}_report.pdf"
    assert re.search(r'<a href="[^"]+README\.pdf"[^>]*>Technical README \(PDF\)</a>', page)


@pytestmark_real
def test_results_deck_links_on_page():
    page = M.render_page(DEFAULT_ROOT)
    assert '<a href="/decks/brisa_om_pk.pdf">Results slides (PDF)</a>' in page
    assert re.search(r'<a href="/decks/preview_om_pk\.html"[^>]*>View slides</a>', page)


def test_deck_links_are_the_only_unresolved_exception():
    assert M.HUB_ORIGIN_LINKS == {"/decks/brisa_om_pk.pdf", "/decks/preview_om_pk.html"}


@pytestmark_real
def test_gallery_shows_every_figure_and_spec_table_is_behind_a_toggle():
    version = M.latest_version(PACKAGE_ROOT)
    page = M.render_page(DEFAULT_ROOT)
    for png in (PACKAGE_ROOT / version / "OM2").glob("*.png"):
        assert re.search(rf'<figure class="tile"><a href="{re.escape(version)}/OM2/{re.escape(png.name)}"[^>]*zoom\(', page), png.name
    assert f"What's new in {version}" in page
    spec = page.split('<section id="conformance">', 1)[1].split("</section>", 1)[0]
    assert spec.index("<details") < spec.index("<table")
    assert page.index("Download report (PDF)") < page.index("<h2>Figures</h2>") < page.index('<section id="files">')


@pytestmark_real
def test_page_names_project_in_parenthetical_form():
    page = M.render_page(DEFAULT_ROOT)
    assert not re.search(r"(?<!Brisa\+ \()MorphoFavela(?!-dash)", re.sub(r"<footer>.*?</footer>", "", page, flags=re.S))
