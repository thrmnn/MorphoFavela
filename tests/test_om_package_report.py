"""Tests for the Octopus OM2 human report (src/om_package/report.py).

Needs the real built package (outputs/_packages/mare_om2/v0.1.3/,
gitignored); skipped when it is absent, as in tests/test_om_package_spec.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.om_package.report import PROJECT_FORM, STUDY_TITLE, FIGURES, render_report_markdown

DEFAULT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
PACKAGE_DIR = DEFAULT_ROOT / "outputs" / "_packages" / "mare_om2" / "v0.1.3"

pytestmark = pytest.mark.skipif(
    not PACKAGE_DIR.is_dir(), reason="mare_om2 v0.1.3 package not built at the default root"
)


@pytest.fixture(scope="module")
def report_md() -> str:
    return (PACKAGE_DIR / "report.md").read_text(encoding="utf-8")


def _linked_copy(dest: Path) -> Path:
    """The package as symlinks, so one file can be swapped without copying
    the shade table."""
    dest.mkdir()
    for f in PACKAGE_DIR.iterdir():
        (dest / f.name).symlink_to(f)
    return dest


def test_report_md_matches_a_fresh_render(report_md):
    assert render_report_markdown(PACKAGE_DIR) == report_md


def test_render_is_deterministic():
    assert render_report_markdown(PACKAGE_DIR) == render_report_markdown(PACKAGE_DIR)


def test_project_named_only_in_parenthetical_form(report_md):
    assert "MorphoFavela" in report_md
    assert "MorphoFavela" not in report_md.replace(PROJECT_FORM, "")
    assert report_md.count(PROJECT_FORM) <= 2


def test_no_internal_ids(report_md):
    for token in ("P-0", "om_", "OCTOPUS_", "src/", ".parquet", ".py"):
        assert token not in report_md, token


def test_four_figures_embedded(report_md):
    for name, _heading, _cls in FIGURES:
        assert re.search(rf"!\[[^\]]+\]\(OM2/{re.escape(name)}\)", report_md), name
        assert (PACKAGE_DIR / "OM2" / name).exists()


def test_study_title_matches_readme():
    readme = " ".join((PACKAGE_DIR / "README.md").read_text(encoding="utf-8").split())
    assert STUDY_TITLE in readme


def test_planted_manifest_value_changes_the_text(tmp_path):
    pkg = _linked_copy(tmp_path / "pkg")
    manifest = json.loads((PACKAGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    om2 = next(r for r in manifest["routes"] if r["route_id"] == "OM_2")
    original = f"{om2['n_points']:,} points"
    om2["n_points"] = 98765
    (pkg / "manifest.json").unlink()
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    text = render_report_markdown(pkg)
    assert "98,765 points" in text
    assert original not in text


def test_planted_spec_status_changes_the_spec_sentence(tmp_path):
    pkg = _linked_copy(tmp_path / "pkg")
    conf = json.loads((PACKAGE_DIR / "p00_spec_conformance.json").read_text(encoding="utf-8"))
    before = render_report_markdown(PACKAGE_DIR)
    for it in conf["items"]:
        it["status"] = "delivered"
    (pkg / "p00_spec_conformance.json").unlink()
    (pkg / "p00_spec_conformance.json").write_text(json.dumps(conf), encoding="utf-8")
    after = render_report_markdown(pkg)
    assert f"Of the {len(conf['items'])} items the team asked for, {len(conf['items'])} are delivered." in after
    assert after != before


def test_report_pdf_ships_in_manifest():
    pdf = PACKAGE_DIR / "report.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    manifest = json.loads((PACKAGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert "report.pdf" in manifest["files"]
    assert "report.md" in manifest["files"]


def test_disclosure_sweep_covers_report():
    hits = (PACKAGE_DIR / "p00_disclosure_hits.txt").read_text(encoding="utf-8")
    assert "report.md" in hits.splitlines()[4]


def test_manifest_daylight_share_equals_parquet():
    import pandas as pd
    manifest = json.loads((PACKAGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    p05 = manifest["p05_shade"]
    assert "shade_fraction_pct" not in p05
    shade = pd.read_parquet(PACKAGE_DIR / "p05_building_shade.parquet", columns=["sun_altitude_deg", "shaded"])
    day = shade[shade["sun_altitude_deg"] > 0]
    assert p05["shade_fraction_daylight_pct"] == round(100 * float(day["shaded"].mean()), 1)
    assert p05["shade_fraction_daylight_pct"] < round(100 * float(shade["shaded"].mean()), 1)
