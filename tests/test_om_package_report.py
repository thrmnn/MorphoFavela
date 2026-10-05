"""Tests for the Octopus OM2 report (src/om_package/report.py) and README
(src/om_package/package_docs.render_readme).

Needs the real built package (outputs/_packages/mare_om2/<VERSION>/,
gitignored); skipped when it is absent, as in tests/test_om_package_spec.py.
Both documents are rendered fresh from the package files, so the tests check
the renderers, not a stale report.md on disk.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.om_package.package_docs import USE_TERMS, VERSION, render_readme
from src.om_package.report import (AUTHOR, FIGURES, PROJECT_FORM, STUDY_TITLE, _Pcts, compute_facts,
                                   render_report_markdown, write_report)

DEFAULT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
PACKAGE_DIR = DEFAULT_ROOT / "outputs" / "_packages" / "mare_om2" / VERSION

pytestmark = pytest.mark.skipif(
    not (PACKAGE_DIR / "p12_walk_points.parquet").is_file(),
    reason=f"mare_om2 {VERSION} package not built at the default root",
)

SECTIONS = [
    "What is in the package",
    "The route",
    "Street form",
    "Sun and shade on the walk dates",
    "Direct sun before each walk",
    "Wind: two regimes",
    "Ventilation for both regimes",
    "Flagged points",
    "Using the data with temperature readings",
    "Street measures and the walk temperature readings",
    "References",
]
FORBIDDEN = ["—", "–", "SBGL", "METAR", "H/W", "λ", "z0", "SVF", "LiDAR", r"\btree", "v0.1", "v0.2",
             "v1.", "novel", "robust", "significant", "Read with care", "sun_envelope.png",
             "items the team asked", "What we need", "later version", "future version"]


@pytest.fixture(scope="module")
def facts() -> dict:
    return compute_facts(PACKAGE_DIR)


@pytest.fixture(scope="module")
def report_md() -> str:
    return render_report_markdown(PACKAGE_DIR)


@pytest.fixture(scope="module")
def readme_md() -> str:
    return render_readme(PACKAGE_DIR)


def _prose(readme: str) -> str:
    """README without the generated column list (dictionary text, checked at its source)."""
    head, rest = readme.split("## Columns", 1)
    return head + "## Manifest" + rest.split("## Manifest", 1)[1]


def _linked_copy(dest: Path) -> Path:
    """The package as symlinks, so one file can be swapped without copying the big tables."""
    dest.mkdir()
    for f in PACKAGE_DIR.iterdir():
        (dest / f.name).symlink_to(f)
    return dest


def _swap_manifest(pkg: Path, edit) -> None:
    manifest = json.loads((PACKAGE_DIR / "manifest.json").read_text(encoding="utf-8"))
    edit(manifest)
    (pkg / "manifest.json").unlink()
    (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_sections_in_r6_order(report_md):
    assert report_md.startswith("::: {.titleblock}\n# ")
    assert re.findall(r"(?m)^## (.+)$", report_md) == SECTIONS


def test_opening_paragraph_names_study_author_and_project(report_md):
    opening = report_md.split("\n## ", 1)[0]
    for s in (STUDY_TITLE, AUTHOR, PROJECT_FORM, "first look at pairing", "Rio local time"):
        assert s in opening, s


def test_project_named_only_in_parenthetical_form(report_md, readme_md):
    for text in (report_md, readme_md):
        assert not re.search(r"(?<!Brisa\+ \()MorphoFavela", text)


@pytest.mark.parametrize("token", FORBIDDEN)
def test_no_forbidden_strings(report_md, readme_md, token):
    pat = token if token.startswith("\\b") else re.escape(token)
    assert not re.search(pat, re.sub(r"\(OM2/fig_\w+\.png\)", "", report_md)), token
    assert not re.search(pat, _prose(readme_md)), token


def test_every_figure_embedded_once_in_order(report_md):
    hits = re.findall(r"!\[Figure (\d+)\. [^\]]+\]\(OM2/(fig_\w+\.png)\)", report_md)
    assert [name for _n, name in hits] == [name for name in FIGURES]
    assert [int(n) for n, _name in hits] == list(range(1, len(FIGURES) + 1))


def test_each_figure_cited_before_it_appears(report_md):
    for i, name in enumerate(FIGURES, start=1):
        image = report_md.index(f"](OM2/{name})")
        assert re.search(rf"Figure {i}\b(?!\.)", report_md[:image]), name


def test_numbers_come_from_the_package(report_md, facts):
    walks = f"{facts['n_walks']} times on {facts['n_dates']} dates"
    assert walks in report_md
    assert f"{facts['n_flagged']:,} of the {facts['n_points']:,} points" in report_md
    assert f"{facts['n_partial']} of the {facts['n_walks']} walks" in report_md
    camp = facts["regimes"]["campaign"]
    for g in camp.values():
        assert f"mean direction of {g['dir']:.0f}°" in report_md
        assert g["name"] in report_md
    assert f"τ = t90 / {facts['ln10']:.3f}" in report_md


def test_height_to_width_names_its_statistic(report_md, facts):
    assert f"median of the point height-to-width ratios is {facts['hw_median_of_ratios']:.1f}" in report_md


def test_no_direct_sun_statement_uses_the_dose_figure_cells(report_md, facts):
    assert f"Counted over the {int(facts['dose_bin_m'])} m stretches of each walk, as the figure draws them" in report_md
    assert f"{100 * facts['dose_cells_zero_1h']:.0f}% of walk stretches got no direct sun" in report_md
    assert f"{100 * facts['dose_rows_zero_1h']:.0f}% of walk points got no direct sun" in report_md


def test_wind_section_states_regimes_tags_and_broad_arc(report_md, facts):
    wind = report_md.split("## Wind: two regimes", 1)[1].split("\n## ", 1)[0]
    assert "16-sector wind rose" in wind
    assert "broad northern arc" in wind and "confirms the east-southeast direction" in wind
    for name, n in facts["walk_tags"].items():
        if name != "none":
            assert f"{n} walks {name}" in wind


def test_r8_statements_present(report_md):
    assert "assumes a clear sky, so it is an upper bound" in report_md
    assert report_md.count("2019 building and terrain geometry") >= 3


def test_one_question_for_the_team(report_md):
    q = report_md.split("**One question for the team.**", 1)[1].split("\n", 1)[0]
    assert "time constant" in q and "housing" in q and "63%" in q and "90%" in q


def test_contact_line(report_md):
    assert report_md.rstrip().endswith(f"{AUTHOR}, {PROJECT_FORM}.")


def test_percentage_collisions_are_detected():
    p = _Pcts()
    p("a", 0.501)
    p("b", 0.499)
    with pytest.raises(ValueError, match="round alike"):
        p.check()


def test_file_table_lists_every_shipped_data_file(report_md):
    table = report_md.split("## What is in the package", 1)[1].split("\n\n", 2)[1]
    for f in PACKAGE_DIR.rglob("*"):
        rel = f.relative_to(PACKAGE_DIR).as_posix()
        if f.is_file() and not rel.endswith(".png") and not rel.startswith(("README", "report", "_")) \
                and rel != "OM2/figure_facts.json":
            assert f"`{rel.rsplit('.', 1)[0]}" in table or f"`{rel}`" in table, rel


def test_unknown_shipped_file_stops_the_render(tmp_path):
    pkg = _linked_copy(tmp_path / "pkg")
    (pkg / "p99_extra.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="p99_extra"):
        render_report_markdown(pkg)


def test_planted_manifest_mismatch_stops_the_build(tmp_path):
    pkg = _linked_copy(tmp_path / "pkg")
    _swap_manifest(pkg, lambda m: m["walks"].__setitem__("n_partial", m["walks"]["n_partial"] + 1))
    with pytest.raises(ValueError, match="n_partial"):
        render_report_markdown(pkg)


def test_planted_route_length_changes_both_documents(tmp_path):
    pkg = _linked_copy(tmp_path / "pkg")
    _swap_manifest(pkg, lambda m: m["routes"][0].__setitem__("length_m", 98765.0))
    assert "over 98,765 m" in render_report_markdown(pkg)
    assert "OM2 route: 98,765 m" in render_readme(pkg)


def test_readme_and_report_agree_on_shared_numbers(report_md, readme_md, facts):
    shared = [
        f"{facts['n_points']:,} points, one every {facts['spacing_m']:g} m over {facts['length_m']:,.0f} m",
        f"{facts['n_dates']} walk dates",
        f"τ = t90 / {facts['ln10']:.3f}",
        f"less than {100 * facts['partial_coverage']:.0f}% of the route",
    ]
    for s in shared:
        assert s in report_md and s in readme_md, s
    assert f"{facts['n_walks']} walks on {facts['n_dates']} dates" in readme_md
    assert f"{facts['n_walks']} times on {facts['n_dates']} dates" in report_md
    for g in facts["regimes"]["campaign"].values():
        assert f"{g['dir']:.0f}°" in report_md and f"{g['dir']:.0f}°" in readme_md


def test_readme_has_spec_headings_and_method_parts(readme_md):
    for h in ("## Sources and dates", "## CRS", "## Methods", "## Known limits", "## Use terms",
              "## How to cite", "## Columns"):
        assert h in readme_md, h
    for s in ("Cassiano and Vincent", "SHA-256", "running maximum", "gap_interpolated", "`partial`",
              "von Mises", "uniform background", "Macdonald et al. (1998)", "regular arrays of blocks",
              "--by walk_id --segment-m 20 --tau 30", "exp(-Δt/τ)"):
        assert s in readme_md, s
    assert USE_TERMS in readme_md


def test_readme_lists_every_column_of_every_data_table(readme_md):
    import pandas as pd

    cols = readme_md.split("## Columns", 1)[1].split("## Manifest", 1)[0]
    for col in pd.read_parquet(PACKAGE_DIR / "p02b_walks.parquet").columns:
        assert f"`{col}`" in cols, col
    for col in pd.read_parquet(PACKAGE_DIR / "OM2" / "points.parquet").columns:
        assert f"`{col}`" in cols, col


def test_write_report_renders_a_pdf(tmp_path):
    pkg = _linked_copy(tmp_path / "pkg")
    for name in ("report.md", "report.pdf"):
        (pkg / name).unlink(missing_ok=True)
    md, pdf = write_report(pkg)
    assert not md.is_symlink() and pdf.read_bytes().startswith(b"%PDF")
