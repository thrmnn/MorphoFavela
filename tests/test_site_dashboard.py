"""Pure-function tests for the Folha de Rua builders (task FOLHA refresh).

No real pipeline outputs are read — these check the per-site metadata
tables and CLI surface only. `scripts/build_site_dashboard.py` is
import-safe (its only module-level side effects are an rcParams update and
two lightweight package imports), so it is imported directly rather than
invoked as a subprocess for the metadata checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_site_dashboard as bsd  # noqa: E402

FIVE_SITES = {"vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"}


def test_sheet_number_covers_five_sites_and_is_01_to_05():
    assert set(bsd.SHEET_NUMBER) == FIVE_SITES
    assert sorted(bsd.SHEET_NUMBER.values()) == ["01", "02", "03", "04", "05"]
    assert bsd.SHEET_NUMBER["vidigal"] == "01"
    assert bsd.SHEET_NUMBER["maré"] == "05"


def test_typology_covers_five_sites():
    for site in FIVE_SITES:
        assert site in bsd.TYPOLOGY
        label, color = bsd.TYPOLOGY[site]
        assert isinstance(label, str) and label
        assert color.startswith("#")


def test_site_display_covers_five_sites():
    for site in FIVE_SITES:
        assert site in bsd.SITE_DISPLAY
        assert isinstance(bsd.SITE_DISPLAY[site], str) and bsd.SITE_DISPLAY[site]
    assert bsd.SITE_DISPLAY["maré"] == "Maré"
    assert bsd.SITE_DISPLAY["riodaspedras"] == "Rio das Pedras"


# ---------------------------------------------------------------------------
# WP04MARE: site_paths() picks up Maré's study_area-territory grid file when
# it exists, beside (never instead of) the canonical file every other site
# keeps reading, and falls back cleanly when it doesn't exist yet.
# ---------------------------------------------------------------------------

def test_site_paths_grid_prefers_mare_study_area_file_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(bsd, "ROOT", tmp_path)
    geom_dir = tmp_path / "outputs" / "maré" / "geometry_indicators"
    geom_dir.mkdir(parents=True)
    (geom_dir / "per_patch_geometry_study_area.csv").write_text("patch_id\n1\n")

    grid_path = bsd.site_paths("maré")["grid"]
    assert grid_path.name == "per_patch_geometry_study_area.csv"


def test_site_paths_grid_falls_back_to_canonical_file_when_study_area_file_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(bsd, "ROOT", tmp_path)
    # No outputs/maré/geometry_indicators/ at all -- site_paths must not error,
    # and must return the same default path pre-WP04MARE code did.
    grid_path = bsd.site_paths("maré")["grid"]
    assert grid_path.name == "per_patch_geometry.csv"


def test_site_paths_grid_unaffected_for_other_sites(tmp_path, monkeypatch):
    monkeypatch.setattr(bsd, "ROOT", tmp_path)
    for site in FIVE_SITES - {"maré"}:
        grid_path = bsd.site_paths(site)["grid"]
        assert grid_path.name == "per_patch_geometry.csv"


@pytest.mark.parametrize(
    "script",
    ["build_site_dashboard.py", "build_html_dashboard.py"],
)
def test_builder_accepts_all_flag(script):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--all" in proc.stdout


# ---------------------------------------------------------------------------
# FOLHA4 (2026-09-24): Maré's horizontal sheet, hero/no-hero variants.
# ---------------------------------------------------------------------------

def test_build_site_dashboard_accepts_folha4_mare_flag():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_site_dashboard.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--folha4-mare" in proc.stdout


class _FakePoint:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeGeom:
    def __init__(self, x, y):
        self._centroid = _FakePoint(x, y)

    @property
    def centroid(self):
        return self._centroid


class _FakeTerritory:
    def __init__(self, x, y):
        self.study_area = _FakeGeom(x, y)


def test_mare_rotation_origin_is_the_study_area_centroid_not_per_layer():
    # The bug this guards: rotate_for_display's default origin="center"
    # rotates each layer around ITS OWN bbox center, scattering boundary/
    # buildings/communities/observers relative to each other. Every FOLHA4
    # Maré layer must share exactly this one point.
    t = _FakeTerritory(123.5, -456.25)
    assert bsd._mare_rotation_origin(t) == (123.5, -456.25)


def test_uses_quadrant_fallback_true_without_tipo_logra_column():
    # No segments at all — Maré-shaped failure mode this helper exists for
    # (round-1 finding 3: an honest ridgeline title when tipo_logra is
    # unavailable and the panels fall back to compass quadrants).
    assert bsd._uses_quadrant_fallback({"seg": None}) is True


def test_uses_quadrant_fallback_true_when_column_missing(monkeypatch=None):
    class FakeSeg:
        columns = ["street_id", "svf_mean"]

    assert bsd._uses_quadrant_fallback({"seg": FakeSeg()}) is True


def test_uses_quadrant_fallback_false_when_tipo_logra_present():
    class FakeSeg:
        columns = ["street_id", "tipo_logra"]

    assert bsd._uses_quadrant_fallback({"seg": FakeSeg()}) is False


def test_js_state_never_writes_true_total_kpi_from_map_sample():
    """Guard for round-2 finding A: recomputeKPIs() in the generated
    js/state.js must never assign textContent to any element selected by
    data-key="n_observers" — that tile is the true dataset total rendered
    server-side from stats.json. state.observers is always the decimated
    map SAMPLE (write_observers_geojson's `target`, ~8,000 points), so a
    live count derived from it (maskObservers().length) belongs only on
    the separate, explicitly-labelled map-sample line
    (data-key="n_observers_sample" .sample-n), never on the true-total
    tile itself.
    """
    import re

    import build_html_dashboard as bhd  # noqa: PLC0415

    src = bhd.JS_STATE
    assert 'data-key="n_observers_sample"' in src, (
        "expected a distinct, JS-updated map-sample element separate from "
        "the true-total KPI tile"
    )
    # The exact bug: querySelectorAll on the *true-total* tile's selector
    # (n_observers, no _sample suffix) followed anywhere by an assignment
    # to .textContent. Matches on the executable selector string, not on
    # any surrounding prose/comments that happen to mention the key.
    bad_selector = re.compile(r'data-key="n_observers"\]')
    for line in src.splitlines():
        if bad_selector.search(line) and ".textContent" in line:
            raise AssertionError(
                "js/state.js assigns textContent to the true-total "
                "N-observers tile directly from the map sample: "
                f"{line.strip()!r}"
            )
    assert ".textContent = n.toLocaleString()" in src

