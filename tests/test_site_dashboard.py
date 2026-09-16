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
