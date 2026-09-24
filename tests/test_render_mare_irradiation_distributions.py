"""Pure-function tests for the Maré distributions module (FOLHA4). No real
pipeline outputs are read — compute_distributions/draw_distributions need
the WP-05 parquet + Maré data files and are exercised instead by
scripts/build_site_dashboard.py --folha4-mare (a real-data smoke run, not a
fast unit test)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_mare_irradiation_distributions as mid  # noqa: E402


def test_distributions_rc_disables_top_and_right_spines():
    # These three panels' own local style, applied via rc_context by every
    # caller (this module's main() and build_site_dashboard.py's
    # build_folha4_mare) — never a global rcParams mutation that would leak
    # into a neighbouring panel's style.
    assert mid.DISTRIBUTIONS_RC["axes.spines.top"] is False
    assert mid.DISTRIBUTIONS_RC["axes.spines.right"] is False


def test_provenance_note_states_run_of_record_and_release_status():
    note = mid.provenance_note({"run_of_record": "wp05_full_20260101T000000Z"})
    assert "wp05_full_20260101T000000Z" in note
    assert "ethics-gate" in note
    assert "PI review" in note


def test_provenance_note_glosses_the_stats_terms_the_panels_use():
    # round-2/3 council finding: decile/IQR/length-weighted appear on the
    # panels with no other gloss anywhere on a print page (no hover the
    # way the interactive twin's <dfn> tooltips have) — this note is where
    # FOLHA4 puts the plain-language definitions.
    note = mid.provenance_note({"run_of_record": "wp05_full_20260101T000000Z"})
    assert "decile" in note
    assert "interquartile range" in note
    assert "length-weighted" in note


def test_draw_distributions_top_and_bottom_are_split_out():
    # FOLHA4's hero/no-hero variants call these directly (not the combined
    # draw_distributions wrapper) so the host sheet can place its own
    # fixed-ratio spacer between them — see build_site_dashboard.py.
    assert callable(mid.draw_distributions_top)
    assert callable(mid.draw_distributions_bottom)


def test_between_label_is_the_module_constant_not_a_retyped_string():
    assert mid.BETWEEN == "between communities"
