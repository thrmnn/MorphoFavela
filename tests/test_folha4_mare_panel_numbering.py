"""FOLHA4 Maré round 5, F1: the sheet drew masthead -> hero map ->
choropleth -> distributions -> box plots on the page, but the titles
printed '2 ·' (choropleth) above '1 ·' (whole distributions) — the two
were numbered as if reading order and draw order disagreed. This is the
one place the fix is asserted: `folha4_mare_panel_numbers` is the single
source build_site_dashboard.py's build_folha4_mare() reads to number
every panel, so a test on it alone proves the whole sheet's numbering is
1..N in reading order, both with and without the hero map row.

No real pipeline outputs needed — `folha4_mare_panel_numbers` is a pure
function of one bool. Panel titles actually carrying these numbers is
covered separately (both spatial panels' titles in build_site_dashboard.py
and the two distributions titles in
tests/test_render_mare_irradiation_distributions.py's own panel_num
tests), so this file only has to prove the numbers themselves are
1..4/1..3 in reading order, never re-derive the title strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_site_dashboard as bsd  # noqa: E402


def test_hero_variant_numbers_panels_1_to_4_in_reading_order():
    nums = bsd.folha4_mare_panel_numbers(hero=True)
    # Reading order on the page: masthead/identity card (unnumbered) ->
    # hero map -> choropleth -> whole distributions -> per-community
    # spread — exactly the draw order build_folha4_mare() uses.
    assert nums == {
        "hero_map": 1,
        "community_choropleth": 2,
        "distributions_top": 3,
        "distributions_bottom": 4,
    }


def test_no_hero_variant_numbers_panels_1_to_3_in_reading_order():
    # No hero-map row on this variant — the choropleth is the first
    # numbered panel on the page, so every later panel shifts down by one
    # relative to the hero variant.
    nums = bsd.folha4_mare_panel_numbers(hero=False)
    assert nums == {
        "community_choropleth": 1,
        "distributions_top": 2,
        "distributions_bottom": 3,
    }


def test_panel_numbers_are_contiguous_starting_at_1_in_draw_order():
    # Generic shape check that would catch a future edit introducing a
    # gap or a duplicate: whatever panels a variant carries, their
    # numbers are exactly 1..N once sorted by draw order (dict insertion
    # order here IS draw order — both branches build the dict that way).
    for hero in (True, False):
        nums = bsd.folha4_mare_panel_numbers(hero)
        ordered = list(nums.values())
        assert ordered == list(range(1, len(ordered) + 1))
