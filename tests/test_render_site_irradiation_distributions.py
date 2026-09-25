"""Pure-function tests for the generic (non-Maré) distributions module
(FOLHA4, cyc4b/folha4-all). No real pipeline outputs are read —
compute_distributions/draw_distributions_* need the WP-05 parquet + each
site's data files and are exercised instead by
scripts/build_site_dashboard.py --folha4-all (a real-data smoke run, not a
fast unit test)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import render_site_irradiation_distributions as sid  # noqa: E402


def test_distributions_rc_disables_top_and_right_spines():
    assert sid.DISTRIBUTIONS_RC["axes.spines.top"] is False
    assert sid.DISTRIBUTIONS_RC["axes.spines.right"] is False


def test_between_label_matches_the_shared_territory_constant():
    # Reused from src.sites.territory.BETWEEN_SUBUNITS_LABEL, not retyped —
    # a drift here would silently split panel 3's "between communities"
    # box away from label_subunits' own output.
    from src.sites.territory import BETWEEN_SUBUNITS_LABEL
    assert sid.BETWEEN == BETWEEN_SUBUNITS_LABEL


def test_draw_distributions_bottom_returns_none_without_subunits():
    # A site with no subunits (order=None) gets no panel 3 at all — the
    # caller (build_folha4_site) decides whether that means "don't route
    # this site through v4", not this function.
    data = dict(order=None)
    assert sid.draw_distributions_bottom(fig=None, spec=None, data=data) is None
