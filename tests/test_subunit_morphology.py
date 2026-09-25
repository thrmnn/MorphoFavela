"""Invariants for src/sites/subunit_morphology.py (site-agnostic per-subunit
morphology, generalised from docs/briefs/mare/mare_subunits.py) and its CLI,
scripts/build_subunit_morphology.py.

Needs the main checkout's data/, outputs/ and runs/ (all gitignored): a
plain worktree checkout has none of these, so the whole module skips
cleanly there, same pattern as tests/test_mare_subunits.py and
tests/test_site_territory.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sites.subunit_morphology import MissingSource, build_subunit_table  # noqa: E402
from src.sites.territory import BETWEEN_SUBUNITS_LABEL  # noqa: E402
from scripts.build_subunit_morphology import SKIP, eligible_sites  # noqa: E402

MAIN_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
OUTPUTS_ROOT = MAIN_ROOT / "outputs"
RUNS_ROOT = MAIN_ROOT / "runs"

pytestmark = pytest.mark.skipif(
    not (OUTPUTS_ROOT.exists() and RUNS_ROOT.exists()),
    reason="outputs/ or runs/ absent on this checkout",
)

SUBUNIT_SITES = ["complexo_do_alemao", "riodaspedras"]
NO_SUBUNIT_SITES = ["vidigal", "rocinha"]


def test_maré_is_skipped_not_generalised():
    assert "maré" in SKIP


def test_no_subunit_sites_are_skipped_by_config():
    from src.sites.territory import load_sites_config
    _, skipped = eligible_sites(load_sites_config())
    for site in NO_SUBUNIT_SITES:
        assert site in skipped


@pytest.fixture(scope="module", params=SUBUNIT_SITES)
def table(request):
    try:
        return build_subunit_table(request.param, OUTPUTS_ROOT, RUNS_ROOT, root=MAIN_ROOT)
    except MissingSource as e:
        pytest.skip(f"missing source: {e}")


def test_row_order_is_north_to_south(table):
    ys = table["mean_y"].to_numpy()
    assert np.all(np.diff(ys) <= 0), "rows must be sorted north (higher y) to south (lower y)"


def test_no_duplicate_subunits(table):
    assert table["name"].is_unique


def test_cell_counts_are_positive(table):
    assert (table["n_cells"] > 0).all()
    assert (table["n_built_cells"] >= 0).all()
    assert (table["n_built_cells"] <= table["n_cells"]).all()


def test_solar_columns_present_and_finite_where_covered(table):
    covered = table["n_solar_points"].fillna(0) > 0
    assert covered.any(), "at least one subunit should have WP-04 solar coverage"
    for col in ("sun_winter_median_h", "kwh_m2_median"):
        vals = table.loc[covered, col]
        assert vals.notna().all()
        assert (vals >= 0).all()


def test_no_subunit_sites_raise_missing_source():
    for site in NO_SUBUNIT_SITES:
        with pytest.raises(MissingSource):
            build_subunit_table(site, OUTPUTS_ROOT, RUNS_ROOT, root=MAIN_ROOT)


def test_between_subunits_label_mechanism_available():
    # Not asserted present for every site (complexo_do_alemao/riodaspedras'
    # subunit polygons happen to tile their study area exactly — see the
    # site's own summary.json `has_between_subunits`), but the label itself
    # must be the shared constant, never a locally retyped string.
    assert BETWEEN_SUBUNITS_LABEL == "between communities"
