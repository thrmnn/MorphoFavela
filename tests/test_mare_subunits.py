"""Invariants for docs/briefs/mare/mare_subunits.py (per-neighbourhood Maré
morphology + the Maré-internal fabric clustering).

Skips cleanly when the real outputs/ or runs/ tree is absent (gitignored;
a plain worktree checkout has neither).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
# Hardcoded to the main checkout, not `ROOT`: data/, outputs/, and runs/ are
# gitignored, so a worktree checkout (this file may run from one) has none
# of them — same pattern tests/test_mare_study_area.py's msa.ROOT and
# tests/test_mare_brief.py's OUTPUTS_ROOT use.
MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")
OUTPUTS_ROOT = MAIN_CHECKOUT / "outputs"
RUNS_ROOT = MAIN_CHECKOUT / "runs"

os.environ["MORPHOFAVELA_ROOT"] = str(MAIN_CHECKOUT)
sys.path.insert(0, str(ROOT / "docs" / "briefs" / "mare"))
sys.path.insert(0, str(ROOT))

import mare_subunits  # noqa: E402
from src.sites.territory import BETWEEN_SUBUNITS_LABEL  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (OUTPUTS_ROOT.exists() and RUNS_ROOT.exists()),
    reason="outputs/ or runs/ absent on this checkout",
)


@pytest.fixture(scope="module")
def table():
    try:
        return mare_subunits.build_subunit_table(OUTPUTS_ROOT, RUNS_ROOT)
    except mare_subunits.MissingSource as e:
        pytest.skip(f"missing source: {e}")


def test_fifteen_communities_plus_between(table):
    names = set(table["name"])
    assert BETWEEN_SUBUNITS_LABEL in names
    assert len(names) == 16
    assert "Marcílio Dias" not in names  # excluded from the study area (PI ruling 2026-09-24)


def test_row_order_is_north_to_south(table):
    ys = table["mean_y"].to_numpy()
    assert np.all(np.diff(ys) <= 0), "rows must be sorted north (higher y) to south (lower y)"


def test_no_duplicate_communities(table):
    assert table["name"].is_unique


def test_cell_counts_are_positive(table):
    assert (table["n_cells"] > 0).all()
    assert (table["n_built_cells"] >= 0).all()
    assert (table["n_built_cells"] <= table["n_cells"]).all()


def test_solar_columns_present_and_finite_where_covered(table):
    covered = table["n_solar_points"].fillna(0) > 0
    assert covered.any(), "at least one community should have WP-04 solar coverage"
    for col in ("sun_winter_median_h", "kwh_m2_median"):
        vals = table.loc[covered, col]
        assert vals.notna().all()
        assert (vals >= 0).all()


@pytest.fixture(scope="module")
def clusters():
    try:
        return mare_subunits.fit_within_mare_clusters(OUTPUTS_ROOT, krange=range(2, 5))
    except mare_subunits.MissingSource as e:
        pytest.skip(f"missing source: {e}")


def test_within_mare_k_is_in_tested_range(clusters):
    assert clusters["k_range"][0] <= clusters["k_selected"] <= clusters["k_range"][1]


def test_within_mare_shares_sum_to_roughly_one_hundred(clusters):
    total = sum(clusters["shares_pct"].values())
    assert abs(total - 100.0) < 0.5


def test_within_mare_reports_bic_honesty_fields(clusters):
    assert "bic_monotonic" in clusters
    assert "bic_argmin_k" in clusters
    assert isinstance(clusters["bic_monotonic"], bool)


def test_bic_elbow_helper_on_a_synthetic_monotonic_curve():
    import pandas as pd

    bic = pd.DataFrame({"k": [2, 3, 4, 5, 6], "bic": [0.0, -1.0, -1.5, -1.7, -1.75]})
    k = mare_subunits._bic_elbow_k(bic)
    assert 2 <= k <= 6
