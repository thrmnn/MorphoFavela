"""Invariants for src/brisa_solar/mare_study_area.py — the study-area /
data-extent split MAREBOUND introduces (docs/research/mare_neighbourhood_sources.md,
data/maré/neighbourhoods_provenance.json).

Skips cleanly where data/maré/{raw/mare_boundary.shp,neighbourhoods.gpkg}
are absent (gitignored; a plain worktree checkout has neither).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.brisa_solar import mare_study_area as msa  # noqa: E402

DATA_PRESENT = msa.BAIRRO_SHP.exists() and msa.NEIGHBOURHOODS_GPKG.exists()
pytestmark = pytest.mark.skipif(not DATA_PRESENT, reason="data/maré/{raw/mare_boundary.shp,neighbourhoods.gpkg} absent on this checkout")


@pytest.fixture(scope="module")
def sa():
    return msa.load_study_area()


def test_sixteen_communities_split_fifteen_in_one_out(sa):
    assert len(sa["communities"]) == 16
    assert len(sa["included"]) == 15
    assert len(sa["excluded"]) == 1


def test_marcilio_dias_is_excluded_by_geometry_not_name(sa):
    """The exclusion rule is a geometry threshold (share of the community's
    own area inside the ACTIVE study area — the IPP Territórios Sociais
    outline since 2026-09-24, PI ruling); it happens to land on Marcílio
    Dias because that community measures ~0% inside the outline, not
    because the code singles it out by name. Proven here by re-deriving the
    split from raw share values (against the study area, not the data
    extent — the two coincide for Marcílio Dias, but for the right reason:
    verified independently) and checking it matches, independent of any
    name."""
    comm = sa["communities"]
    share = comm.geometry.intersection(sa["study_area"]).area / comm.geometry.area
    expected_excluded = set(comm.loc[share.to_numpy() < msa.INSIDE_EXTENT_MIN_SHARE, "community"])
    assert expected_excluded == set(sa["excluded"]["community"])
    assert "Marcílio Dias" in expected_excluded
    # and it is a clean separation, not a threshold that happens to bite once
    included_shares = share[comm["community"].isin(sa["included"]["community"])]
    assert (included_shares >= 0.85).all()
    excluded_shares = share[comm["community"].isin(sa["excluded"]["community"])]
    assert (excluded_shares < 0.01).all()


def test_study_area_extends_slightly_beyond_data_extent(sa):
    # Since the 2026-09-24 promotion, study_area is the IPP Territórios
    # Sociais outline — an independently digitized source from the bairro
    # polygon used as the data extent, no longer built by intersecting with
    # it (contrast the retired union-of-communities definition, which by
    # construction WAS a subset). A small sliver falls outside; it must
    # stay small (source-mismatch slop), never balloon into real drift.
    outside = sa["study_area"].difference(sa["data_extent"]).area
    assert 0 < outside < 20_000  # m² — observed ~13,400 m² at promotion time
    assert outside / sa["study_area"].area < 0.01


def test_study_area_smaller_than_whole_bairro(sa):
    # The outline covers most of the bairro (unlike the retired
    # union-of-communities definition, which covered under half of it) but
    # still excludes the bairro's fringes outside the outline.
    assert 0.5 * sa["data_extent"].area < sa["study_area"].area < 0.95 * sa["data_extent"].area


def test_excluded_community_contributes_nothing_to_study_area(sa):
    marcilio = sa["excluded"][sa["excluded"]["community"] == "Marcílio Dias"]
    assert len(marcilio) == 1
    overlap = marcilio.geometry.iloc[0].intersection(sa["study_area"]).area
    assert overlap < 1.0


def test_within_mask_matches_shapely_contains(sa):
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = sa["data_extent"].bounds
    xs = rng.uniform(x0, x1, 500)
    ys = rng.uniform(y0, y1, 500)
    mask = msa.within_mask(xs, ys, sa["study_area"])
    import shapely
    expected = shapely.contains_xy(sa["study_area"], xs, ys)
    assert (mask == expected).all()


def test_within_mask_handles_empty_and_all_outside():
    import shapely
    geom = shapely.geometry.box(0, 0, 10, 10)
    assert msa.within_mask(np.array([]), np.array([]), geom).shape == (0,)
    mask = msa.within_mask(np.array([100.0, 200.0]), np.array([100.0, 200.0]), geom)
    assert not mask.any()
