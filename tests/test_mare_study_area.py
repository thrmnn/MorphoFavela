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
    own area inside the data extent); it happens to land on Marcílio Dias
    because that community measures ~0% inside the bairro, not because the
    code singles it out by name. Proven here by re-deriving the split from
    raw share values and checking it matches, independent of any name."""
    comm = sa["communities"]
    share = comm.geometry.intersection(sa["data_extent"]).area / comm.geometry.area
    expected_excluded = set(comm.loc[share.to_numpy() < msa.INSIDE_EXTENT_MIN_SHARE, "community"])
    assert expected_excluded == set(sa["excluded"]["community"])
    assert "Marcílio Dias" in expected_excluded
    # and it is a clean separation, not a threshold that happens to bite once
    included_shares = share[comm["community"].isin(sa["included"]["community"])]
    assert (included_shares >= 0.99).all()
    excluded_shares = share[comm["community"].isin(sa["excluded"]["community"])]
    assert (excluded_shares < 0.01).all()


def test_study_area_is_subset_of_data_extent(sa):
    # study_area = included communities ∩ data_extent, so it can never
    # extend beyond the data extent (unlike the raw union of all 16, which
    # bleeds ~51,000 m² north around Marcílio Dias — see neighbourhoods_provenance.json).
    assert sa["study_area"].difference(sa["data_extent"]).area < 1.0  # numerical slop only


def test_study_area_smaller_than_whole_bairro(sa):
    # The pre-MAREBOUND definition of "Maré" was the whole bairro polygon;
    # the 16-community union covers under half of it (open ground, canals,
    # roads, and non-community land inside the bairro are not part of any
    # of the 16 communities).
    assert sa["study_area"].area < 0.6 * sa["data_extent"].area


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
