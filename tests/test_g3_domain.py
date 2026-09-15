"""G3 / HD-02 acceptance: spec docs/g3_domain_sensitivity_spec.md, "Tests".

Real DTM/footprints/EPW/wp05_full.parquet/distribution.json live only in the
main checkout (data/ and runs/ are gitignored, never copied into this
worktree) — same convention as tests/test_wp05_full.py. Tests 1 and 2 use
synthetic arrays with a known answer; test 3 needs the real WP-05 base run
and is skipped if that run isn't present on this machine (it reads the run
by its absolute, main-checkout path, not a worktree-relative one).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.brisa_solar import g3_domain, wp05_full

BASE_RUN_DIR = Path("/home/theo/SCL/SCR/MorphoFavela/runs/wp05_full_20260914T215419Z")


# ---------------------------------------------------------------------------
# 1. Subset variants are exact subsets of the 0.10/10 frame (cell ids).
# ---------------------------------------------------------------------------

def test_tighter_variants_are_exact_subsets_of_the_base_frame():
    rng = np.random.default_rng(42)
    shape = (80, 80)
    not_building_and_valid = rng.random(shape) > 0.1
    coverage = rng.random(shape).astype("float32")
    dist_m = (rng.random(shape) * 30.0).astype("float32")

    base_mask = g3_domain.variant_mask(not_building_and_valid, coverage, dist_m, 0.10, 10.0)
    base_cells = set(zip(*np.where(base_mask)))
    assert len(base_cells) > 0, "synthetic base frame must be non-empty for this test to mean anything"

    # Every grid cell with threshold >= base and distance <= base is, by
    # construction (variant_mask is an AND of two monotone conditions), a
    # subset of the base frame's cell set.
    for threshold in g3_domain.FABRIC_COVERAGE_GRID:
        for distance_m in g3_domain.FABRIC_FOOTPRINT_DISTANCE_GRID_M:
            if threshold >= 0.10 and distance_m <= 10.0:
                variant = g3_domain.variant_mask(not_building_and_valid, coverage, dist_m, threshold, distance_m)
                variant_cells = set(zip(*np.where(variant)))
                assert variant_cells <= base_cells, (
                    f"variant ({threshold}, {distance_m}) is not a subset of the base 0.10/10 frame"
                )

    # A looser combo (lower threshold OR larger distance) is NOT guaranteed a
    # subset — it must add at least the cells the base frame's own tighter
    # conditions excluded. Sanity check the inverse holds for the grid's
    # loosest cell against this synthetic array (it would be a trivial/wrong
    # test if the two masks were identical).
    loosest = g3_domain.variant_mask(not_building_and_valid, coverage, dist_m, 0.05, 20.0)
    assert set(zip(*np.where(loosest))) >= base_cells


def test_build_variant_frame_pulls_base_and_added_rows_correctly():
    # 3x3 raster; only (1,1) passes the base (t=0.10, d=10) rule; (0,0) only
    # passes a looser rule (lower coverage AND farther distance) — this is
    # exactly the "added cell" case build_added_observers/build_variant_frame
    # must route through the added_df branch, not the base_df branch.
    not_building_and_valid = np.ones((3, 3), dtype=bool)
    coverage = np.array([[0.06, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.0]], dtype="float32")
    dist_m = np.array([[18.0, 999.0, 999.0], [999.0, 4.0, 999.0], [999.0, 999.0, 999.0]], dtype="float32")
    arrays = {"not_building_and_valid": not_building_and_valid, "coverage": coverage, "dist_m": dist_m}

    base_df = pd.DataFrame({
        "row": [1], "col": [1], "favela_id": [0], "svf": [0.5], "kwh_m2": [1000.0],
    })
    added_df = pd.DataFrame({
        "row": [0], "col": [0], "favela_id": [0], "coverage": [0.06], "dist_m": [18.0],
        "svf": [0.8], "kwh_m2": [1500.0],
    })

    # Base-only variant (0.10, 10): only the base cell qualifies.
    variant_df, n_added = g3_domain.build_variant_frame(base_df, added_df, arrays, 0.10, 10.0)
    assert n_added == 0
    assert len(variant_df) == 1
    assert variant_df.iloc[0]["svf"] == 0.5

    # Looser variant (0.05, 20): both cells qualify, the added one via the
    # added_df branch.
    variant_df, n_added = g3_domain.build_variant_frame(base_df, added_df, arrays, 0.05, 20.0)
    assert n_added == 1
    assert len(variant_df) == 2
    assert set(variant_df["svf"]) == {0.5, 0.8}


# ---------------------------------------------------------------------------
# 2. Percentile-of-median is computed on the variant's own distribution
#    (synthetic check with a known answer).
# ---------------------------------------------------------------------------

def test_percentile_of_median_known_answer():
    citywide = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    favela = np.array([28.0, 30.0, 32.0])  # median == 30.0, exactly the citywide array's own p50

    median, pct = g3_domain.favela_percentile_of_median(favela, citywide)

    assert median == 30.0
    # scipy percentileofscore(kind="mean"): (count_strictly_below + count_below_or_equal) / (2n) * 100
    # = (2 + 3) / (2*5) * 100 = 50.0 exactly, for this hand-picked array.
    assert pct == pytest.approx(50.0)


def test_percentile_of_median_empty_inputs_return_none_not_a_fabricated_number():
    median, pct = g3_domain.favela_percentile_of_median(np.array([]), np.array([1.0, 2.0, 3.0]))
    assert median is None and pct is None

    median, pct = g3_domain.favela_percentile_of_median(np.array([1.0]), np.array([]))
    assert median is None and pct is None


# ---------------------------------------------------------------------------
# 3. The WP-05 frame variant (threshold=0.10, distance=10) reproduces
#    distribution.json's five favela percentiles to 0.1 point.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not BASE_RUN_DIR.exists(), reason="WP-05 base run not present on this machine (data/runs are gitignored)")
def test_wp05_frame_variant_reproduces_distribution_json_percentiles():
    import geopandas as gpd

    distribution = json.loads((BASE_RUN_DIR / "distribution.json").read_text())
    study_favelas = distribution["study_favelas"]

    consolidated = pd.read_parquet(
        BASE_RUN_DIR / "wp05_full.parquet", columns=["favela_id", "svf", "kwh_m2"],
    )
    citywide_svf = consolidated["svf"].to_numpy()
    citywide_kwh = consolidated["kwh_m2"].to_numpy()

    favelas_gdf = gpd.read_file("/home/theo/SCL/SCR/MorphoFavela/data/RJ/Favelas_Limit_2019.shp")

    for name in wp05_full.STUDY_FAVELAS:
        matched, _method = wp05_full.match_favela_group(favelas_gdf, name)
        cod_ids = matched["cod_favela"].astype(int).tolist()
        sub = consolidated[consolidated["favela_id"].isin(cod_ids)]

        _svf_median, svf_pct = g3_domain.favela_percentile_of_median(sub["svf"].to_numpy(), citywide_svf)
        _kwh_median, kwh_pct = g3_domain.favela_percentile_of_median(sub["kwh_m2"].to_numpy(), citywide_kwh)

        expected_svf_pct = study_favelas[name]["svf"]["citywide_percentile_position"]
        expected_kwh_pct = study_favelas[name]["kwh_m2"]["citywide_percentile_position"]

        assert svf_pct == pytest.approx(expected_svf_pct, abs=0.1), f"{name} svf percentile drifted from distribution.json"
        assert kwh_pct == pytest.approx(expected_kwh_pct, abs=0.1), f"{name} kwh_m2 percentile drifted from distribution.json"
