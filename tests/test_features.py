"""WS-0 feature substrate — the change-of-support logic.

Guards the rules in docs/morpho_signature_plan.md: fabric excludes the naive
street-aggregated SVF; street→cell summaries are distributional + support-aware;
empty cells are NULL, never interpolated; areal→point enrichment carries the
cell's fabric onto each observer.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from src.morphometry.features import (
    build_experience_table,
    build_fabric_table,
    summarize_streets_to_cells,
)

CRS = "EPSG:31983"


def _grid():
    # cell A = x[0,10], cell B = x[10,20]; B will get no observers
    return gpd.GeoDataFrame(
        {
            "zone_id": ["A", "B"],
            "zone_area": [100.0, 100.0],
            "lambda_p": [0.6, 0.0],
            "far": [1.2, 0.0],
            "sigma_h": [2.0, 0.0],
            "H_mean": [9.0, 0.0],
            "building_count": [4, 0],
            "lambda_f_mean": [0.4, 0.1],
            "porosity": [0.3, 0.9],
            "slope_deg": [15.0, 5.0],
            "aspect_deg": [0.0, 180.0],
            "street_orientation_entropy": [0.5, 0.5],
            "svf": [0.4, 0.8],        # naive aggregate — must NOT survive into fabric
            "svf_count": [3, 0],
            "geometry": [box(0, 0, 10, 10), box(10, 0, 20, 10)],
        },
        crs=CRS,
    )


def _svf_solar():
    pts = [Point(2, 2), Point(4, 4), Point(6, 6)]  # all inside cell A
    return gpd.GeoDataFrame(
        {
            "svf": [0.2, 0.4, 0.6],
            "solar_hours_winter": [1.0, 3.0, 5.0],
            "geometry": pts,
        },
        crs=CRS,
    )


def _openness(svf_solar):
    g = svf_solar[["geometry"]].copy()
    g["openness_class"] = ["DEEP_CANYON", "OPEN", "INTERMEDIATE"]
    return g


def test_fabric_excludes_naive_svf_and_encodes_aspect():
    fab = build_fabric_table(_grid())
    assert "svf" not in fab.columns and "svf_count" not in fab.columns
    assert {"northness", "eastness", "built_mask"} <= set(fab.columns)
    # aspect 0° → northness 1; aspect 180° → northness -1
    assert np.isclose(fab.loc[fab.zone_id == "A", "northness"].iat[0], 1.0)
    assert np.isclose(fab.loc[fab.zone_id == "B", "northness"].iat[0], -1.0)
    assert fab.loc[fab.zone_id == "A", "built_mask"].iat[0]
    assert not fab.loc[fab.zone_id == "B", "built_mask"].iat[0]


def test_experience_table_carries_cell_fabric_arealtopoint():
    svf = _svf_solar()
    exp = build_experience_table(svf, _grid(), openness=_openness(svf), hw=None)
    assert (exp["zone_id"] == "A").all()                 # all observers in cell A
    assert np.allclose(exp["lambda_p"], 0.6)             # carried A's fabric
    assert list(exp["openness_class"]) == ["DEEP_CANYON", "OPEN", "INTERMEDIATE"]
    assert (~exp["has_hw"]).all()                        # no canyon layer supplied


def test_summary_is_support_aware_and_distributional():
    grid = _grid()
    svf = _svf_solar()
    exp = build_experience_table(svf, grid, openness=_openness(svf), hw=None)
    summ = summarize_streets_to_cells(exp, grid).set_index("zone_id")

    a = summ.loc["A"]
    assert a["n_street_obs"] == 3 and bool(a["has_street_support"])
    assert np.isclose(a["solar_winter_p50"], 3.0)
    assert np.isclose(a["solar_winter_frac_below2h"], 1 / 3)   # only the 1.0 h point
    assert np.isclose(a["frac_deep_canyon"], 1 / 3)
    assert a["svf_p10"] < a["svf_p50"]                          # worst-decile below median


def test_empty_cell_is_null_not_interpolated():
    grid = _grid()
    svf = _svf_solar()
    exp = build_experience_table(svf, grid, openness=_openness(svf), hw=None)
    summ = summarize_streets_to_cells(exp, grid).set_index("zone_id")

    b = summ.loc["B"]
    assert b["n_street_obs"] == 0 and not bool(b["has_street_support"])
    assert pd.isna(b["svf_p50"]) and pd.isna(b["solar_winter_p50"])
    # every grid cell is represented — none silently dropped
    assert set(summ.index) == {"A", "B"}
