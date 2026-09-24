"""Tests for the Maré morphology, OM2 data package (v0.1).

Uses real Maré inputs at the default repo root (data/outputs are gitignored
and not present in a worktree checkout — these tests need to run from a
checkout that has them, same convention as this repo's other real-data
integration tests, e.g. tests/test_wp01_footprints.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.om_package.buffers import BUFFER_RADII_M, compute_buffer_variables
from src.om_package.dictionary import dictionary_dataframe, full_dictionary
from src.om_package.formvars import compute_form_variables
from src.om_package.io_utils import Paths
from src.om_package.quality import PENDING_ITEMS
from src.om_package.routes import POINT_SPACING_M, densify_route, stable_point_id
from src.om_package.segments import aggregate_to_segments
from src.om_package.shade import SHADE_TABLE_COLUMNS, build_empty_shade_table
from src.om_package.ventilation import compute_ventilation_proxies

PATHS = Paths()
pytestmark = pytest.mark.skipif(
    not PATHS.route_json("OM_2").exists(), reason="Maré/Octopus data not present at the default root"
)


@pytest.fixture(scope="module")
def om2_points():
    return densify_route(PATHS.route_json("OM_2"))


@pytest.fixture(scope="module")
def om2_points_small(om2_points):
    return om2_points.iloc[:60].reset_index(drop=True)


# --- P-02: point spacing -----------------------------------------------

def test_point_spacing_is_one_metre(om2_points):
    # spacing is defined along the route's arc length (distance_along_m),
    # not the straight-line chord between consecutive points — those two
    # coincide except at a sharp bend, where the chord is shorter than the
    # 1 m arc between them (real OM2 geometry has such bends).
    xy = np.column_stack([om2_points.geometry.x.to_numpy(), om2_points.geometry.y.to_numpy()])
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # a chord can never exceed the arc length between the same two points
    assert (step <= POINT_SPACING_M + 1e-6).all()


def test_distance_along_matches_spacing(om2_points):
    d = om2_points["distance_along_m"].to_numpy()
    diffs = np.diff(d)
    assert np.allclose(diffs[:-1], POINT_SPACING_M, atol=1e-6)


# --- P-02: stable IDs ----------------------------------------------------

def test_point_ids_stable_across_two_builds():
    a = densify_route(PATHS.route_json("OM_2"))
    b = densify_route(PATHS.route_json("OM_2"))
    assert list(a["point_id"]) == list(b["point_id"])


def test_point_id_deterministic_from_route_and_distance():
    assert stable_point_id("OM_2", 0) == "OM2-000000"
    assert stable_point_id("OM_2", 42) == "OM2-000042"


def test_point_ids_unique(om2_points):
    assert om2_points["point_id"].is_unique


# --- P-03: buffers exist at 5/10/20/50 m ---------------------------------

def test_buffer_radii_are_5_10_20_50():
    assert tuple(BUFFER_RADII_M) == (5, 10, 20, 50)


def test_buffer_columns_exist_for_every_radius(om2_points_small):
    buf = compute_buffer_variables(om2_points_small, PATHS)
    for r in BUFFER_RADII_M:
        assert f"lambda_p_buffer_{r}m" in buf.columns
        assert f"building_count_buffer_{r}m" in buf.columns
        assert f"building_height_mean_buffer_{r}m" in buf.columns
    # lambda_p buffer is a fraction
    for r in BUFFER_RADII_M:
        col = buf[f"lambda_p_buffer_{r}m"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0 + 1e-9


# --- P-03: segment aggregation conserves point count ----------------------

def test_segment_aggregation_conserves_point_count(om2_points):
    df = pd.DataFrame(om2_points.drop(columns="geometry"))
    df["dummy_value"] = np.arange(len(df), dtype=float)
    for seg_len in (5, 10, 20, 50, 137):
        segments = aggregate_to_segments(df, seg_len)
        assert int(segments["n_points"].sum()) == len(df)


def test_segment_aggregation_rejects_nonpositive_length(om2_points):
    df = pd.DataFrame(om2_points.drop(columns="geometry"))
    with pytest.raises(ValueError):
        aggregate_to_segments(df, 0)


# --- P-04 / P-06: joined variables sane ------------------------------------

def test_form_variables_computed_and_bounded(om2_points_small):
    form = compute_form_variables(om2_points_small, PATHS)
    assert len(form) == len(om2_points_small)
    svf = form["sky_view_factor"].dropna()
    assert (svf >= 0).all() and (svf <= 1 + 1e-9).all()
    assert form["street_orientation_deg"].between(0, 180).all()


def test_ventilation_proxies_present(om2_points_small):
    form = compute_form_variables(om2_points_small, PATHS)
    vent = compute_ventilation_proxies(om2_points_small, form["street_orientation_deg"].to_numpy(), PATHS)
    for col in [
        "ventilation_wind_alignment_proxy",
        "ventilation_frontal_area_proxy",
        "ventilation_openness_proxy",
        "ventilation_dist_open_space_proxy_m",
    ]:
        assert col in vent.columns
    align = vent["ventilation_wind_alignment_proxy"]
    assert (align.dropna() >= -1e-9).all() and (align.dropna() <= 1 + 1e-9).all()


# --- P-05: shade table ships with the right schema, empty (dates unknown) --

def test_empty_shade_table_schema():
    t = build_empty_shade_table()
    assert list(t.columns) == SHADE_TABLE_COLUMNS
    assert len(t) == 0


# --- P-07: PENDING items are real, named PENDING items --------------------

def test_pending_items_listed():
    assert "building_shade_per_5min" in PENDING_ITEMS
    assert "sky_view_factor_terrestrial" in PENDING_ITEMS
    assert "tree_shade" in PENDING_ITEMS


# --- P-08: dictionary covers every column, both directions ----------------

def test_dictionary_has_pending_rows():
    d = full_dictionary()
    pending = [k for k, v in d.items() if v["status"] == "PENDING"]
    assert set(PENDING_ITEMS).issubset(set(pending))
    assert len(pending) > 0


def test_dictionary_covers_every_output_column(om2_points_small):
    form = compute_form_variables(om2_points_small, PATHS)
    vent = compute_ventilation_proxies(om2_points_small, form["street_orientation_deg"].to_numpy(), PATHS)
    buf = compute_buffer_variables(om2_points_small, PATHS)
    points_cols = set(om2_points_small.drop(columns="geometry").columns) | {"x", "y"}
    all_cols = points_cols | set(form.columns) | set(vent.columns) | set(buf.columns)
    all_cols -= {"point_id"}  # join key, already covered once

    dict_ids = set(dictionary_dataframe()["id"])
    missing = all_cols - dict_ids
    assert not missing, f"columns with no data-dictionary row: {sorted(missing)}"


def test_no_dictionary_row_is_orphaned_from_a_real_table():
    # every dictionary id belongs to either the points table, the buffer
    # template, the shade table, or the PENDING registry — this is a
    # structural check (dictionary.py's own composition), not a live-data one.
    from src.om_package.dictionary import _BASE, _BUFFER_TEMPLATES, _PENDING, _SHADE_TABLE_ONLY

    known_sources = set(_BASE) | {t.format(r=r) for t in _BUFFER_TEMPLATES for r in BUFFER_RADII_M} | set(_PENDING) | set(_SHADE_TABLE_ONLY)
    assert set(full_dictionary()) == known_sources
