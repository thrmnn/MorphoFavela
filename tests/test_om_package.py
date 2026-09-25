"""Tests for the Maré morphology, OM2 data package (v0.1).

Uses real Maré inputs at the default repo root (data/outputs are gitignored
and not present in a worktree checkout — these tests need to run from a
checkout that has them, same convention as this repo's other real-data
integration tests, e.g. tests/test_wp01_footprints.py).
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.om_package.buffers import BUFFER_RADII_M, compute_buffer_variables
from src.om_package.dictionary import dictionary_dataframe, full_dictionary
from src.om_package.formvars import compute_form_variables
from src.om_package.io_utils import Paths, hash_tree, write_table
from src.om_package.package_docs import USE_TERMS, render_readme
from src.om_package.quality import PENDING_ITEMS, coverage_report
from src.om_package.routes import (
    POINT_SPACING_M,
    ROUTE_FLAG_MAX_STREET_DIST_M,
    compute_route_geometry_flag,
    densify_route,
    stable_point_id,
)
from src.om_package.segments import aggregate_to_segments
from src.om_package.shade import (
    SHADE_TABLE_COLUMNS,
    build_empty_shade_table,
    drop_nofix_rows,
    infer_campaign_windows,
    sun_positions,
)
from src.om_package.ventilation import LAMBDA_F_DIRECTION_COLS, compute_ventilation_proxies

PATHS = Paths()


def _load_build_module():
    """scripts/build_om_package.py isn't a package — load it by path so its
    small pure helpers (route_output_dir) are unit-testable without paying
    for a full 4-route build."""
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_om_package", repo_root / "scripts" / "build_om_package.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
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
    # building_shade_per_5min moved out of PENDING in v0.1.2: point_horizon_profiles()
    # is wired for real and compute_shade() produced a real (non-empty) table from the
    # pilot CSV pull — see the 'shaded' dictionary row instead (_SHADE_TABLE_ONLY).
    assert "building_shade_per_5min" not in PENDING_ITEMS
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
    flag_col = compute_route_geometry_flag(om2_points_small, PATHS)
    points_cols = set(om2_points_small.drop(columns="geometry").columns) | {"x", "y", flag_col.name}
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


def test_segment_columns_have_dictionary_rows():
    dict_ids = set(dictionary_dataframe()["id"])
    for col in ("segment_id", "segment_start_m", "segment_end_m", "n_points"):
        assert col in dict_ids


# --- must-fix 2: OM2-only in the shared package path -----------------------

def test_om2_only_route_output_dir(tmp_path):
    mod = _load_build_module()
    shared = tmp_path / "mare_om2" / "v0.1.1"
    internal = tmp_path / "_internal" / "mare_routes" / "v0.1.1"
    assert mod.route_output_dir("OM_2", shared, internal) == shared
    for om in ("OM_1", "OM_3", "OM_4"):
        assert mod.route_output_dir(om, shared, internal) == internal


# --- must-fix 1: route_geometry_flag ---------------------------------------

def test_route_geometry_flag_is_boolean_and_aligned(om2_points_small):
    flag = compute_route_geometry_flag(om2_points_small, PATHS)
    assert len(flag) == len(om2_points_small)
    assert flag.dtype == bool
    assert list(flag.index) == list(om2_points_small.index)


def test_route_geometry_flag_counted_in_quality_report(om2_points_small):
    df = om2_points_small.copy()
    df["route_geometry_flag"] = compute_route_geometry_flag(om2_points_small, PATHS).to_numpy()
    df["dummy"] = 1.0
    report = coverage_report(df, ["dummy", "route_geometry_flag"])
    assert report["route_geometry_flagged_points"] == int(df["route_geometry_flag"].sum())


def test_route_flag_threshold_is_10m():
    assert ROUTE_FLAG_MAX_STREET_DIST_M == 10.0


# --- must-fix 4: schema freeze ----------------------------------------------

def test_lambda_f_direction_columns_joined(om2_points_small):
    form = compute_form_variables(om2_points_small, PATHS)
    vent = compute_ventilation_proxies(om2_points_small, form["street_orientation_deg"].to_numpy(), PATHS)
    assert len(LAMBDA_F_DIRECTION_COLS) == 8
    for col in LAMBDA_F_DIRECTION_COLS:
        assert col in vent.columns


def test_grid_cell_id_present(om2_points_small):
    form = compute_form_variables(om2_points_small, PATHS)
    assert "grid_cell_id" in form.columns


def test_shade_table_reserves_tree_shade_column():
    assert "tree_shade" in SHADE_TABLE_COLUMNS
    t = build_empty_shade_table()
    assert "tree_shade" in t.columns


# --- must-fix 5: timezone is a required parameter, no default --------------

def test_sun_positions_requires_tz():
    with pytest.raises(TypeError):
        sun_positions(["2026-01-01"], ("08:00", "18:00"), 5, -22.86, -43.24)


def test_drop_nofix_rows_removes_zero_zero_sentinel():
    df = pd.DataFrame(
        {
            "Latitude": [-22.86, 0.0, -22.87],
            "Longitude": [-43.24, 0.0, -43.25],
            "Temperature": [28.0, 29.0, 27.5],
        }
    )
    out = drop_nofix_rows(df)
    assert len(out) == 2
    assert not ((out["Latitude"] == 0.0) & (out["Longitude"] == 0.0)).any()


def test_infer_campaign_windows(tmp_path):
    csv_path = tmp_path / "log0.csv"
    csv_path.write_text(
        "Timestamp,Latitude,Longitude,Temperature\n"
        "2026-03-01 08:00:00,-22.860,-43.240,27.5\n"
        "2026-03-01 08:00:05,0.0,0.0,27.6\n"
        "2026-03-01 08:00:10,-22.861,-43.241,27.6\n"
    )
    windows = infer_campaign_windows([csv_path])
    assert len(windows) == 1
    row = windows.iloc[0]
    assert str(row["date"]) == "2026-03-01"
    assert row["n_rows"] == 3
    assert row["n_fix"] == 2
    assert row["n_no_fix"] == 1
    assert row["first_timestamp"] == pd.Timestamp("2026-03-01 08:00:00")
    assert row["last_timestamp"] == pd.Timestamp("2026-03-01 08:00:10")
    assert row["has_gps"] == True  # noqa: E712
    assert row["n_epoch_reset"] == 0


def test_infer_campaign_windows_no_gps_schema(tmp_path):
    # v0.1.2: the Zenodo_release/fixed_data pilot pull (2026-09-25) has no
    # Latitude/Longitude column at all (I_1/I_3/I_4/O_3/O_4 device schema) —
    # infer_campaign_windows must report has_gps=False, n_fix=n_rows,
    # n_no_fix=0 instead of raising a KeyError.
    csv_path = tmp_path / "O_4_log.csv"
    csv_path.write_text(
        "Timestamp,Temperature,Humidity,PM1.0,PM2.5,PM2.5_cal,PM4.0,PM10.0\n"
        "2026-01-06 13:32:09,30.80,60.1,0.0,0.0,0,0.0,0.0\n"
        "2026-01-06 13:32:14,30.83,60.2,0.1,0.2,4,0.3,0.4\n"
    )
    windows = infer_campaign_windows([csv_path])
    row = windows.iloc[0]
    assert row["has_gps"] == False  # noqa: E712
    assert row["n_fix"] == 2
    assert row["n_no_fix"] == 0


def test_infer_campaign_windows_flags_epoch_reset(tmp_path):
    csv_path = tmp_path / "log_epoch.csv"
    csv_path.write_text(
        "Timestamp,Latitude,Longitude,Temperature\n"
        "2000-01-01 00:00:00,0.0,0.0,25.0\n"
        "2000-01-01 00:00:05,0.0,0.0,25.1\n"
        "2026-03-01 08:00:10,-22.861,-43.241,27.6\n"
    )
    windows = infer_campaign_windows([csv_path])
    assert windows.iloc[0]["n_epoch_reset"] == 2


# --- must-fix 7: manifest and GeoParquet -------------------------------------

def test_hash_tree_matches_file_contents(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("world")

    hashes = hash_tree(tmp_path)
    assert hashes["a.txt"] == hashlib.sha256(b"hello").hexdigest()
    assert hashes["sub/b.txt"] == hashlib.sha256(b"world").hexdigest()
    assert set(hashes) == {"a.txt", "sub/b.txt"}


def test_write_table_geo_keeps_geoparquet_metadata_and_xy(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"point_id": ["p0", "p1"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:31983"
    )
    written = write_table(gdf, tmp_path, "test_points", geo=True)
    pq_path = tmp_path / "test_points.parquet"
    assert pq_path in written

    roundtrip = gpd.read_parquet(pq_path)
    assert "geometry" in roundtrip.columns
    assert roundtrip.crs is not None
    assert list(roundtrip["x"]) == [0.0, 1.0]
    assert list(roundtrip["y"]) == [0.0, 1.0]

    csv_df = pd.read_csv(tmp_path / "test_points.csv")
    assert "geometry" not in csv_df.columns
    assert "x" in csv_df.columns and "y" in csv_df.columns


# --- must-fix 3: use-terms banner, no PLACEHOLDER --------------------------

_README_STATS = dict(
    n_om2_points=1559,
    n_route_geometry_flagged=38,
    n_lambda_p_ones=124,
    n_lambda_p_ones_flagged=38,
    lambda_p_share_explained_pct=30.6,
)


def test_readme_has_no_placeholder():
    readme = render_readme(**_README_STATS)
    assert "PLACEHOLDER" not in readme


def test_readme_has_use_terms_banner():
    readme = render_readme(**_README_STATS)
    assert USE_TERMS in readme
    assert readme.strip().startswith(">")


def test_readme_states_om2_only_release_scope():
    readme = render_readme(**_README_STATS)
    assert "OM2 only" in readme
    assert "_internal" in readme


def test_readme_has_how_to_cite_acknowledgment():
    readme = render_readme(**_README_STATS)
    assert "Théo Hermann" in readme
    assert "How to cite" in readme


def test_readme_no_pending_surface_cover_row():
    readme = render_readme(**_README_STATS)
    coverage_section = readme.split("## Coverage vs Table 1")[1].split("## CRS")[0]
    assert "surface structure only" in coverage_section
    assert "façade materials" in coverage_section
    # the panel's suggested PENDING surface-cover row was overruled (PI, 2026-09-24)
    from src.om_package.dictionary import _PENDING

    assert not any("surface_cover" in k or "surface cover" in v.get("definition", "").lower() for k, v in _PENDING.items())
