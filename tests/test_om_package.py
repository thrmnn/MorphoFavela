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
from src.om_package.package_docs import USE_TERMS, render_changelog, render_readme
from src.om_package.quality import DESCOPED_ITEMS, PENDING_ITEMS, coverage_report
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
    assert PENDING_ITEMS == []
    assert "sky_view_factor_terrestrial" in DESCOPED_ITEMS
    assert "tree_shade" in DESCOPED_ITEMS
    assert not set(PENDING_ITEMS) & set(DESCOPED_ITEMS)


# --- P-08: dictionary covers every column, both directions ----------------

def test_dictionary_marks_descoped_rows_not_pending():
    d = full_dictionary()
    for k in DESCOPED_ITEMS:
        assert d[k]["status"] == "DESCOPED (om_v013_descope)"
    assert not [k for k, v in d.items() if v["status"] == "PENDING"]


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
    # template, the shade table, or the DESCOPED registry — this is a
    # structural check (dictionary.py's own composition), not a live-data one.
    from src.om_package.dictionary import _BASE, _BUFFER_TEMPLATES, _DESCOPED, _SHADE_TABLE_ONLY

    known_sources = set(_BASE) | {t.format(r=r) for t in _BUFFER_TEMPLATES for r in BUFFER_RADII_M} | set(_DESCOPED) | set(_SHADE_TABLE_ONLY)
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

#: Fake but well-formed decisions fixture — same shape provenance.read_om_decisions()
#: returns, so render_readme/render_changelog can be exercised without a real
#: brisaverse checkout. Covers all seven ids the audit requires (2026-09-27).
_FAKE_DECISIONS = [
    {"id": "om_use_terms", "question": "q", "resolution": "Named team only, internal review draft.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_scope", "question": "q", "resolution": "Surface structure only.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_shade_release", "question": "q", "resolution": "Release building-only shade once dates are known.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_credit", "question": "q", "resolution": "Acknowledgment now, authorship later.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_dates_tz", "question": "q", "resolution": "Timezone stays an open question until confirmed.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_lidar", "question": "q", "resolution": "Location pending from the PI.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_route_geometry", "question": "q", "resolution": "Flag now; v0.2 rebuilds with a crosswalk.", "resolved_utc": "2026-09-24T18:39:11Z"},
    {"id": "om_v013_descope", "question": "q", "resolution": "Dropped from this version, deliberately.", "resolved_utc": "2026-10-01T13:46:07Z"},
]

_FAKE_NODATA_FLOOR_M = {"min": 105.1, "median": 338.4, "max": 598.9}

_README_STATS = dict(
    n_om2_points=1559,
    n_route_geometry_flagged=38,
    n_lambda_p_ones=124,
    n_lambda_p_ones_flagged=38,
    lambda_p_share_explained_pct=30.6,
    n_lambda_p_remainder=86,
    n_lambda_p_remainder_plausible=80,
    route_fetch_date_label="file dates 2026-09-24 (route JSON mtimes; no fetch manifest recorded)",
    nodata_floor_m=_FAKE_NODATA_FLOOR_M,
    internal_routes_status="not built in this version — no `outputs/_packages/_internal/mare_routes/v0.1.3` directory exists yet.",
    decisions=_FAKE_DECISIONS,
    dtm_native_resolution_m=5.0,
    p10_summary={"window": ["2025-12-01", "2026-04-30"], "dose_slot_min": 15, "date_dependent_share": 0.49,
                 "clock_agreement_all": 0.37},
    wind_source={"window_utc": ["2025-12-01", "2026-04-30"], "fetched_utc": "2026-10-01T17:31:31+00:00"},
    geometry_label="test epoch",
    route_length_m=1557.8,
    prevailing_deg=91.5,
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
    from src.om_package.dictionary import _DESCOPED

    assert not any("surface_cover" in k or "surface cover" in v.get("definition", "").lower() for k, v in _DESCOPED.items())


# --- 2026-09-27 numerical audit: docs/provenance defects --------------------

def test_render_readme_requires_nodata_floor_m_no_default():
    """A sabotaged call missing nodata_floor_m must fail loudly (TypeError:
    missing required argument), never silently render a default number —
    this used to be `shade_min_nan_dist_m: float = 104.15`."""
    stats = {k: v for k, v in _README_STATS.items() if k != "nodata_floor_m"}
    with pytest.raises(TypeError):
        render_readme(**stats)


def test_require_nodata_floor_m_fails_loudly_on_a_sabotaged_manifest():
    """A manifest whose p05_shade block lacks nodata_floor_m (e.g. from a
    build that skipped the measurement) must raise, not default, when a
    consumer tries to read it out for rendering."""
    from src.om_package.package_docs import require_nodata_floor_m

    sabotaged_p05_shade = {"n_csv_pilot": 5, "n_campaign_dates": 5}  # no nodata_floor_m key
    with pytest.raises(KeyError):
        require_nodata_floor_m(sabotaged_p05_shade)

    intact_p05_shade = {"nodata_floor_m": _FAKE_NODATA_FLOOR_M}
    assert require_nodata_floor_m(intact_p05_shade) == _FAKE_NODATA_FLOOR_M


def test_readme_cites_decision_ids_not_interview_codes():
    readme = render_readme(**_README_STATS)
    assert "Q6a" not in readme
    assert "om_use_terms" in readme


def test_readme_states_internal_routes_status_not_a_fixed_claim():
    readme = render_readme(**_README_STATS)
    assert _README_STATS["internal_routes_status"] in readme


def test_readme_lambda_p_remainder_is_a_computed_check():
    readme = render_readme(**_README_STATS)
    known_limits = readme.split("## Known limits")[1].split("## Manifest")[0]
    assert "building_count_buffer_10m > 0" in known_limits
    assert "86" in known_limits and "80" in known_limits


def test_readme_join_example_points_at_shipped_script():
    readme = render_readme(**_README_STATS)
    assert "OM2/join_shade_example.py" in readme
    known_limits = readme.split("## Known limits")[1].split("## Manifest")[0]
    assert "OM2/join_shade_example.py" in known_limits


def test_readme_manifest_section_states_self_hash_exclusion():
    readme = render_readme(**_README_STATS)
    manifest_section = readme.split("## Manifest")[1].split("## Use terms")[0]
    assert "excludes its own hash" in manifest_section


def test_readme_sources_no_invented_vintage():
    readme = render_readme(**_README_STATS)
    sources = readme.split("## Sources and dates")[1].split("## Methods")[0]
    assert "vintage not recorded in data/README.md" in sources
    assert "2019 cadastral clip" in sources
    assert "5 m native resolution" in sources


def _shipped_entry(version: str, heading: str):
    shipped_path = PATHS.root / "outputs" / "_packages" / "mare_om2" / version / "CHANGELOG.md"
    if not shipped_path.exists():
        pytest.skip(f"{version}/CHANGELOG.md not present at the default root")
    shipped = shipped_path.read_text(encoding="utf-8")
    return shipped, heading + shipped.split(heading, 1)[1]


def _entry(text: str, heading: str, next_heading: str | None) -> str:
    body = heading + text.split(heading, 1)[1]
    return body.split(next_heading, 1)[0] if next_heading else body


def _normalised(text: str) -> str:
    import re

    return re.sub(r"(?<!Brisa\+ \()MorphoFavela", "Brisa+ (MorphoFavela)", text)


def test_render_changelog_old_entries_are_frozen_not_rendered_from_state():
    """v0.1.3 and older render identically whatever the current version is
    (no live state reaches them)."""
    a = render_changelog(version="v0.2.0")
    b = render_changelog(version="v9.9.9", version_date="2099-01-01")
    assert a.split("## v0.1.3", 1)[1] == b.split("## v0.1.3", 1)[1]


def test_render_changelog_frozen_entries_match_shipped_v013_modulo_project_name():
    shipped, _ = _shipped_entry("v0.1.3", "## v0.1.3 — 2026-10-01")
    rendered = render_changelog()
    for heading, nxt in (("## v0.1.3 — 2026-10-01", "## v0.1.2"), ("## v0.1.2 — 2026-09-25", "## v0.1.1"),
                         ("## v0.1.1 — 2026-09-24", "## v0.1 — 2026-09-24"), ("## v0.1 — 2026-09-24", None)):
        want = _normalised(_entry(shipped, heading, nxt))
        got = _entry(rendered, heading, nxt)
        assert got.rstrip() == want.rstrip(), heading


def test_render_changelog_v011_entry_fixes_the_version_drift_bug():
    rendered = render_changelog()
    v011 = rendered.split("## v0.1.1 — 2026-09-24\n", 1)[1].split("## v0.1 — 2026-09-24", 1)[0]
    assert "mare_routes/v0.1.1/" in v011
    assert "OM1" in v011
    assert "mare_routes/v0.1.2/" not in v011


def test_render_changelog_v012_entry_no_qcodes_no_hardcoded_floor():
    rendered = render_changelog()
    v012 = rendered.split("## v0.1.2 — 2026-09-25\n", 1)[1].split("## v0.1.1 — 2026-09-24", 1)[0]
    assert "Q1/Q5" not in v012
    assert "~104-330" not in v012
    assert "om_shade_release" in v012 and "om_dates_tz" in v012


def test_changelog_no_bare_project_name():
    import re

    assert not re.search(r"(?<!Brisa\+ \()MorphoFavela", render_changelog())


def test_readme_has_using_the_data_and_segment_note_without_a_typed_time_constant():
    readme = render_readme(**_README_STATS)
    section = readme.split("## Using the data")[1].split("## Known limits")[0]
    assert "--segment-m" in section and "time constant" in section
    assert "does not state an L" in section
    assert "95%" in section  # computed 1 - e^-3


TASKS_JSON = Path("/home/theo/SCL/SCR/brisaverse/shared/facts/tasks.json")


@pytest.mark.skipif(not TASKS_JSON.exists(), reason="brisaverse shared/facts/tasks.json not found at the default root")
def test_provenance_decisions_present_with_all_ids():
    from src.om_package.provenance import OM_DECISION_IDS, read_om_decisions

    decisions = read_om_decisions()
    assert [d["id"] for d in decisions] == OM_DECISION_IDS
    for d in decisions:
        assert d["resolution"]
        assert d["resolved_utc"]


def test_hash_tree_excludes_manifest_self_hash(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    # simulate a STALE manifest.json already on disk from a previous build
    (tmp_path / "manifest.json").write_text('{"package_version": "stale"}')
    files = hash_tree(tmp_path, exclude={"manifest.json"})
    assert "manifest.json" not in files
    assert "a.txt" in files


def test_internal_routes_status_and_route_fetch_date_label(tmp_path):
    build_mod = _load_build_module()
    root = tmp_path
    (root / "outputs" / "_packages" / "_internal" / "mare_routes" / "v9.9.9" / "OM1").mkdir(parents=True)
    status = build_mod.internal_routes_status(root, "v9.9.9")
    assert "OM1" in status
    status_missing = build_mod.internal_routes_status(root, "v0.0.0")
    assert "not built in this version" in status_missing

    routes_dir = root / "data" / "maré" / "octopus" / "routes"
    routes_dir.mkdir(parents=True)
    (routes_dir / "OM_2_inferred_route.json").write_text("{}")
    fake_paths = Paths(root)
    label = build_mod.route_fetch_date_label(fake_paths)
    assert "file dates" in label
