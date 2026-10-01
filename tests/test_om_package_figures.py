"""Tests for src/om_package/figures.py — the OM2 package figures (PI,
2026-09-27: spatial result first, then sampling along the route; route
overlaid on the favela buildings). Synthetic data only: no real data/
tree is read here.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import box

from src.om_package import figures
from src.om_package.segments import aggregate_to_segments

N_POINTS = 60


@pytest.fixture
def points_df() -> pd.DataFrame:
    x = np.arange(N_POINTS, dtype=float)
    y = np.zeros(N_POINTS)
    neighbourhood = np.where(x < N_POINTS / 2, "CommunityA", "CommunityB")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "point_id": [f"OM2-{i:06d}" for i in range(N_POINTS)],
            "route_id": "OM_2",
            "seq": np.arange(N_POINTS),
            "x": x,
            "y": y,
            "distance_along_m": x,
            "neighbourhood": neighbourhood,
            "building_height_m": 5 + 2 * np.sin(x / 5),
            "height_width_ratio": 0.5 + 0.1 * np.cos(x / 7),
            "sky_view_factor": np.clip(0.5 + 0.3 * np.sin(x / 10), 0, 1),
            "plan_density_lambda_p": np.clip(0.4 + 0.2 * np.cos(x / 9), 0, 1),
            "ventilation_frontal_area_proxy": rng.uniform(0, 1, N_POINTS),
        }
    )


@pytest.fixture
def buildings() -> gpd.GeoDataFrame:
    geoms = [box(10, -6, 14, -2), box(38, 2, 42, 6), box(-5, -8, -1, -4)]
    return gpd.GeoDataFrame({"altura": [6.0, 9.0, 4.0]}, geometry=geoms, crs="EPSG:31983")


@pytest.fixture
def subunits() -> gpd.GeoDataFrame:
    geoms = [box(-20, -20, N_POINTS / 2, 20), box(N_POINTS / 2, -20, N_POINTS + 20, 20)]
    return gpd.GeoDataFrame({"name": ["CommunityA", "CommunityB"]}, geometry=geoms, crs="EPSG:31983")


@pytest.fixture
def shade_df(points_df) -> pd.DataFrame:
    dates = ["2026-01-06", "2026-01-07"]
    rows = []
    for d in dates:
        times = pd.date_range(f"{d} 09:00", f"{d} 10:00", freq="5min", tz="UTC")
        for pid, x in zip(points_df["point_id"], points_df["x"]):
            for t in times:
                # deterministic shaded pattern: shaded where (x + minute) is even
                shaded = bool((int(x) + t.minute) % 4 == 0)
                # the first step of each date is night: shaded=True as the
                # shade engine writes it, never counted as building shade
                night = t.minute == 0 and t.hour == 9
                rows.append({"point_id": pid, "timestamp": t, "date": d,
                             "sun_altitude_deg": -5.0 if night else 30.0,
                             "shaded": True if night else shaded, "tree_shade": None})
    return pd.DataFrame(rows)


@pytest.fixture
def campaign_windows_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-06", "2026-01-07"],
            "first_timestamp": pd.to_datetime(["2026-01-06 09:00", "2026-01-07 09:00"], utc=True),
            "last_timestamp": pd.to_datetime(["2026-01-06 10:00", "2026-01-07 10:00"], utc=True),
        }
    )


@pytest.fixture
def dictionary_df() -> pd.DataFrame:
    rows = [
        {"id": "building_height_m", "unit": "m"},
        {"id": "height_width_ratio", "unit": "-"},
        {"id": "sky_view_factor", "unit": "fraction [0,1]"},
        {"id": "plan_density_lambda_p", "unit": "fraction [0,1]"},
        {"id": "ventilation_frontal_area_proxy", "unit": "proxy, lambda_f (dimensionless)"},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------- F1/F2 --

def test_build_map_form_writes_png(tmp_path, points_df, buildings, subunits):
    out = tmp_path / "map_form.png"
    result = figures.build_map_form(points_df, buildings, subunits, out, route_id="OM2", version="v0.1.3")
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.width > 100 and im.height > 100


def test_build_map_form_runs_without_buildings_or_subunits(tmp_path, points_df):
    out = tmp_path / "map_form.png"
    figures.build_map_form(points_df, None, None, out, route_id="OM2", version="v0.1.3")
    assert out.exists() and out.stat().st_size > 0


def test_build_map_shade_writes_png(tmp_path, points_df, shade_df, buildings, subunits):
    out = tmp_path / "map_shade.png"
    figures.build_map_shade(points_df, shade_df, buildings, subunits, out, route_id="OM2", version="v0.1.3", tz="UTC")
    assert out.exists() and out.stat().st_size > 0


def test_build_map_shade_handles_empty_shade_table(tmp_path, points_df, buildings, subunits):
    empty = pd.DataFrame(columns=["point_id", "timestamp", "date", "shaded", "tree_shade"])
    out = tmp_path / "map_shade.png"
    figures.build_map_shade(points_df, empty, buildings, subunits, out, route_id="OM2", version="v0.1.3")
    assert out.exists() and out.stat().st_size > 0


def test_mean_shaded_fraction_by_point_is_correct(shade_df):
    frac = figures.mean_shaded_fraction_by_point(shade_df)
    manual = shade_df[shade_df["sun_altitude_deg"] > 0].groupby("point_id")["shaded"].mean()
    pd.testing.assert_series_equal(frac.sort_index(), manual.sort_index(), check_names=False)
    with_night = shade_df.groupby("point_id")["shaded"].mean()
    assert (with_night.sort_index() > frac.sort_index()).all()


def test_mean_shaded_fraction_by_point_empty_input_is_empty_series():
    frac = figures.mean_shaded_fraction_by_point(pd.DataFrame(columns=["point_id", "shaded"]))
    assert len(frac) == 0


# ------------------------------------------------------------------ F3 --

def test_build_profiles_writes_png(tmp_path, points_df, shade_df, dictionary_df):
    out = tmp_path / "profiles.png"
    figures.build_profiles(points_df, shade_df, out, dictionary_df=dictionary_df, route_id="OM2", version="v0.1.3")
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.height > im.width  # stacked panels: tall figure


def test_profile_segment_means_equal_segments_aggregate_to_segments(points_df, shade_df):
    frac = figures.mean_shaded_fraction_by_point(shade_df)
    expected_frame = points_df.merge(frac.rename("mean_shaded_fraction"), left_on="point_id", right_index=True, how="left")
    expected = aggregate_to_segments(expected_frame, figures.SEGMENT_LENGTH_M)

    got = figures.segment_means_for_profiles(points_df, shade_df, figures.SEGMENT_LENGTH_M)

    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))
    # sanity: segments really are ~10 m and conserve point count
    assert got["n_points"].sum() == len(points_df)
    assert (got["segment_end_m"] - got["segment_start_m"]).le(figures.SEGMENT_LENGTH_M).all()


# ------------------------------------------------------------------ F4 --

def test_build_shade_calendar_writes_png(tmp_path, points_df, shade_df, campaign_windows_df):
    out = tmp_path / "shade_calendar.png"
    figures.build_shade_calendar(points_df, shade_df, campaign_windows_df, out, route_id="OM2", version="v0.1.3")
    assert out.exists() and out.stat().st_size > 0


def test_build_shade_calendar_one_strip_per_date(tmp_path, points_df, shade_df, campaign_windows_df):
    """More campaign dates -> a taller image (one strip per date)."""
    one_date = shade_df[shade_df["date"] == "2026-01-06"]
    out_one = tmp_path / "one.png"
    out_two = tmp_path / "two.png"
    figures.build_shade_calendar(points_df, one_date, campaign_windows_df, out_one)
    figures.build_shade_calendar(points_df, shade_df, campaign_windows_df, out_two)
    with Image.open(out_one) as im_one, Image.open(out_two) as im_two:
        assert im_two.height > im_one.height


def test_build_shade_calendar_handles_empty_shade_table(tmp_path, points_df):
    empty = pd.DataFrame(columns=["point_id", "timestamp", "date", "shaded", "tree_shade"])
    out = tmp_path / "shade_calendar.png"
    figures.build_shade_calendar(points_df, empty, None, out)
    assert out.exists() and out.stat().st_size > 0


def test_dates_in_shade_table_sorted_unique(shade_df):
    dates = figures._dates_in_shade_table(shade_df)
    assert dates == ["2026-01-06", "2026-01-07"]


# --------------------------------------------------------------- rcParams --

def test_figures_do_not_leak_rcparams(tmp_path, points_df, buildings, subunits):
    """A leaked rcParam from one figure once broke the next drawn in the
    same process — every renderer must restore the caller's rcParams.
    Scoped with rc_context (not rcdefaults(), which would itself clobber
    whatever rcParams other, unrelated test modules rely on persisting)."""
    with matplotlib.rc_context({"font.size": 42}):
        figures.build_map_form(points_df, buildings, subunits, tmp_path / "map_form.png")
        assert matplotlib.rcParams["font.size"] == 42
