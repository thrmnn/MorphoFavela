"""Tests for the OM2 report figures (src/om_package/figures.py, vent_figures.py,
fig_style.py). Synthetic data only: no real data/ tree is read here."""
from __future__ import annotations

import re

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import box

from src.om_package import fig_style, figures, vent_figures
from src.om_package.wind_regimes import find_regimes

N = 120


@pytest.fixture
def points() -> pd.DataFrame:
    x = np.arange(N, dtype=float)
    rng = np.random.default_rng(0)
    d = {
        "point_id": [f"OM2-{i:06d}" for i in range(N)], "x": x, "y": 0.1 * x, "distance_along_m": x,
        "neighbourhood": np.where(x < 50, "Alpha Town", np.where(x < 60, None, "Beta Park")),
        "building_height_m": 5 + 2 * np.sin(x / 5), "height_width_ratio": 1 + 0.2 * np.cos(x / 7),
        "sky_view_factor": np.clip(0.5 + 0.3 * np.sin(x / 10), 0, 1),
        "plan_density_lambda_p": np.clip(0.4 + 0.2 * np.cos(x / 9), 0, 1),
    }
    for slug in ("east_southeast", "north_northwest"):
        d[f"frontal_area_density_windward_{slug}"] = rng.uniform(0, 2, N)
        d[f"canyon_alignment_deg_{slug}"] = rng.uniform(0, 90, N)
        d[f"upwind_shelter_angle_deg_{slug}"] = rng.uniform(0, 80, N)
    return pd.DataFrame(d)


@pytest.fixture
def buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"h": [6.0, 9.0]}, geometry=[box(10, -6, 14, -2), box(38, 2, 42, 6)], crs="EPSG:31983")


@pytest.fixture
def shade_df(points) -> pd.DataFrame:
    ts = pd.date_range("2026-01-06 05:00", "2026-01-06 18:00", freq="5min", tz="America/Sao_Paulo")
    ts2 = pd.date_range("2026-01-07 05:00", "2026-01-07 18:00", freq="5min", tz="America/Sao_Paulo")
    rows = []
    rng = np.random.default_rng(1)
    for t in list(ts) + list(ts2):
        for pid in points["point_id"][::10]:
            rows.append((pid, t, str(t.date()), 30.0, bool(rng.random() < 0.5)))
    return pd.DataFrame(rows, columns=["point_id", "timestamp_local", "date", "sun_altitude_deg", "shaded"])


@pytest.fixture
def walks() -> pd.DataFrame:
    return pd.DataFrame({
        "walk_id": ["w1", "w2", "w3", "w4"], "date": ["2026-01-06", "2026-01-07", "2026-01-06", "2026-01-07"],
        "period": ["morning", "morning", "evening", "evening"],
        "start_local": ["2026-01-06T09:30:00-03:00", "2026-01-07T09:31:00-03:00",
                        "2026-01-06T15:30:00-03:00", "2026-01-07T15:29:00-03:00"],
        "duration_min": [20.0, 27.0, 26.0, 40.0], "coverage_share": [0.99, 0.97, 0.80, 0.99],
    })


@pytest.fixture
def p12(points, walks) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    parts = []
    for w in walks["walk_id"]:
        d = pd.DataFrame({"walk_id": w, "distance_along_m": points["distance_along_m"],
                          "dose_1h_before_wh_m2": rng.uniform(0, 500, N), "dose_3h_before_wh_m2": rng.uniform(0, 1500, N),
                          "sky_view_factor_tau10s": 0.5, "sky_view_factor_tau30s": 0.5})
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _size(path):
    return Image.open(path).size


def test_route_form_and_shade_figures_are_written_within_print_width(tmp_path, points, buildings, shade_df):
    figures.build_fig_route(points, buildings, tmp_path / "r.png")
    figures.build_fig_form(points, tmp_path / "f.png")
    figures.build_fig_shade_map(points, shade_df, buildings, tmp_path / "s.png")
    max_px = round(fig_style.TEXT_WIDTH_IN * fig_style.DPI)
    assert _size(tmp_path / "f.png")[0] == max_px
    for n in ("r", "s"):
        assert _size(tmp_path / f"{n}.png")[0] <= max_px


def test_shade_calendar_matrix_is_share_of_points_by_date_and_time(shade_df):
    mat, dates = figures.shade_calendar_matrix(shade_df)
    assert dates == ["2026-01-06", "2026-01-07"]
    assert mat.shape[0] == 2 and np.nanmin(mat.to_numpy()) >= 0 and np.nanmax(mat.to_numpy()) <= 1


def test_shade_calendar_written_with_facts(tmp_path, shade_df):
    _, facts = figures.build_fig_shade_calendar(shade_df, tmp_path / "c.png")
    assert facts["n_dates"] == 2 and facts["bin_min"] == 5 and (tmp_path / "c.png").exists()


def test_walk_order_groups_mornings_first_and_labels_with_start_time(walks):
    o = figures.walk_order(walks.iloc[::-1])
    assert o["walk_id"].tolist() == ["w1", "w2", "w3", "w4"]
    assert o["label"].tolist() == ["6 Jan 09:30", "7 Jan 09:31", "6 Jan 15:30", "7 Jan 15:29"]


def test_dose_matrix_bins_10_m_means_and_leaves_unwalked_bins_blank(p12):
    p = p12[~((p12["walk_id"] == "w1") & (p12["distance_along_m"] >= 60))]
    m = figures.dose_matrix(p, ["w1", "w2"], "dose_1h_before_wh_m2", 120.0)
    assert m.shape == (2, 12)
    assert np.isnan(m[0, 6:]).all() and np.isfinite(m[1]).all()
    expect = p[(p["walk_id"] == "w2") & (p["distance_along_m"] < 10)]["dose_1h_before_wh_m2"].mean()
    assert m[1, 0] == pytest.approx(expect)


def test_sun_dose_figure_and_facts(tmp_path, walks, p12):
    _, facts = figures.build_fig_sun_dose(walks, p12, 120.0, tmp_path / "d.png")
    assert facts["n_walks"] == 4 and facts["n_morning"] == 2 and facts["zero_drawn_below_wh_m2"] == figures.DOSE_ZERO_BELOW


def test_representative_walk_is_full_coverage_and_closest_to_median(walks):
    w = figures.pick_representative_walk(walks)
    assert w["walk_id"] == "w2"  # median duration 26.5; w3 is below 0.95 coverage


def test_svf_sensor_figure_names_the_walk(tmp_path, points, walks, p12):
    _, facts = figures.build_fig_svf_sensor(points, walks, p12, tmp_path / "v.png")
    assert facts["walk_id"] == "w2" and facts["start_local"] == "09:31"


def test_neighbourhood_stretches_fill_unnamed_points_and_cover_the_route(points):
    s = fig_style.neighbourhood_stretches(points)
    assert [x["name"] for x in s] == ["Alpha Town", "Beta Park"]
    assert s[0]["start"] == 0 and s[0]["end"] == s[1]["start"]


def _season():
    rng = np.random.default_rng(3)
    d = np.concatenate([rng.normal(120, 15, 600), rng.normal(340, 25, 400)]) % 360
    d = np.round(d, -1) % 360
    obs = pd.DataFrame({"valid_utc": pd.date_range("2026-01-01", periods=1000, freq="h", tz="UTC"),
                        "drct": d, "speed_ms": 3.0, "calm": False, "variable": False})
    res = find_regimes(obs)
    return {"campaign": res, "climatology": res}, obs


def test_wind_and_ventilation_figures(tmp_path, points, buildings):
    season, obs = _season()
    regs = [{"key": g["key"], "name": g["name"], "slug": g["name"].replace("-", "_"), "mean_direction_deg": g["mean_direction_deg"]}
            for g in season["campaign"]["regimes"]]
    pts = points.rename(columns=lambda c: c)
    for stem in ("frontal_area_density_windward", "canyon_alignment_deg", "upwind_shelter_angle_deg"):
        for slug, g in zip(("east_southeast", "north_northwest"), regs):
            pts[f"{stem}_{g['slug']}"] = pts[f"{stem}_{slug}"]
    rows = [(p, h, g["name"], g["key"], 0.5) for p in ("campaign", "climatology") for h in range(24) for g in regs]
    by_hour = pd.DataFrame(rows, columns=["period", "local_hour", "regime", "regime_key", "share"])
    calm = pd.DataFrame([(p, h, "calm", "calm", 0.01) for p in ("campaign", "climatology") for h in range(24)], columns=by_hour.columns)
    _, facts = vent_figures.build_fig_wind(season, obs, obs, pd.concat([by_hour, calm]), tmp_path / "w.png")
    assert facts["n_sectors"] == 16
    vent_figures.build_fig_vent_profiles(pts, regs, tmp_path / "p.png")
    _, f2 = vent_figures.build_fig_shelter_maps(pts, regs, buildings, tmp_path / "m.png")
    assert f2["shelter_colour_limits_deg"][0] == 0.0


def test_sector_shares_sum_to_100():
    _, obs = _season()
    assert vent_figures.sector_shares(obs).sum() == pytest.approx(100.0)


def test_figures_do_not_leak_rcparams(tmp_path, points, buildings):
    with matplotlib.rc_context({"font.size": 42}):
        figures.build_fig_route(points, buildings, tmp_path / "r.png")
        assert matplotlib.rcParams["font.size"] == 42


def test_no_dashes_or_abbreviations_in_figure_text():
    src = "".join(open(f"src/om_package/{m}.py", encoding="utf-8").read() for m in ("figures", "vent_figures", "fig_style"))
    labels = re.findall(r'(?:set_[xy]?label|label=|text\()[^\n]*?"([^"\n]*)"', src)
    for t in labels:
        assert "–" not in t and "—" not in t, t
        assert not re.search(r"SBGL|METAR|H/W|SVF|\bz0\b|λ", t), t


def test_print_width_survives_a_global_tight_bbox(tmp_path, points):
    import matplotlib

    # src/cartography.py sets savefig.bbox="tight" at import; in the full suite
    # that cropped fig_form below text width.
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        figures.build_fig_form(points, tmp_path / "f.png")
    assert _size(tmp_path / "f.png")[0] == round(fig_style.TEXT_WIDTH_IN * fig_style.DPI)
