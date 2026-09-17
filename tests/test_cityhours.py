"""CITYHOURS acceptance: spec docs/cityhours_spec.md.

Real DTM/footprints/EPW/Favelas_Limit_2019/the WP-05 FULL run of record live
only in the main checkout (data/ and runs/ are gitignored, never copied into
this worktree) — same convention as tests/test_wp02_horizon.py and
tests/test_wp05_pilot.py. These tests exercise the pure array/DataFrame logic
(pilot tile selection, reproduction check, route-(a) mask recovery, the
summary shapes) on synthetic inputs; anything that needs the real EPW reads
it absolute from the main checkout and skips if it is not on disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.brisa_solar import cityhours, wp05_full
from src.brisa_solar.constants import P1_SKY_PATCHES, load_params
from src.svf_v2.compute import generate_tregenza_patches

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")


# ---------------------------------------------------------------------------
# (a) pilot tile selection is stratified by tile SIZE (not a scattered cell
# fraction) and stops once it reaches the target fraction of the frame
# ---------------------------------------------------------------------------

def _synthetic_frame(n_tiles_side: int, cells_per_tile_range: tuple[int, int], seed: int = 0) -> tuple[pd.DataFrame, float, float]:
    rng = np.random.default_rng(seed)
    tile_m = cityhours.TILE_M
    origin_x, origin_y = 0.0, 0.0
    rows = []
    for ti in range(n_tiles_side):
        for tj in range(n_tiles_side):
            n = int(rng.integers(*cells_per_tile_range))
            xs = origin_x + ti * tile_m + rng.uniform(0, tile_m, size=n)
            ys = origin_y + tj * tile_m + rng.uniform(0, tile_m, size=n)
            rows.append(pd.DataFrame({"x": xs, "y": ys}))
    df = pd.concat(rows, ignore_index=True)
    df["row"] = np.arange(len(df))
    df["col"] = np.arange(len(df))
    df["stratum"] = rng.integers(0, 9, size=len(df)).astype("int16")
    df["favela_id"] = np.zeros(len(df), dtype="int32")
    return df, origin_x, origin_y


def test_select_pilot_tiles_reaches_target_without_touching_every_tile():
    frame_df, ox, oy = _synthetic_frame(n_tiles_side=6, cells_per_tile_range=(100, 5000), seed=1)
    n_total_tiles = 36
    picked = cityhours.select_pilot_tiles(frame_df, ox, oy, target_fraction=0.10)

    assert len(picked) > 0
    assert len(picked) < n_total_tiles, "a stratified-by-tile pilot must not touch every tile"
    assert len(picked) == len(set(picked)), "no tile picked twice"

    i, j = cityhours.tile_index_for_xy(frame_df["x"].to_numpy(), frame_df["y"].to_numpy(), ox, oy)
    frame_df = frame_df.assign(_i=i, _j=j)
    picked_cells = sum(
        int(((frame_df["_i"] == pi) & (frame_df["_j"] == pj)).sum()) for pi, pj in picked
    )
    assert picked_cells >= 0.10 * len(frame_df) * 0.5, (
        "pilot should reach roughly its target fraction (allowing some overshoot/undershoot "
        "from picking whole tiles rather than a precise cell count)"
    )


def test_select_pilot_tiles_small_target_still_picks_at_least_one_tile():
    frame_df, ox, oy = _synthetic_frame(n_tiles_side=4, cells_per_tile_range=(50, 200), seed=2)
    picked = cityhours.select_pilot_tiles(frame_df, ox, oy, target_fraction=0.01)
    assert len(picked) >= 1


# ---------------------------------------------------------------------------
# (b) reproduction check: identical svf/kwh_m2 passes; a shifted copy fails
# ---------------------------------------------------------------------------

def test_reproduction_check_passes_on_identical_values(tmp_path):
    n = 500
    rng = np.random.default_rng(3)
    record = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "svf": rng.uniform(0.0, 1.0, size=n),
        "kwh_m2": rng.uniform(600.0, 2000.0, size=n),
    })
    record_path = tmp_path / "record.parquet"
    record.to_parquet(record_path, index=False)

    consolidated = record.copy()
    result = cityhours.reproduction_check(consolidated, record_path)

    assert result["pass"] is True
    assert result["svf"]["corr"] == pytest.approx(1.0, abs=1e-9)
    assert result["svf"]["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["kwh_m2"]["pass"] is True
    assert result["n_matched"] == n


def test_reproduction_check_fails_on_a_real_deviation(tmp_path):
    n = 500
    rng = np.random.default_rng(4)
    record = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "svf": rng.uniform(0.0, 1.0, size=n),
        "kwh_m2": rng.uniform(600.0, 2000.0, size=n),
    })
    record_path = tmp_path / "record.parquet"
    record.to_parquet(record_path, index=False)

    consolidated = record.copy()
    consolidated["svf"] = consolidated["svf"] + 0.05  # well past max_abs_diff_max=0.01
    result = cityhours.reproduction_check(consolidated, record_path)

    assert result["pass"] is False
    assert result["svf"]["pass"] is False
    assert result["svf"]["max_abs_diff"] == pytest.approx(0.05, abs=1e-9)


def test_reproduction_check_never_loosens_tolerance_by_default():
    # Spec: "Never loosen an acceptance tolerance to make a check pass" —
    # pin the defaults so a future edit can't quietly weaken them.
    import inspect

    sig = inspect.signature(cityhours.reproduction_check)
    assert sig.parameters["corr_min"].default == 0.9999
    assert sig.parameters["max_abs_diff_max"].default == 0.01


# ---------------------------------------------------------------------------
# (c) route (a) from the binary mask: shapes/dtypes and a closed-form case
# (an observer visible at every patch is visible whenever daylight holds)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def directions():
    d, _w = generate_tregenza_patches()
    return d


@pytest.fixture(scope="module")
def epw_meta():
    epw = MAIN_CHECKOUT / load_params()["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    from src.brisa_solar.wp04_sites import epw_meta as _epw_meta
    return _epw_meta(epw)


def test_route_a_fully_open_sky_hours_equal_daylight_hours(directions, epw_meta):
    from src.brisa_solar.wp05_pilot import pack_visibility

    n = 3
    vis = np.ones((n, P1_SKY_PATCHES), dtype=bool)  # every patch visible: no obstruction at all
    vis_packed = pack_visibility(vis)

    params = load_params()
    date_str = params["reference_days"]["equinox"]
    thresholds = params["reference_days"]["duration_thresholds_h"]

    result = cityhours.route_a_hours_for_day(vis_packed, P1_SKY_PATCHES, directions, date_str, epw_meta, thresholds)

    from src.brisa_solar.wp04_sites import sun_positions
    fine = sun_positions(date_str, epw_meta, "10min")
    expected_daylight_h = float((fine["apparent_elevation"].to_numpy() > 0.0).sum()) / 6.0

    assert result["hours_fractional"].shape == (n,)
    for v in result["hours_fractional"]:
        assert v == pytest.approx(expected_daylight_h, abs=1e-6)


def test_route_a_fully_obstructed_sky_gives_zero_hours(directions, epw_meta):
    from src.brisa_solar.wp05_pilot import pack_visibility

    n = 3
    vis = np.zeros((n, P1_SKY_PATCHES), dtype=bool)  # no patch ever visible
    vis_packed = pack_visibility(vis)

    params = load_params()
    date_str = params["reference_days"]["winter_solstice"]
    thresholds = params["reference_days"]["duration_thresholds_h"]

    result = cityhours.route_a_hours_for_day(vis_packed, P1_SKY_PATCHES, directions, date_str, epw_meta, thresholds)

    assert np.all(result["hours_fractional"] == 0.0)
    for k in thresholds:
        assert not np.any(result[f"ge_{k}h"])


# ---------------------------------------------------------------------------
# (d) summary.json shape: citywide percentiles use the SAME set the ledger's
# citywide.* entries use (wp05_full.QUANTILES), threshold shares are exact
# means, and no favela-vs-citywide contrast/difference/ratio field exists
# ---------------------------------------------------------------------------

def _square(x0, y0, s=10.0):
    from shapely.geometry import Polygon
    return Polygon([(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)])


def test_build_summary_shape_and_no_contrast_field():
    import geopandas as gpd

    n = 2000
    rng = np.random.default_rng(5)
    reference_days = {"winter_solstice": "2026-06-21", "equinox": "2026-03-20"}
    thresholds = [1, 2, 3, 4]

    consolidated = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "favela_id": np.where(np.arange(n) < 100, 1, 0).astype("int32"),
    })
    for label in reference_days:
        hours = rng.uniform(0.0, 10.0, size=n)
        consolidated[f"hours_{label}"] = hours
        for k in thresholds:
            consolidated[f"ge_{k}h_{label}"] = hours >= k

    favelas = gpd.GeoDataFrame({
        "objectid": [1], "cod_favela": [1], "nome": ["Vidigal"], "complexo": ["Isolada"],
        "geometry": [_square(0, 0)],
    }, crs="EPSG:31983")

    summary = cityhours.build_summary(consolidated, favelas, reference_days, thresholds)

    ordered_keys = [f"p{round(q * 100)}" for q in wp05_full.QUANTILES]
    for label in reference_days:
        block = summary["citywide"][f"sun_h_{label}"]
        values = [block[k] for k in ordered_keys]
        assert values == sorted(values), "sun-hour percentiles must be monotone non-decreasing"

        shares = summary["citywide"][f"share_ge_{label}"]
        for k in thresholds:
            expected = float(consolidated[f"ge_{k}h_{label}"].mean())
            assert shares[f"share_ge_{k}h"] == pytest.approx(expected, abs=1e-12)

    assert "Vidigal" in summary["study_favelas"]
    vidigal = summary["study_favelas"]["Vidigal"]
    for label in reference_days:
        assert "citywide_percentile_position" in vidigal[f"sun_h_{label}"]

    # Red line L1: no favela-vs-formal/non-favela contrast, difference, ratio, or
    # deficit anywhere in the summary document. Word-boundary match — "duration"
    # legitimately contains "ratio" as a substring and must not false-positive.
    import re

    banned = ("non_favela", "formal", "contrast", "difference", "deficit", "ratio")
    text = str(summary).lower()
    for token in banned:
        assert not re.search(rf"\b{token}\b", text), f"banned contrast token {token!r} found in summary.json"


def test_citywide_percentile_set_matches_ledger_convention():
    # docs/cityhours_spec.md: "the same percentile set the existing citywide.*
    # ledger entries use — read that set from the ledger, do not retype it."
    from src.brisa_solar.wp07_ledger import CITYWIDE_PERCENTILES

    ledger_set = set(CITYWIDE_PERCENTILES)
    cityhours_set = {f"p{round(q * 100)}" for q in wp05_full.QUANTILES}
    assert ledger_set == cityhours_set


# ---------------------------------------------------------------------------
# (e) per-tile timing carries the two costs CITYHOURS adds over WP-05 FULL
# ---------------------------------------------------------------------------

def test_cityhours_tile_timing_has_hours_s_field():
    t = cityhours.CityHoursTileTiming(
        tile_id="0_0", n_obs=10, build_s=1.0, engine_s=2.0, hours_s=0.5, peak_gb=0.1
    )
    assert t.hours_s == 0.5
    assert set(t.__dict__) == {"tile_id", "n_obs", "build_s", "engine_s", "hours_s", "peak_gb"}


# ---------------------------------------------------------------------------
# (f) consolidate_tiles (reused unmodified from wp05_full) accepts the
# cityhours tile-checkpoint shape and still catches a duplicate (x, y)
# ---------------------------------------------------------------------------

def test_consolidate_tiles_accepts_cityhours_shaped_checkpoints(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    rng = np.random.default_rng(6)
    for t in range(3):
        n = 20
        df = pd.DataFrame({
            "row": np.arange(t * n, (t + 1) * n), "col": np.arange(t * n, (t + 1) * n),
            "x": np.arange(t * n, (t + 1) * n, dtype="float64"), "y": np.full(n, float(t)),
            "favela_id": np.zeros(n, dtype="int32"), "svf": rng.uniform(0, 1, size=n),
            "kwh_m2": rng.uniform(600, 2000, size=n),
            "hours_winter_solstice": rng.uniform(0, 10, size=n),
            "ge_2h_winter_solstice": rng.uniform(0, 10, size=n) >= 2,
        })
        df.to_parquet(tile_dir / f"tile_{t}_0.parquet", index=False)

    merged = wp05_full.consolidate_tiles(tile_dir, tmp_path / "consolidated.parquet")
    assert len(merged) == 60
    assert "hours_winter_solstice" in merged.columns
