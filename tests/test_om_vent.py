"""Tests for the OM2 ventilation-proxy lane (wind_obs, vent_indices). Synthetic data only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.om_package import vent_indices as vi
from src.om_package import wind_obs as wo


@pytest.mark.parametrize("wind,street,expected", [
    (0, 0, 0), (90, 0, 90), (180, 0, 0), (270, 0, 90),
    (0, 180, 0), (45, 0, 45), (135, 0, 45), (350, 10, 20),
])
def test_canyon_alignment_folds(wind, street, expected):
    assert vi.canyon_alignment_deg(street, wind) == pytest.approx(expected)


def test_shelter_angle_is_horizon_at_upwind_azimuth():
    az = np.array([0.0, 90.0, 180.0, 270.0])
    horizon = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
    assert vi.upwind_shelter_deg(horizon, az, 180.0).tolist() == [30.0, 3.0]
    assert vi.upwind_shelter_deg(horizon, az, 358.0).tolist() == [10.0, 1.0]
    assert vi.upwind_shelter_deg(horizon, az, np.array([90.0, 270.0])).tolist() == [20.0, 4.0]


def test_windward_lambda_f_interpolates_circularly():
    lf = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    assert vi.windward_lambda_f(lf, 90.0)[0] == pytest.approx(2.0)
    assert vi.windward_lambda_f(lf, 22.5)[0] == pytest.approx(0.5)
    assert vi.windward_lambda_f(lf, 337.5)[0] == pytest.approx(3.5)


def test_macdonald_hand_computed():
    # lambda_p = 0.25: zd/H = 1 + 4.43**-0.25 * (0.25 - 1) = 0.48304
    # lambda_f = 0.2:  inner = 0.5 * (1.2 / 0.4**2) * (1 - 0.48304) * 0.2 = 0.38773
    #                  z0/H = 0.51696 * exp(-inner**-0.5) = 0.10375
    zd, z0 = vi.macdonald_zd_z0(0.25, 0.2, 10.0)
    assert zd == pytest.approx(4.8304, abs=1e-3)
    assert z0 == pytest.approx(1.0375, abs=1e-3)


def test_macdonald_limits():
    zd, z0 = vi.macdonald_zd_z0(np.array([0.0, 1.0, 0.3]), np.array([0.5, 0.5, 0.0]), np.array([8.0, 8.0, np.nan]))
    assert zd[0] == pytest.approx(0.0)
    assert zd[1] == pytest.approx(8.0)
    assert math.isnan(zd[2]) and math.isnan(z0[2])
    z0b = vi.macdonald_zd_z0(0.3, 0.0, 8.0)[1]
    assert z0b == 0.0


def _obs():
    t = pd.to_datetime(["2026-01-01 12:00", "2026-01-01 13:00", "2026-01-01 14:00"], utc=True)
    return pd.DataFrame({"valid_utc": t, "drct": [100.0, np.nan, 200.0],
                         "speed_ms": [3.0, 0.0, 4.0], "calm": [False, True, False],
                         "variable": [False, False, False]})


def test_wind_at_within_and_beyond_60_min():
    obs = _obs()
    got = wo.wind_at("2026-01-01 12:20", obs=obs)
    assert got["drct"] == 100.0 and got["gap_min"] == pytest.approx(20.0)
    assert wo.wind_at("2026-01-01 12:00", obs=obs)["drct"] == 100.0
    assert wo.wind_at("2026-01-01 15:01", obs=obs) is None
    assert wo.wind_at("2026-01-01 10:59", obs=obs) is None
    assert wo.wind_at("2026-01-01 13:00", obs=obs)["gap_min"] == pytest.approx(60.0)


def test_device_to_utc():
    assert wo.device_to_utc("2026-01-01 09:00", "local") == pd.Timestamp("2026-01-01 12:00", tz="UTC")
    assert wo.device_to_utc("2026-01-01 09:00", "utc") == pd.Timestamp("2026-01-01 09:00", tz="UTC")


def test_rose_sectors_and_exclusions():
    r = wo.rose(np.array([0.0, 359.0, 11.0, 90.0]), np.array([2.0, 4.0, 6.0, 8.0]), 16)
    assert r["frequencies"][0] == pytest.approx(0.75)
    assert r["mean_speed_ms"][0] == pytest.approx(4.0)
    assert r["frequencies"][4] == pytest.approx(0.25)
    assert sum(r["frequencies"]) == pytest.approx(1.0)
    c = wo.campaign_window_rose(obs=_obs())
    assert c["n_directional"] == 2 and c["calm_fraction"] == pytest.approx(1 / 3)


def test_compute_indices_columns_are_proxies():
    n = 2
    pts = pd.DataFrame({"point_id": ["a", "b"], "street_orientation_deg": [0.0, 90.0],
                        "lambda_p_buffer_50m": [0.25, 0.5], "building_height_mean_buffer_50m": [10.0, 6.0],
                        **{c: [0.2] * n for c in vi.LAMBDA_F_COLS}})
    out = vi.compute_indices(pts, 90.0, np.zeros((n, 4)), np.array([0.0, 90.0, 180.0, 270.0]))
    for c in vi.INDEX_COLUMNS:
        assert c.endswith("_proxy") and c in out.columns
    assert out["canyon_alignment_deg_proxy"].tolist() == [90.0, 0.0]
    assert out["open_space_fraction_proxy"].tolist() == [0.75, 0.5]
    assert out["z0_m_proxy"].iloc[0] == pytest.approx(1.0375, abs=1e-3)


def test_climatology_rebin_matches_wind_rose_json():
    import json
    from src.om_package.io_utils import Paths
    p = Paths()
    if not p.wind_rose_json.exists() or not (p.root / "data" / "asos" / "SBGL_2015_2024.csv").exists():
        pytest.skip("climatology data not on disk")
    ref = json.loads(p.wind_rose_json.read_text())["frequencies"]
    r8 = wo.climatology_rose(p.root, n_sectors=8)
    for k, d in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]):
        assert r8["frequencies"][k] == pytest.approx(ref[d], abs=1e-9)
