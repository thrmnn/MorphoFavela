"""Effective wind-exposure scalar (the pure freq-weighting core)."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_wind_exposure import SECTORS, wind_exposure  # noqa: E402


def _grid(rows):
    return gpd.GeoDataFrame(rows, geometry=[Point(0, 0)] * len(rows), crs="EPSG:31983")


def test_uniform_wind_recovers_mean_of_sectors():
    g = _grid([{f"lambda_f_{s}": v for s, v in zip(SECTORS, range(8))}])
    freq = {s: 1 / 8 for s in SECTORS}
    assert wind_exposure(g, freq)[0] == pytest.approx(np.mean(range(8)))


def test_single_sector_picks_that_sector():
    g = _grid([{f"lambda_f_{s}": (10.0 if s == "E" else 1.0) for s in SECTORS}])
    freq = {s: (1.0 if s == "E" else 0.0) for s in SECTORS}
    assert wind_exposure(g, freq)[0] == pytest.approx(10.0)


def test_is_a_convex_weighting():
    # exposure must lie within [min sector λf, max sector λf] for any valid freq
    g = _grid([{f"lambda_f_{s}": v for s, v in zip(SECTORS, [2, 3, 9, 1, 2, 3, 9, 1])}])
    freq = {"N": .3, "NE": .1, "E": .2, "SE": .05, "S": .05, "SW": .1, "W": .15, "NW": .05}
    e = wind_exposure(g, freq)[0]
    assert 1.0 <= e <= 9.0
