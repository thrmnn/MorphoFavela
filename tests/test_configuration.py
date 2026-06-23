"""Configuration metrics — party-wall ratio on touching vs detached buildings."""

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from src.morphometry.configuration import aggregate_to_grid, party_wall_ratio

CRS = "EPSG:31983"


def test_detached_building_has_zero_party_wall():
    b = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=CRS)
    r = party_wall_ratio(b, snap=0.0)
    assert r[0] < 0.02  # isolated → no shared wall


def test_touching_buildings_share_a_wall():
    # two 10×10 boxes sharing the x=10 edge → each shares 10 of 40 m perimeter = 0.25
    b = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=CRS)
    r = party_wall_ratio(b, snap=0.1)
    assert np.all(r > 0.2)
    assert np.all(r < 0.35)


def test_fully_enclosed_courtyard_building_high_ratio():
    # a small box surrounded on 3 sides shares more of its perimeter
    inner = box(10, 0, 20, 10)
    left, right, top = box(0, 0, 10, 10), box(20, 0, 30, 10), box(10, 10, 20, 20)
    b = gpd.GeoDataFrame(geometry=[inner, left, right, top], crs=CRS)
    r = party_wall_ratio(b, snap=0.1)
    assert r[0] > 0.7  # inner shares 3 of 4 walls


def test_aggregate_to_grid_area_weighted_and_nulls():
    b = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=CRS)
    grid = gpd.GeoDataFrame(
        {"zone_id": ["A", "B"]},
        geometry=[box(0, 0, 20, 10), box(100, 100, 120, 110)], crs=CRS)
    out = aggregate_to_grid(b, grid, party_wall_ratio(b, snap=0.1)).set_index("zone_id")
    assert out.loc["A", "n_buildings_cfg"] == 2 and bool(out.loc["A", "has_config"])
    assert out.loc["B", "n_buildings_cfg"] == 0 and not bool(out.loc["B", "has_config"])
    assert np.isnan(out.loc["B", "party_wall_ratio"])
