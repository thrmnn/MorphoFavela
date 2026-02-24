import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from src.morphology_metrics import calculate_morphology_metrics


def _square(x0, y0, size):
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)])


def test_basic_shape_metrics_square():
    gdf = gpd.GeoDataFrame(geometry=[_square(0, 0, 10)])
    gdf = calculate_morphology_metrics(gdf)

    assert math.isclose(gdf.loc[0, "area"], 100.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "perimeter"], 40.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "longest_axis_length"], 10.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "elongation"], 1.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "rectangularity"], 1.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "square_compactness"], 1.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "convexity"], 1.0, rel_tol=1e-6)


def test_rectangle_elongation():
    rect = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)])
    gdf = gpd.GeoDataFrame(geometry=[rect])
    gdf = calculate_morphology_metrics(gdf)

    assert math.isclose(gdf.loc[0, "elongation"], 2.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[0, "rectangularity"], 1.0, rel_tol=1e-6)


def test_shared_walls_and_adjacency():
    a = _square(0, 0, 10)
    b = _square(10, 0, 10)
    gdf = gpd.GeoDataFrame(geometry=[a, b])
    gdf = calculate_morphology_metrics(gdf)

    assert math.isclose(gdf.loc[0, "shared_walls"], 10.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[1, "shared_walls"], 10.0, rel_tol=1e-6)
    assert gdf.loc[0, "building_adjacency"] == 1
    assert gdf.loc[1, "building_adjacency"] == 1
    assert math.isclose(gdf.loc[0, "perimeter_wall"], 30.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[1, "perimeter_wall"], 30.0, rel_tol=1e-6)


def test_covered_area_ratio():
    a = _square(0, 0, 10)
    b = _square(10, 0, 10)
    gdf = gpd.GeoDataFrame(geometry=[a, b])
    gdf = calculate_morphology_metrics(gdf)

    assert math.isclose(gdf.loc[0, "covered_area_ratio"], 1.0, rel_tol=1e-6)
    assert math.isclose(gdf.loc[1, "covered_area_ratio"], 1.0, rel_tol=1e-6)


def test_voronoi_metrics_present():
    a = _square(0, 0, 10)
    b = _square(30, 0, 10)
    gdf = gpd.GeoDataFrame(geometry=[a, b])
    gdf = calculate_morphology_metrics(gdf)

    assert not np.isnan(gdf.loc[0, "tessellation_area"])
    assert "cwt" in gdf.columns
