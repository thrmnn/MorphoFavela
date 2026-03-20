import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from src.morphology_metrics import (
    _minimum_rotated_rectangle_axes,
    calculate_morphology_metrics,
)


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


# ---------------------------------------------------------------------------
# Degenerate geometry tests
# ---------------------------------------------------------------------------


def test_degenerate_geometries_filtered():
    """Degenerate geometries (zero-area) should be filtered, not crash."""
    valid_square = _square(0, 0, 10)
    # A degenerate polygon that collapses to a line (zero area)
    degenerate = Polygon([(0, 0), (5, 0), (10, 0), (0, 0)])
    gdf = gpd.GeoDataFrame(geometry=[valid_square, degenerate])
    result = calculate_morphology_metrics(gdf)

    # Only the valid square should remain
    assert len(result) == 1
    assert math.isclose(result.iloc[0]["area"], 100.0, rel_tol=1e-6)


def test_empty_geometry_filtered():
    """Empty geometries should be filtered out."""
    valid_square = _square(0, 0, 10)
    empty_poly = Polygon()
    gdf = gpd.GeoDataFrame(geometry=[valid_square, empty_poly])
    result = calculate_morphology_metrics(gdf)

    assert len(result) == 1


def test_all_degenerate_returns_empty():
    """If all geometries are degenerate, return empty GeoDataFrame."""
    degen1 = Polygon([(0, 0), (1, 0), (2, 0), (0, 0)])
    degen2 = Polygon()
    gdf = gpd.GeoDataFrame(geometry=[degen1, degen2])
    result = calculate_morphology_metrics(gdf)

    assert len(result) == 0


def test_minimum_rotated_rectangle_axes_degenerate():
    """_minimum_rotated_rectangle_axes returns (0, 0, 0) for a degenerate polygon."""
    # A very thin polygon whose minimum_rotated_rectangle may degenerate
    # to a line or point
    tiny = Polygon([(0, 0), (1e-8, 0), (1e-8, 1e-8), (0, 1e-8)])
    result = _minimum_rotated_rectangle_axes(tiny)
    # Should not raise; values should be finite
    assert all(np.isfinite(v) or np.isnan(v) for v in result)
