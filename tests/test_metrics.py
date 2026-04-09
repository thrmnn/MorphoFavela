"""Comprehensive tests for src/metrics.py."""

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from src.metrics import (
    calculate_basic_metrics,
    calculate_inter_building_distance,
    normalize_height_columns,
    validate_footprints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square(x0: float, y0: float, size: float) -> Polygon:
    """Return an axis-aligned square polygon."""
    return box(x0, y0, x0 + size, y0 + size)


def _make_gdf(geoms, base_height=None, top_height=None, crs="EPSG:32723"):
    """Build a GeoDataFrame with height columns and a projected CRS."""
    data = {"geometry": geoms}
    if base_height is not None:
        data["base_height"] = base_height
    if top_height is not None:
        data["top_height"] = top_height
    gdf = gpd.GeoDataFrame(data, crs=crs)
    return gdf


# ---------------------------------------------------------------------------
# normalize_height_columns
# ---------------------------------------------------------------------------


class TestNormalizeHeightColumns:
    def test_normalize_height_columns_standard(self):
        """GDF with 'base_height' and 'top_height' passes through unchanged."""
        gdf = gpd.GeoDataFrame(
            {"base_height": [0.0], "top_height": [10.0], "geometry": [_square(0, 0, 5)]}
        )
        result = normalize_height_columns(gdf)
        assert "base_height" in result.columns
        assert "top_height" in result.columns
        assert result["base_height"].iloc[0] == 0.0
        assert result["top_height"].iloc[0] == 10.0

    def test_normalize_height_columns_alternative(self):
        """GDF with 'base' and 'altura' gets 'base_height' and 'top_height'."""
        gdf = gpd.GeoDataFrame(
            {"base": [2.0], "altura": [8.0], "geometry": [_square(0, 0, 5)]}
        )
        result = normalize_height_columns(gdf)
        assert "base_height" in result.columns
        assert "top_height" in result.columns
        assert result["base_height"].iloc[0] == 2.0
        # top_height = base + altura
        assert math.isclose(result["top_height"].iloc[0], 10.0, rel_tol=1e-9)

    def test_normalize_missing_columns_raises(self):
        """GDF without any recognised height columns raises ValueError."""
        gdf = gpd.GeoDataFrame({"foo": [1.0], "geometry": [_square(0, 0, 5)]})
        with pytest.raises(ValueError, match="Expected columns"):
            normalize_height_columns(gdf)


# ---------------------------------------------------------------------------
# calculate_basic_metrics
# ---------------------------------------------------------------------------


class TestCalculateBasicMetrics:
    def test_basic_metrics_square_building(self):
        """10x10 square building with height 5 m."""
        gdf = _make_gdf(
            geoms=[_square(0, 0, 10)],
            base_height=[0.0],
            top_height=[5.0],
        )
        result = calculate_basic_metrics(gdf)

        assert math.isclose(result["area"].iloc[0], 100.0, rel_tol=1e-6)
        assert math.isclose(result["perimeter"].iloc[0], 40.0, rel_tol=1e-6)
        assert math.isclose(result["height"].iloc[0], 5.0, rel_tol=1e-6)
        assert math.isclose(result["volume"].iloc[0], 500.0, rel_tol=1e-6)

    def test_basic_metrics_hw_ratio(self):
        """5x10 rectangle with height 20 m gives hw_ratio = 20 / 5 = 4.0."""
        rect = box(0, 0, 5, 10)
        gdf = _make_gdf(
            geoms=[rect],
            base_height=[0.0],
            top_height=[20.0],
        )
        result = calculate_basic_metrics(gdf)

        # width = min bounding-box dimension = 5
        assert math.isclose(result["width"].iloc[0], 5.0, rel_tol=1e-6)
        assert math.isclose(result["hw_ratio"].iloc[0], 4.0, rel_tol=1e-6)

    def test_basic_metrics_with_alternative_columns(self):
        """Using 'base' and 'altura' instead of standard column names."""
        gdf = gpd.GeoDataFrame(
            {
                "base": [0.0],
                "altura": [5.0],
                "geometry": [_square(0, 0, 10)],
            },
            crs="EPSG:32723",
        )
        result = calculate_basic_metrics(gdf)
        assert math.isclose(result["height"].iloc[0], 5.0, rel_tol=1e-6)
        assert math.isclose(result["volume"].iloc[0], 500.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# calculate_inter_building_distance
# ---------------------------------------------------------------------------


class TestInterBuildingDistance:
    def test_inter_building_distance(self):
        """Two 5x5 buildings separated by 10 m gap."""
        # Building A: (0,0)-(5,5), Building B: (15,0)-(20,5)
        a = _square(0, 0, 5)
        b = _square(15, 0, 5)
        gdf = _make_gdf(
            geoms=[a, b],
            base_height=[0.0, 0.0],
            top_height=[5.0, 5.0],
        )
        distances = calculate_inter_building_distance(gdf)

        # The gap between the two boxes is 10 m
        assert math.isclose(distances[0], 10.0, rel_tol=1e-6)
        assert math.isclose(distances[1], 10.0, rel_tol=1e-6)

    def test_single_building(self):
        """Single building: inter-distance should be NaN."""
        gdf = _make_gdf(
            geoms=[_square(0, 0, 10)],
            base_height=[0.0],
            top_height=[5.0],
        )
        distances = calculate_inter_building_distance(gdf)
        assert np.isnan(distances[0])

    def test_adjacent_buildings_zero_distance(self):
        """Two touching buildings have inter-building distance ~0."""
        a = _square(0, 0, 10)
        b = _square(10, 0, 10)
        gdf = _make_gdf(
            geoms=[a, b],
            base_height=[0.0, 0.0],
            top_height=[5.0, 5.0],
        )
        distances = calculate_inter_building_distance(gdf)

        assert math.isclose(distances[0], 0.0, abs_tol=1e-6)
        assert math.isclose(distances[1], 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# validate_footprints
# ---------------------------------------------------------------------------


class TestValidateFootprints:
    def test_validate_footprints_valid(self):
        """Valid GDF passes validation with no issues."""
        gdf = _make_gdf(
            geoms=[_square(0, 0, 10)],
            base_height=[0.0],
            top_height=[5.0],
        )
        is_valid, issues = validate_footprints(gdf)
        assert is_valid
        assert len(issues) == 0

    def test_validate_footprints_returns_issues(self):
        """GDF with negative height should report issues."""
        gdf = _make_gdf(
            geoms=[_square(0, 0, 10)],
            base_height=[10.0],
            top_height=[5.0],  # height = -5 (invalid)
        )
        is_valid, issues = validate_footprints(gdf)
        assert not is_valid
        assert any("height" in issue.lower() for issue in issues)

    def test_validate_footprints_no_crs(self):
        """GDF without CRS should report an issue."""
        gdf = _make_gdf(
            geoms=[_square(0, 0, 10)],
            base_height=[0.0],
            top_height=[5.0],
            crs=None,
        )
        is_valid, issues = validate_footprints(gdf)
        assert not is_valid
        assert any("CRS" in issue or "crs" in issue.lower() for issue in issues)

    def test_validate_footprints_alternative_columns(self):
        """GDF with 'base'/'altura' columns passes column validation."""
        gdf = gpd.GeoDataFrame(
            {
                "base": [0.0],
                "altura": [5.0],
                "geometry": [_square(0, 0, 10)],
            },
            crs="EPSG:32723",
        )
        is_valid, issues = validate_footprints(gdf)
        assert is_valid
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_gdf(self):
        """Empty GDF returns empty GDF, no crash."""
        gdf = gpd.GeoDataFrame(
            {"base_height": [], "top_height": [], "geometry": []},
            crs="EPSG:32723",
        )
        # Set the geometry column explicitly so geopandas recognises it
        gdf = gdf.set_geometry("geometry")
        result = calculate_basic_metrics(gdf)
        assert len(result) == 0

    def test_single_building_full_pipeline(self):
        """Single building through full calculate_basic_metrics pipeline."""
        gdf = _make_gdf(
            geoms=[_square(0, 0, 10)],
            base_height=[0.0],
            top_height=[5.0],
        )
        result = calculate_basic_metrics(gdf)
        assert len(result) == 1
        # Inter-building distance should be NaN for a single building
        assert np.isnan(result["inter_building_distance"].iloc[0])
