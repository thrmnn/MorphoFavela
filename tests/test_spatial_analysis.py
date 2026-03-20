"""Tests for src.spatial_analysis — spatial autocorrelation & hotspot analysis."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from src.spatial_analysis import (
    build_spatial_weights,
    compute_getis_ord,
    compute_global_morans_i,
    compute_lisa,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRID_SIZE = 8


def _make_grid(values: np.ndarray) -> gpd.GeoDataFrame:
    """Build an 8x8 grid of unit-square polygons with the given values."""
    geoms = []
    vals = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            geoms.append(box(j, i, j + 1, i + 1))
            vals.append(float(values[i, j]))
    return gpd.GeoDataFrame({"value": vals}, geometry=geoms)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checkerboard_grid() -> gpd.GeoDataFrame:
    """8x8 checkerboard — alternating high / low values (negative Moran's I)."""
    np.random.seed(42)
    values = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            values[i, j] = 100.0 if (i + j) % 2 == 0 else 0.0
    return _make_grid(values)


@pytest.fixture
def clustered_grid() -> gpd.GeoDataFrame:
    """8x8 grid — top half high, bottom half low (positive Moran's I)."""
    np.random.seed(42)
    values = np.zeros((GRID_SIZE, GRID_SIZE))
    values[:4, :] = 100.0  # top half high
    values[4:, :] = 0.0  # bottom half low
    return _make_grid(values)


@pytest.fixture
def random_grid() -> gpd.GeoDataFrame:
    """8x8 grid — spatially random values (Moran's I ≈ 0)."""
    np.random.seed(42)
    values = np.random.rand(GRID_SIZE, GRID_SIZE) * 100.0
    return _make_grid(values)


# ---------------------------------------------------------------------------
# Tests: build_spatial_weights
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestBuildSpatialWeights:
    def test_queen_weights(self, checkerboard_grid: gpd.GeoDataFrame) -> None:
        w = build_spatial_weights(checkerboard_grid, method="queen")
        assert w.n == len(checkerboard_grid)

    def test_knn_weights(self, checkerboard_grid: gpd.GeoDataFrame) -> None:
        w = build_spatial_weights(checkerboard_grid, method="knn", k=4)
        assert w.n == len(checkerboard_grid)

    def test_distance_weights(self, checkerboard_grid: gpd.GeoDataFrame) -> None:
        w = build_spatial_weights(checkerboard_grid, method="distance")
        assert w.n == len(checkerboard_grid)

    def test_invalid_method_raises(self, checkerboard_grid: gpd.GeoDataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported weights method"):
            build_spatial_weights(checkerboard_grid, method="invalid")


# ---------------------------------------------------------------------------
# Tests: compute_global_morans_i
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestGlobalMoransI:
    def test_checkerboard_negative(self, checkerboard_grid: gpd.GeoDataFrame) -> None:
        result = compute_global_morans_i(checkerboard_grid, "value")
        assert result["I"] < 0, "Checkerboard pattern should yield negative Moran's I"
        assert "p_value" in result
        assert "z_score" in result
        assert "expected_I" in result

    def test_clustered_positive(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_global_morans_i(clustered_grid, "value")
        assert result["I"] > 0, "Clustered pattern should yield positive Moran's I"

    def test_random_near_zero(self, random_grid: gpd.GeoDataFrame) -> None:
        result = compute_global_morans_i(random_grid, "value")
        # Random pattern: Moran's I should be close to expected value.
        # Use lenient bounds — small grids can produce noisy results.
        assert -0.5 < result["I"] < 0.5, (
            f"Random pattern Moran's I should be near zero, got {result['I']}"
        )


# ---------------------------------------------------------------------------
# Tests: compute_lisa
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestLISA:
    def test_returns_correct_columns(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_lisa(clustered_grid, "value")
        for col in ("lisa_I", "lisa_p", "lisa_q", "lisa_cluster"):
            assert col in result.columns, f"Missing column: {col}"

    def test_cluster_labels_valid(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_lisa(clustered_grid, "value")
        valid_labels = {"HH", "HL", "LH", "LL", "ns"}
        actual_labels = set(result["lisa_cluster"].unique())
        assert actual_labels.issubset(valid_labels), (
            f"Unexpected cluster labels: {actual_labels - valid_labels}"
        )

    def test_quadrant_values(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_lisa(clustered_grid, "value")
        valid_quads = {1, 2, 3, 4}
        actual_quads = set(result["lisa_q"].unique())
        assert actual_quads.issubset(valid_quads), (
            f"Unexpected quadrant values: {actual_quads - valid_quads}"
        )

    def test_handles_nan_values(self, clustered_grid: gpd.GeoDataFrame) -> None:
        gdf = clustered_grid.copy()
        gdf.loc[0, "value"] = np.nan
        result = compute_lisa(gdf, "value")
        assert len(result) == len(gdf)
        assert result.loc[0, "lisa_cluster"] == "ns"


# ---------------------------------------------------------------------------
# Tests: compute_getis_ord
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestGetisOrd:
    def test_returns_correct_columns(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_getis_ord(clustered_grid, "value")
        for col in ("hotspot_gi", "hotspot_p", "hotspot_class"):
            assert col in result.columns, f"Missing column: {col}"

    def test_hotspot_class_values(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_getis_ord(clustered_grid, "value")
        valid_classes = {-1, 0, 1}
        actual_classes = set(result["hotspot_class"].unique())
        assert actual_classes.issubset(valid_classes), (
            f"Unexpected hotspot classes: {actual_classes - valid_classes}"
        )

    def test_clustered_has_hotspots(self, clustered_grid: gpd.GeoDataFrame) -> None:
        result = compute_getis_ord(clustered_grid, "value")
        # A strongly clustered pattern should produce at least one hotspot
        # or coldspot.
        n_significant = (result["hotspot_class"] != 0).sum()
        assert n_significant > 0, "Clustered grid should have significant hotspots"

    def test_handles_nan_values(self, clustered_grid: gpd.GeoDataFrame) -> None:
        gdf = clustered_grid.copy()
        gdf.loc[0, "value"] = np.nan
        result = compute_getis_ord(gdf, "value")
        assert len(result) == len(gdf)
        # NaN row should be classified as 0 (not significant).
        assert result.loc[0, "hotspot_class"] == 0
