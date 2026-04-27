"""Tests for zone-level urban morphology metrics."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from src.urban_morphology import (
    compute_bcr,
    compute_far,
    compute_frontal_area_ratio,
    compute_height_variability,
    compute_street_orientation_entropy,
    compute_zone_metrics,
    create_analysis_zones,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def four_buildings() -> gpd.GeoDataFrame:
    """Four 5x5 m buildings (height=10 m) inside a 100x100 m area."""
    geoms = [
        box(10, 10, 15, 15),
        box(30, 30, 35, 35),
        box(60, 60, 65, 65),
        box(80, 80, 85, 85),
    ]
    return gpd.GeoDataFrame(
        {"height": [10.0, 10.0, 10.0, 10.0]},
        geometry=geoms,
    )


@pytest.fixture
def single_zone() -> gpd.GeoDataFrame:
    """One 100x100 m zone covering the buildings extent."""
    return gpd.GeoDataFrame(
        {"zone_id": [0], "geometry": [box(0, 0, 100, 100)]},
    ).assign(zone_area=lambda df: df.geometry.area)


@pytest.fixture
def mixed_height_buildings() -> gpd.GeoDataFrame:
    """Buildings with varying heights for variability tests."""
    geoms = [
        box(10, 10, 15, 15),
        box(30, 30, 35, 35),
        box(60, 60, 65, 65),
        box(80, 80, 85, 85),
    ]
    return gpd.GeoDataFrame(
        {"height": [5.0, 10.0, 20.0, 30.0]},
        geometry=geoms,
    )


@pytest.fixture
def orthogonal_streets() -> gpd.GeoDataFrame:
    """Streets forming a perfect orthogonal grid (only 0 and 90 degrees)."""
    lines = []
    # Horizontal streets (bearing ~90 deg)
    for y in range(0, 110, 10):
        lines.append(LineString([(0, y), (100, y)]))
    # Vertical streets (bearing ~0 deg)
    for x in range(0, 110, 10):
        lines.append(LineString([(x, 0), (x, 100)]))
    return gpd.GeoDataFrame(geometry=lines)


@pytest.fixture
def random_streets() -> gpd.GeoDataFrame:
    """Streets at many different orientations."""
    rng = np.random.default_rng(42)
    lines = []
    for _ in range(200):
        angle = rng.uniform(0, 2 * math.pi)
        cx, cy = rng.uniform(0, 100, size=2)
        length = 20.0
        dx = length * math.cos(angle)
        dy = length * math.sin(angle)
        lines.append(LineString([(cx, cy), (cx + dx, cy + dy)]))
    return gpd.GeoDataFrame(geometry=lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeBCR:
    def test_known_bcr(
        self, four_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        """4 buildings x 25 m^2 / 10 000 m^2 = 0.01."""
        result = compute_bcr(four_buildings, single_zone)
        assert "bcr" in result.columns
        assert math.isclose(result.loc[0, "bcr"], 0.01, rel_tol=1e-6)

    def test_empty_zone_bcr(self, single_zone: gpd.GeoDataFrame) -> None:
        """No buildings => BCR = 0."""
        empty = gpd.GeoDataFrame(geometry=[], columns=["geometry"])
        result = compute_bcr(empty, single_zone)
        assert result.loc[0, "bcr"] == 0.0


@pytest.mark.fast
class TestComputeFAR:
    def test_known_far(
        self, four_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        """height=10, floor_height=3 => n_floors=3.  FAR = 4*25*3/10000 = 0.03."""
        result = compute_far(four_buildings, single_zone, floor_height=3.0)
        assert "far" in result.columns
        assert math.isclose(result.loc[0, "far"], 0.03, rel_tol=1e-6)

    def test_far_floor_height_10(
        self, four_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        """floor_height=10 => n_floors=1.  FAR = 4*25*1/10000 = 0.01."""
        result = compute_far(four_buildings, single_zone, floor_height=10.0)
        assert math.isclose(result.loc[0, "far"], 0.01, rel_tol=1e-6)


@pytest.mark.fast
class TestComputeHeightVariability:
    def test_uniform_heights(
        self, four_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        """All heights equal => sigma_h ~ 0."""
        result = compute_height_variability(four_buildings, single_zone)
        assert "sigma_h" in result.columns
        assert math.isclose(result.loc[0, "sigma_h"], 0.0, abs_tol=1e-9)

    def test_mixed_heights(
        self, mixed_height_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        """Non-uniform heights => sigma_h > 0."""
        result = compute_height_variability(mixed_height_buildings, single_zone)
        assert result.loc[0, "sigma_h"] > 0.0
        # Verify against numpy
        expected = float(np.std([5.0, 10.0, 20.0, 30.0], ddof=0))
        assert math.isclose(result.loc[0, "sigma_h"], expected, rel_tol=1e-6)


@pytest.mark.fast
class TestComputeStreetOrientationEntropy:
    def test_orthogonal_low_entropy(self, orthogonal_streets: gpd.GeoDataFrame) -> None:
        """Only two distinct orientations => low normalised entropy."""
        entropy = compute_street_orientation_entropy(orthogonal_streets, n_bins=36)
        assert 0.0 < entropy < 0.5

    def test_random_high_entropy(self, random_streets: gpd.GeoDataFrame) -> None:
        """Many orientations => higher normalised entropy."""
        entropy = compute_street_orientation_entropy(random_streets, n_bins=36)
        assert entropy > 0.5

    def test_empty_streets(self) -> None:
        """Empty street network => entropy = 0."""
        empty = gpd.GeoDataFrame(geometry=[], columns=["geometry"])
        assert compute_street_orientation_entropy(empty) == 0.0

    def test_none_streets(self) -> None:
        """None input => entropy = 0."""
        assert compute_street_orientation_entropy(None) == 0.0


@pytest.mark.fast
class TestCreateAnalysisZones:
    def test_grid_covers_extent(self, four_buildings: gpd.GeoDataFrame) -> None:
        zones = create_analysis_zones(four_buildings, method="grid", cell_size=50.0)
        assert "zone_id" in zones.columns
        assert "zone_area" in zones.columns
        assert len(zones) > 0

        # Every building centroid must fall inside at least one zone
        from shapely.ops import unary_union

        zone_union = unary_union(zones.geometry)
        for geom in four_buildings.geometry:
            assert zone_union.contains(geom.centroid)

    def test_grid_cell_area(self, four_buildings: gpd.GeoDataFrame) -> None:
        zones = create_analysis_zones(four_buildings, method="grid", cell_size=25.0)
        expected_area = 25.0 * 25.0
        for area in zones["zone_area"]:
            assert math.isclose(area, expected_area, rel_tol=1e-6)

    def test_hexgrid_returns_polygons(self, four_buildings: gpd.GeoDataFrame) -> None:
        zones = create_analysis_zones(four_buildings, method="hexgrid", cell_size=30.0)
        assert len(zones) > 0
        for geom in zones.geometry:
            assert geom.is_valid
            assert geom.area > 0

    def test_invalid_method_raises(self, four_buildings: gpd.GeoDataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            create_analysis_zones(four_buildings, method="triangles")


@pytest.mark.fast
class TestComputeFrontalAreaRatio:
    def test_frontal_area_nonzero(
        self, four_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        result = compute_frontal_area_ratio(four_buildings, single_zone, wind_dir=0.0)
        assert "lambda_f" in result.columns
        assert result.loc[0, "lambda_f"] > 0.0

    def test_frontal_area_different_winds(
        self, four_buildings: gpd.GeoDataFrame, single_zone: gpd.GeoDataFrame
    ) -> None:
        """Square buildings should give similar lambda_f at 0 and 90 degrees."""
        r0 = compute_frontal_area_ratio(four_buildings, single_zone, wind_dir=0.0)
        r90 = compute_frontal_area_ratio(four_buildings, single_zone, wind_dir=90.0)
        assert math.isclose(r0.loc[0, "lambda_f"], r90.loc[0, "lambda_f"], rel_tol=1e-6)


@pytest.mark.fast
class TestComputeZoneMetrics:
    def test_all_columns_present(
        self,
        four_buildings: gpd.GeoDataFrame,
        single_zone: gpd.GeoDataFrame,
        orthogonal_streets: gpd.GeoDataFrame,
    ) -> None:
        result = compute_zone_metrics(
            four_buildings,
            single_zone,
            streets=orthogonal_streets,
        )
        for col in ["bcr", "far", "sigma_h", "lambda_f", "street_orientation_entropy"]:
            assert col in result.columns, f"Missing column {col}"

    def test_without_streets(
        self,
        four_buildings: gpd.GeoDataFrame,
        single_zone: gpd.GeoDataFrame,
    ) -> None:
        result = compute_zone_metrics(four_buildings, single_zone)
        assert "street_orientation_entropy" in result.columns
        assert np.isnan(result.loc[0, "street_orientation_entropy"])

    def test_values_match_individual(
        self,
        four_buildings: gpd.GeoDataFrame,
        single_zone: gpd.GeoDataFrame,
    ) -> None:
        """Orchestrator values must match individual function outputs."""
        combined = compute_zone_metrics(four_buildings, single_zone)
        bcr_only = compute_bcr(four_buildings, single_zone)
        far_only = compute_far(four_buildings, single_zone, floor_height=3.0)

        assert math.isclose(combined.loc[0, "bcr"], bcr_only.loc[0, "bcr"], rel_tol=1e-9)
        assert math.isclose(combined.loc[0, "far"], far_only.loc[0, "far"], rel_tol=1e-9)
