"""Tests for src/exposure.py."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.exposure import (
    compute_exposure_index,
    compute_zone_building_density,
    compute_zone_solar_deficit,
    compute_zone_svf_deficit,
    plot_exposure_bivariate,
    plot_exposure_panel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_zone() -> gpd.GeoDataFrame:
    """One 100x100 m zone."""
    return gpd.GeoDataFrame(
        {"zone_id": [0], "geometry": [box(0, 0, 100, 100)]},
        crs="EPSG:32723",
    ).assign(zone_area=lambda df: df.geometry.area)


@pytest.fixture
def two_zones() -> gpd.GeoDataFrame:
    """Two adjacent 100x100 m zones."""
    return gpd.GeoDataFrame(
        {
            "zone_id": [0, 1],
            "geometry": [box(0, 0, 100, 100), box(100, 0, 200, 100)],
        },
        crs="EPSG:32723",
    ).assign(zone_area=lambda df: df.geometry.area)


@pytest.fixture
def solar_points() -> gpd.GeoDataFrame:
    """Solar access points with known values in two zones."""
    # Zone 0 gets points at x=50 with 2, 4, 6 hours => mean=4
    # Zone 1 gets points at x=150 with 8, 10 hours => mean=9
    return gpd.GeoDataFrame(
        {
            "solar_hours": [2.0, 4.0, 6.0, 8.0, 10.0],
            "geometry": [
                Point(50, 50),
                Point(30, 30),
                Point(70, 70),
                Point(150, 50),
                Point(150, 80),
            ],
        },
        crs="EPSG:32723",
    )


@pytest.fixture
def svf_points_perfect() -> gpd.GeoDataFrame:
    """SVF points all at 1.0 (perfect sky view)."""
    return gpd.GeoDataFrame(
        {
            "svf": [1.0, 1.0, 1.0, 1.0],
            "geometry": [Point(25, 25), Point(75, 75), Point(50, 50), Point(50, 25)],
        },
        crs="EPSG:32723",
    )


@pytest.fixture
def svf_points_mixed() -> gpd.GeoDataFrame:
    """SVF points with varying values in two zones."""
    return gpd.GeoDataFrame(
        {
            "svf": [0.2, 0.4, 0.8, 1.0],
            "geometry": [
                Point(50, 50),
                Point(30, 30),
                Point(150, 50),
                Point(150, 80),
            ],
        },
        crs="EPSG:32723",
    )


@pytest.fixture
def simple_footprints() -> gpd.GeoDataFrame:
    """Two 10x10 m buildings (height=5 m each) inside zone 0."""
    return gpd.GeoDataFrame(
        {
            "altura": [5.0, 5.0],
            "geometry": [box(10, 10, 20, 20), box(30, 30, 40, 40)],
        },
        crs="EPSG:32723",
    )


@pytest.fixture
def zone_gdf_with_metrics() -> gpd.GeoDataFrame:
    """Zones with pre-computed deficit metrics for index testing."""
    n = 10
    rng = np.random.default_rng(99)
    geoms = [box(i * 50, 0, (i + 1) * 50, 50) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "zone_id": range(n),
            "solar_deficit": rng.uniform(0, 1, n),
            "svf_deficit": rng.uniform(0, 1, n),
            "density_ratio": rng.uniform(0, 20, n),
        },
        geometry=geoms,
        crs="EPSG:32723",
    )


# ---------------------------------------------------------------------------
# Solar deficit tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeZoneSolarDeficit:
    def test_known_values(self, two_zones, solar_points):
        """Zone 0 mean_solar=4, Zone 1 mean_solar=9; reference=90th pctile."""
        result = compute_zone_solar_deficit(two_zones, solar_points)
        assert "mean_solar" in result.columns
        assert "solar_deficit" in result.columns

        zone0 = result.loc[result["zone_id"] == 0].iloc[0]
        zone1 = result.loc[result["zone_id"] == 1].iloc[0]
        assert math.isclose(zone0["mean_solar"], 4.0, rel_tol=1e-6)
        assert math.isclose(zone1["mean_solar"], 9.0, rel_tol=1e-6)

    def test_explicit_reference(self, two_zones, solar_points):
        """With reference_hours=10, zone0 deficit = 1 - 4/10 = 0.6."""
        result = compute_zone_solar_deficit(
            two_zones, solar_points, reference_hours=10.0
        )
        zone0 = result.loc[result["zone_id"] == 0].iloc[0]
        assert math.isclose(zone0["solar_deficit"], 0.6, rel_tol=1e-6)

    def test_min_solar_computed(self, two_zones, solar_points):
        result = compute_zone_solar_deficit(two_zones, solar_points)
        zone0 = result.loc[result["zone_id"] == 0].iloc[0]
        assert math.isclose(zone0["min_solar"], 2.0, rel_tol=1e-6)

    def test_empty_solar_gdf(self, single_zone):
        """Empty solar data => NaN columns, no crash."""
        empty = gpd.GeoDataFrame(
            {"solar_hours": [], "geometry": []},
            crs="EPSG:32723",
        )
        empty = empty.set_geometry("geometry")
        result = compute_zone_solar_deficit(single_zone, empty)
        assert np.isnan(result.loc[0, "mean_solar"])
        assert np.isnan(result.loc[0, "solar_deficit"])

    def test_none_solar_gdf(self, single_zone):
        """None solar data => NaN columns, no crash."""
        result = compute_zone_solar_deficit(single_zone, None)
        assert np.isnan(result.loc[0, "solar_deficit"])

    def test_no_points_in_zone(self, single_zone):
        """Points outside all zones => NaN."""
        outside = gpd.GeoDataFrame(
            {"solar_hours": [5.0], "geometry": [Point(999, 999)]},
            crs="EPSG:32723",
        )
        result = compute_zone_solar_deficit(single_zone, outside)
        assert np.isnan(result.loc[0, "mean_solar"])

    def test_deficit_clamped_to_01(self, single_zone):
        """Very high reference should not produce negative deficit."""
        pts = gpd.GeoDataFrame(
            {"solar_hours": [100.0], "geometry": [Point(50, 50)]},
            crs="EPSG:32723",
        )
        result = compute_zone_solar_deficit(single_zone, pts, reference_hours=50.0)
        assert result.loc[0, "solar_deficit"] == 0.0  # clamped from negative


# ---------------------------------------------------------------------------
# SVF deficit tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeZoneSvfDeficit:
    def test_perfect_svf_zero_deficit(self, single_zone, svf_points_perfect):
        """All SVF=1.0 => deficit=0.0."""
        result = compute_zone_svf_deficit(single_zone, svf_points_perfect)
        assert "mean_svf" in result.columns
        assert "svf_deficit" in result.columns
        assert math.isclose(result.loc[0, "mean_svf"], 1.0, rel_tol=1e-6)
        assert math.isclose(result.loc[0, "svf_deficit"], 0.0, abs_tol=1e-9)

    def test_mixed_svf_values(self, two_zones, svf_points_mixed):
        """Zone 0: mean(0.2, 0.4)=0.3, deficit=0.7; Zone 1: mean(0.8, 1.0)=0.9, deficit=0.1."""
        result = compute_zone_svf_deficit(two_zones, svf_points_mixed)
        zone0 = result.loc[result["zone_id"] == 0].iloc[0]
        zone1 = result.loc[result["zone_id"] == 1].iloc[0]
        assert math.isclose(zone0["mean_svf"], 0.3, rel_tol=1e-6)
        assert math.isclose(zone0["svf_deficit"], 0.7, rel_tol=1e-6)
        assert math.isclose(zone1["mean_svf"], 0.9, rel_tol=1e-6)
        assert math.isclose(zone1["svf_deficit"], 0.1, rel_tol=1e-6)

    def test_median_computed(self, two_zones, svf_points_mixed):
        result = compute_zone_svf_deficit(two_zones, svf_points_mixed)
        assert "median_svf" in result.columns
        zone0 = result.loc[result["zone_id"] == 0].iloc[0]
        assert math.isclose(zone0["median_svf"], 0.3, rel_tol=1e-6)

    def test_empty_svf(self, single_zone):
        """Empty SVF data => NaN, no crash."""
        empty = gpd.GeoDataFrame(
            {"svf": [], "geometry": []},
            crs="EPSG:32723",
        )
        empty = empty.set_geometry("geometry")
        result = compute_zone_svf_deficit(single_zone, empty)
        assert np.isnan(result.loc[0, "svf_deficit"])

    def test_none_svf(self, single_zone):
        """None SVF data => NaN, no crash."""
        result = compute_zone_svf_deficit(single_zone, None)
        assert np.isnan(result.loc[0, "svf_deficit"])


# ---------------------------------------------------------------------------
# Building density tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeZoneBuildingDensity:
    def test_known_density(self, single_zone, simple_footprints):
        """Two 10x10 buildings (h=5) in 100x100 zone.
        built_volume = 2 * 100 * 5 = 1000 m^3.
        density_ratio = 1000 / 10000 = 0.1.
        open_space_ratio = 1 - 200/10000 = 0.98.
        """
        result = compute_zone_building_density(single_zone, simple_footprints)
        assert "built_volume" in result.columns
        assert "open_space_ratio" in result.columns
        assert "density_ratio" in result.columns
        assert math.isclose(result.loc[0, "built_volume"], 1000.0, rel_tol=1e-6)
        assert math.isclose(result.loc[0, "density_ratio"], 0.1, rel_tol=1e-6)
        assert math.isclose(result.loc[0, "open_space_ratio"], 0.98, rel_tol=1e-6)

    def test_empty_footprints(self, single_zone):
        """No footprints => density=0, open_space=1."""
        empty = gpd.GeoDataFrame(
            {"altura": [], "geometry": []},
            crs="EPSG:32723",
        )
        empty = empty.set_geometry("geometry")
        result = compute_zone_building_density(single_zone, empty)
        assert result.loc[0, "built_volume"] == 0.0
        assert result.loc[0, "open_space_ratio"] == 1.0
        assert result.loc[0, "density_ratio"] == 0.0

    def test_none_footprints(self, single_zone):
        """None footprints => zero density, no crash."""
        result = compute_zone_building_density(single_zone, None)
        assert result.loc[0, "density_ratio"] == 0.0

    def test_missing_height_column(self, single_zone):
        """Footprints without height column => defaults to 1.0 m."""
        fp = gpd.GeoDataFrame(
            {"geometry": [box(10, 10, 20, 20)]},
            crs="EPSG:32723",
        )
        result = compute_zone_building_density(single_zone, fp)
        # Volume = 100 m^2 * 1.0 m = 100 m^3
        assert math.isclose(result.loc[0, "built_volume"], 100.0, rel_tol=1e-6)

    def test_building_crossing_zone_boundary(self, two_zones):
        """Building spanning two zones is correctly clipped."""
        fp = gpd.GeoDataFrame(
            {"altura": [10.0], "geometry": [box(90, 10, 110, 30)]},
            crs="EPSG:32723",
        )
        result = compute_zone_building_density(two_zones, fp, height_col="altura")
        # Building is 20x20 = 400 m^2, half in each zone => 200 m^2 each
        zone0 = result.loc[result["zone_id"] == 0].iloc[0]
        zone1 = result.loc[result["zone_id"] == 1].iloc[0]
        assert math.isclose(zone0["built_volume"], 2000.0, rel_tol=1e-3)
        assert math.isclose(zone1["built_volume"], 2000.0, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Exposure index tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeExposureIndex:
    def test_equal_weights(self, zone_gdf_with_metrics):
        metrics = {
            "solar": "solar_deficit",
            "svf": "svf_deficit",
            "density": "density_ratio",
        }
        result = compute_exposure_index(zone_gdf_with_metrics, metrics)
        assert "exposure_index" in result.columns
        # All values should be in [0, 1] (normalised)
        valid = result["exposure_index"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_custom_weights(self, zone_gdf_with_metrics):
        metrics = {"solar": "solar_deficit", "svf": "svf_deficit"}
        weights = {"solar": 3.0, "svf": 1.0}
        result = compute_exposure_index(zone_gdf_with_metrics, metrics, weights=weights)
        assert "exposure_index" in result.columns
        valid = result["exposure_index"].dropna()
        assert len(valid) == len(zone_gdf_with_metrics)

    def test_single_metric(self, zone_gdf_with_metrics):
        """With one metric, exposure_index == min-max normalised column."""
        metrics = {"solar": "solar_deficit"}
        result = compute_exposure_index(zone_gdf_with_metrics, metrics)
        vals = result["solar_deficit"].values
        expected = (vals - vals.min()) / (vals.max() - vals.min())
        np.testing.assert_allclose(result["exposure_index"].values, expected, rtol=1e-6)

    def test_empty_metrics_dict(self, zone_gdf_with_metrics):
        result = compute_exposure_index(zone_gdf_with_metrics, {})
        assert result["exposure_index"].isna().all()

    def test_missing_columns(self, zone_gdf_with_metrics):
        """Requesting columns that don't exist => NaN exposure_index."""
        metrics = {"foo": "nonexistent_col"}
        result = compute_exposure_index(zone_gdf_with_metrics, metrics)
        assert result["exposure_index"].isna().all()

    def test_nan_handling(self, single_zone):
        """Zones with NaN metrics => NaN exposure_index."""
        gdf = single_zone.copy()
        gdf["solar_deficit"] = np.nan
        gdf["svf_deficit"] = 0.5
        metrics = {"solar": "solar_deficit", "svf": "svf_deficit"}
        result = compute_exposure_index(gdf, metrics)
        assert np.isnan(result.loc[0, "exposure_index"])

    def test_constant_column(self):
        """All identical values => normalised to 0 (zero range)."""
        gdf = gpd.GeoDataFrame(
            {
                "zone_id": [0, 1, 2],
                "solar_deficit": [0.5, 0.5, 0.5],
                "svf_deficit": [0.0, 0.5, 1.0],
            },
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
            crs="EPSG:32723",
        )
        metrics = {"solar": "solar_deficit", "svf": "svf_deficit"}
        result = compute_exposure_index(gdf, metrics)
        # solar_deficit is constant -> contributes 0; svf_deficit drives index
        assert result.loc[0, "exposure_index"] < result.loc[2, "exposure_index"]


# ---------------------------------------------------------------------------
# Visualisation tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPlotExposurePanel:
    def test_creates_file(self, zone_gdf_with_metrics, tmp_path):
        out = tmp_path / "exposure_panel.png"
        plot_exposure_panel(zone_gdf_with_metrics, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_custom_metrics(self, zone_gdf_with_metrics, tmp_path):
        out = tmp_path / "custom_panel.png"
        custom = [("solar_deficit", "viridis", "Custom Solar")]
        plot_exposure_panel(zone_gdf_with_metrics, out, metrics_to_plot=custom)
        assert out.exists()

    def test_with_boundary(self, zone_gdf_with_metrics, tmp_path):
        boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 250, 50)], crs="EPSG:32723")
        out = tmp_path / "panel_boundary.png"
        plot_exposure_panel(zone_gdf_with_metrics, out, boundary_gdf=boundary)
        assert out.exists()

    def test_with_footprints(self, zone_gdf_with_metrics, tmp_path):
        buildings = gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20)], crs="EPSG:32723")
        out = tmp_path / "panel_buildings.png"
        plot_exposure_panel(zone_gdf_with_metrics, out, footprints_gdf=buildings)
        assert out.exists()

    def test_no_matching_columns(self, tmp_path):
        """GeoDataFrame without any matching columns."""
        gdf = gpd.GeoDataFrame({"x": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32723")
        out = tmp_path / "panel_empty.png"
        plot_exposure_panel(gdf, out)
        # Should not raise


@pytest.mark.fast
class TestPlotExposureBivariate:
    def test_creates_file(self, zone_gdf_with_metrics, tmp_path):
        out = tmp_path / "bivariate.png"
        plot_exposure_bivariate(
            zone_gdf_with_metrics, out, "solar_deficit", "svf_deficit"
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_missing_column_skips(self, zone_gdf_with_metrics, tmp_path):
        out = tmp_path / "bivariate_missing.png"
        plot_exposure_bivariate(
            zone_gdf_with_metrics, out, "nonexistent", "svf_deficit"
        )
        # Should not raise, file may or may not exist

    def test_with_boundary_and_footprints(self, zone_gdf_with_metrics, tmp_path):
        boundary = gpd.GeoDataFrame(geometry=[box(0, 0, 500, 50)], crs="EPSG:32723")
        buildings = gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20)], crs="EPSG:32723")
        out = tmp_path / "bivariate_full.png"
        plot_exposure_bivariate(
            zone_gdf_with_metrics,
            out,
            "solar_deficit",
            "svf_deficit",
            footprints_gdf=buildings,
            boundary_gdf=boundary,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestEdgeCases:
    def test_empty_zones_no_crash(self):
        """All operations with empty zones should return without error."""
        empty_zones = gpd.GeoDataFrame(
            {"zone_id": [], "zone_area": []},
            geometry=[],
            crs="EPSG:32723",
        )
        empty_zones = empty_zones.set_geometry("geometry")

        pts = gpd.GeoDataFrame(
            {"solar_hours": [5.0], "geometry": [Point(50, 50)]},
            crs="EPSG:32723",
        )
        svf_pts = gpd.GeoDataFrame(
            {"svf": [0.8], "geometry": [Point(50, 50)]},
            crs="EPSG:32723",
        )
        fp = gpd.GeoDataFrame(
            {"altura": [10.0], "geometry": [box(10, 10, 20, 20)]},
            crs="EPSG:32723",
        )

        r1 = compute_zone_solar_deficit(empty_zones, pts)
        assert len(r1) == 0

        r2 = compute_zone_svf_deficit(empty_zones, svf_pts)
        assert len(r2) == 0

        r3 = compute_zone_building_density(empty_zones, fp)
        assert len(r3) == 0

        r4 = compute_exposure_index(empty_zones, {"solar": "solar_deficit"})
        assert len(r4) == 0
