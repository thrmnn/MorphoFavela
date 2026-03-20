"""Tests for src/visualize_morphology.py."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

from src.visualize_morphology import (
    plot_cluster_profiles,
    plot_elbow_silhouette,
    plot_lisa_clusters,
    plot_lisa_clusters_panel,
    plot_typology_map,
    plot_zone_metrics_panel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zone_gdf():
    """Synthetic zone GeoDataFrame with metric + LISA columns."""
    rng = np.random.default_rng(42)
    n = 20
    geoms = [box(i * 50, 0, (i + 1) * 50, 50) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "zone_id": range(n),
            "bcr": rng.uniform(0, 0.8, n),
            "far": rng.uniform(0, 3.0, n),
            "sigma_h": rng.uniform(0, 10, n),
            "lambda_f": rng.uniform(0, 0.5, n),
            "lisa_cluster_bcr": rng.choice(["HH", "HL", "LH", "LL", "ns"], n),
            "lisa_cluster_far": rng.choice(["HH", "HL", "LH", "LL", "ns"], n),
            "lisa_cluster_sigma_h": rng.choice(["HH", "ns"], n),
            "cluster": rng.choice([0, 1, 2], n),
        },
        geometry=geoms,
        crs="EPSG:32723",
    )


@pytest.fixture
def profiles_df():
    """Synthetic cluster profiles DataFrame."""
    return pd.DataFrame(
        {
            "bcr": [0.3, 0.6, 0.1],
            "far": [1.2, 2.5, 0.4],
            "sigma_h": [3.0, 7.0, 1.5],
            "count": [10, 5, 15],
        },
        index=pd.Index([0, 1, 2], name="cluster"),
    )


@pytest.fixture
def optimal_k_result():
    """Synthetic find_optimal_k() output."""
    return {
        "k_range": [2, 3, 4, 5],
        "inertia": [100.0, 60.0, 40.0, 35.0],
        "silhouette": [0.4, 0.6, 0.55, 0.5],
        "optimal_k": 3,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotZoneMetricsPanel:
    def test_creates_file(self, zone_gdf, tmp_path):
        out = tmp_path / "panel.png"
        plot_zone_metrics_panel(zone_gdf, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_nan_column(self, zone_gdf, tmp_path):
        zone_gdf["bcr"] = np.nan
        out = tmp_path / "panel_nan.png"
        plot_zone_metrics_panel(zone_gdf, out)
        assert out.exists()

    def test_missing_columns(self, tmp_path):
        """GeoDataFrame with none of the default metric columns."""
        gdf = gpd.GeoDataFrame(
            {"x": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:32723"
        )
        out = tmp_path / "panel_empty.png"
        plot_zone_metrics_panel(gdf, out)
        # Should not raise; file may or may not be created

    def test_with_boundary_clipping(self, zone_gdf, tmp_path):
        boundary = gpd.GeoDataFrame(
            geometry=[box(0, 0, 500, 50)], crs="EPSG:32723"
        )
        out = tmp_path / "panel_clipped.png"
        plot_zone_metrics_panel(zone_gdf, out, boundary_gdf=boundary)
        assert out.exists()

    def test_with_footprints(self, zone_gdf, tmp_path):
        buildings = gpd.GeoDataFrame(
            geometry=[box(10, 10, 20, 20), box(30, 10, 40, 20)],
            crs="EPSG:32723",
        )
        out = tmp_path / "panel_buildings.png"
        plot_zone_metrics_panel(zone_gdf, out, footprints_gdf=buildings)
        assert out.exists()


class TestPlotLisaClusters:
    def test_creates_file(self, zone_gdf, tmp_path):
        out = tmp_path / "lisa.png"
        plot_lisa_clusters(zone_gdf, "lisa_cluster_bcr", out)
        assert out.exists()

    def test_missing_column(self, zone_gdf, tmp_path):
        out = tmp_path / "lisa_missing.png"
        plot_lisa_clusters(zone_gdf, "nonexistent", out)
        # Should not raise

    def test_panel_discovers_columns(self, zone_gdf, tmp_path):
        paths = plot_lisa_clusters_panel(zone_gdf, tmp_path)
        assert len(paths) == 3  # bcr, far, sigma_h
        for p in paths:
            assert p.exists()

    def test_with_boundary_clipping(self, zone_gdf, tmp_path):
        boundary = gpd.GeoDataFrame(
            geometry=[box(0, 0, 500, 50)], crs="EPSG:32723"
        )
        out = tmp_path / "lisa_clipped.png"
        plot_lisa_clusters(zone_gdf, "lisa_cluster_bcr", out, boundary_gdf=boundary)
        assert out.exists()

    def test_with_footprints(self, zone_gdf, tmp_path):
        buildings = gpd.GeoDataFrame(
            geometry=[box(10, 10, 20, 20)], crs="EPSG:32723"
        )
        out = tmp_path / "lisa_buildings.png"
        plot_lisa_clusters(
            zone_gdf, "lisa_cluster_bcr", out, footprints_gdf=buildings
        )
        assert out.exists()


class TestPlotTypologyMap:
    def test_creates_file(self, zone_gdf, tmp_path):
        out = tmp_path / "typology.png"
        plot_typology_map(zone_gdf, out)
        assert out.exists()

    def test_single_cluster(self, zone_gdf, tmp_path):
        zone_gdf["cluster"] = 0
        out = tmp_path / "typology_single.png"
        plot_typology_map(zone_gdf, out)
        assert out.exists()

    def test_missing_column(self, zone_gdf, tmp_path):
        out = tmp_path / "typology_missing.png"
        plot_typology_map(zone_gdf, out, cluster_col="nonexistent")
        # Should not raise


class TestPlotClusterProfiles:
    def test_creates_file(self, profiles_df, tmp_path):
        out = tmp_path / "profiles.png"
        plot_cluster_profiles(profiles_df, out)
        assert out.exists()

    def test_feature_subset(self, profiles_df, tmp_path):
        out = tmp_path / "profiles_sub.png"
        plot_cluster_profiles(profiles_df, out, features=["bcr", "far"])
        assert out.exists()


class TestPlotElbowSilhouette:
    def test_creates_file(self, optimal_k_result, tmp_path):
        out = tmp_path / "elbow.png"
        plot_elbow_silhouette(optimal_k_result, out)
        assert out.exists()


class TestEdgeCases:
    def test_empty_geodataframe(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {"bcr": pd.Series(dtype=float), "cluster": pd.Series(dtype=int)},
            geometry=[],
            crs="EPSG:32723",
        )
        # These should not raise
        plot_zone_metrics_panel(gdf, tmp_path / "empty_panel.png")
        plot_typology_map(gdf, tmp_path / "empty_typology.png")
