"""Tests for src/typology.py — settlement typology classification."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.typology import (
    classify_typology,
    compute_cluster_profiles,
    find_optimal_k,
    flag_zones,
    normalize_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_gdf() -> gpd.GeoDataFrame:
    """Synthetic GeoDataFrame with 3 well-separated clusters (30 points)."""
    np.random.seed(42)
    n_per_cluster = 10

    # Cluster A centred at (0, 0)
    a_x = np.random.normal(0, 0.3, n_per_cluster)
    a_y = np.random.normal(0, 0.3, n_per_cluster)

    # Cluster B centred at (10, 10)
    b_x = np.random.normal(10, 0.3, n_per_cluster)
    b_y = np.random.normal(10, 0.3, n_per_cluster)

    # Cluster C centred at (20, 0)
    c_x = np.random.normal(20, 0.3, n_per_cluster)
    c_y = np.random.normal(20, 0.3, n_per_cluster)

    feat_a = np.concatenate([a_x, b_x, c_x])
    feat_b = np.concatenate([a_y, b_y, c_y])

    geometry = [Point(x, y).buffer(0.5) for x, y in zip(feat_a, feat_b)]

    return gpd.GeoDataFrame(
        {"feat_a": feat_a, "feat_b": feat_b},
        geometry=geometry,
    )


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestNormalizeFeatures:
    def test_zscore(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        result = normalize_features(synthetic_gdf, ["feat_a", "feat_b"], method="zscore")
        for col in ["feat_a", "feat_b"]:
            vals = result[col].to_numpy()
            assert abs(np.mean(vals)) < 1e-10, f"zscore mean of {col} should be ~0"
            assert abs(np.std(vals) - 1.0) < 1e-10, f"zscore std of {col} should be ~1"

    def test_minmax(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        result = normalize_features(synthetic_gdf, ["feat_a", "feat_b"], method="minmax")
        for col in ["feat_a", "feat_b"]:
            vals = result[col].to_numpy()
            assert vals.min() == pytest.approx(0.0, abs=1e-10)
            assert vals.max() == pytest.approx(1.0, abs=1e-10)

    def test_original_unchanged(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        original_vals = synthetic_gdf["feat_a"].to_numpy().copy()
        normalize_features(synthetic_gdf, ["feat_a", "feat_b"])
        np.testing.assert_array_equal(synthetic_gdf["feat_a"].to_numpy(), original_vals)

    def test_nan_handling(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        gdf = synthetic_gdf.copy()
        gdf.loc[0, "feat_a"] = np.nan
        result = normalize_features(gdf, ["feat_a", "feat_b"], method="zscore")
        assert not result["feat_a"].isna().any(), "NaN should be imputed before normalizing"


# ---------------------------------------------------------------------------
# classify_typology
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestClassifyTypology:
    def test_kmeans_three_clusters(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        result = classify_typology(synthetic_gdf, ["feat_a", "feat_b"], n_clusters=3)
        assert "cluster" in result.columns
        labels = result["cluster"].to_numpy()
        assert set(labels) == {0, 1, 2}

        # With well-separated data, each cluster should have exactly 10 members
        counts = pd.Series(labels).value_counts()
        assert all(c == 10 for c in counts), f"Expected 10 per cluster, got {counts.to_dict()}"

    def test_hierarchical(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        result = classify_typology(
            synthetic_gdf, ["feat_a", "feat_b"], n_clusters=3, method="hierarchical"
        )
        labels = result["cluster"].to_numpy()
        assert len(set(labels)) == 3

    def test_original_unchanged(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        assert "cluster" not in synthetic_gdf.columns
        classify_typology(synthetic_gdf, ["feat_a", "feat_b"], n_clusters=3)
        assert "cluster" not in synthetic_gdf.columns


# ---------------------------------------------------------------------------
# find_optimal_k
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestFindOptimalK:
    def test_optimal_k_is_three(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        X = synthetic_gdf[["feat_a", "feat_b"]].to_numpy()
        result = find_optimal_k(X, k_range=range(2, 7))

        assert result["optimal_k"] == 3
        assert len(result["k_range"]) == 5
        assert len(result["inertia"]) == 5
        assert len(result["silhouette"]) == 5

    def test_silhouette_values_valid(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        X = synthetic_gdf[["feat_a", "feat_b"]].to_numpy()
        result = find_optimal_k(X, k_range=range(2, 5))
        for s in result["silhouette"]:
            assert -1.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# flag_zones
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestFlagZones:
    def test_default_thresholds_missing_columns(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        """Default columns (svf_mean, bcr, sigma_h) are absent — flags should be False."""
        result = flag_zones(synthetic_gdf)
        assert "flag_svf_mean" in result.columns
        assert "flag_bcr" in result.columns
        assert "flag_sigma_h" in result.columns
        assert "flag_count" in result.columns
        assert (result["flag_count"] == 0).all()

    def test_custom_thresholds(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        thresholds = {
            "feat_a": ("gt", 5.0),
            "feat_b": ("lt", 5.0),
        }
        result = flag_zones(synthetic_gdf, thresholds=thresholds)

        # feat_a > 5 should flag clusters B and C (values ~10 and ~20)
        assert result["flag_feat_a"].sum() > 0
        # feat_b < 5 should flag clusters A and C (values ~0 and ~0)
        assert result["flag_feat_b"].sum() > 0
        # flag_count should be 0, 1, or 2
        assert result["flag_count"].max() <= 2

    def test_original_unchanged(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        flag_zones(synthetic_gdf)
        assert "flag_count" not in synthetic_gdf.columns


# ---------------------------------------------------------------------------
# compute_cluster_profiles
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeClusterProfiles:
    def test_shape_and_count(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        gdf = classify_typology(synthetic_gdf, ["feat_a", "feat_b"], n_clusters=3)
        profiles = compute_cluster_profiles(gdf, ["feat_a", "feat_b"])

        assert profiles.shape == (3, 3)  # 3 clusters × (feat_a, feat_b, count)
        assert "count" in profiles.columns
        assert profiles["count"].sum() == 30

    def test_mean_values_reasonable(self, synthetic_gdf: gpd.GeoDataFrame) -> None:
        gdf = classify_typology(synthetic_gdf, ["feat_a", "feat_b"], n_clusters=3)
        profiles = compute_cluster_profiles(gdf, ["feat_a", "feat_b"])

        # Each cluster mean for feat_a should be near 0, 10, or 20
        means = sorted(profiles["feat_a"].tolist())
        assert means[0] == pytest.approx(0.0, abs=1.0)
        assert means[1] == pytest.approx(10.0, abs=1.0)
        assert means[2] == pytest.approx(20.0, abs=1.0)
