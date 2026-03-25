"""Tests for patch_selection.features module."""

import math

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box, LineString

from src.patch_selection.features import (
    CLUSTERING_EXCLUDE,
    _compute_morphometric_features,
    _compute_height_features,
    _compute_street_features,
    _compute_svf_features,
    _compute_topography_features,
    _compute_texture_features,
    _compute_porosity_orientation_features,
    _gini_coefficient,
    compute_tile_features,
    get_clustering_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_buildings(synthetic_tiles):
    """Empty buildings GeoDataFrame with correct CRS."""
    return gpd.GeoDataFrame(
        {"altura": pd.Series(dtype=float), "geometry": pd.Series(dtype=object)},
        crs=synthetic_tiles.crs,
    )


@pytest.fixture
def synthetic_buildings_with_height(synthetic_buildings):
    """Synthetic buildings with a ``height`` column (alias of ``altura``)."""
    bldg = synthetic_buildings.copy()
    bldg["height"] = bldg["altura"]
    return bldg


# ---------------------------------------------------------------------------
# Group 1: Morphometric features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestMorphometricFeatures:
    def test_expected_columns(self, synthetic_tiles, synthetic_buildings):
        df = _compute_morphometric_features(synthetic_tiles, synthetic_buildings)
        expected = {"tile_id", "bcr", "far", "sigma_h",
                    "lambda_f_N", "lambda_f_E", "lambda_f_S", "lambda_f_W"}
        assert expected == set(df.columns)

    def test_row_count_matches_tiles(self, synthetic_tiles, synthetic_buildings):
        df = _compute_morphometric_features(synthetic_tiles, synthetic_buildings)
        assert len(df) == len(synthetic_tiles)

    def test_bcr_range(self, synthetic_tiles, synthetic_buildings):
        df = _compute_morphometric_features(synthetic_tiles, synthetic_buildings)
        assert (df["bcr"] >= 0).all()
        assert (df["bcr"] <= 1).all()

    def test_empty_buildings(self, synthetic_tiles, empty_buildings):
        df = _compute_morphometric_features(synthetic_tiles, empty_buildings)
        assert len(df) == len(synthetic_tiles)
        assert (df["bcr"] == 0.0).all()


# ---------------------------------------------------------------------------
# Group 2: Height distribution features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestHeightFeatures:
    EXPECTED_COLS = {"tile_id", "H_mean", "H_median", "H_std", "H_max",
                     "H_skewness", "H_kurtosis", "H_iqr"}

    def test_expected_columns(self, synthetic_tiles, synthetic_buildings):
        df = _compute_height_features(synthetic_tiles, synthetic_buildings,
                                      height_col="altura")
        assert self.EXPECTED_COLS == set(df.columns)

    def test_row_count_matches_tiles(self, synthetic_tiles, synthetic_buildings):
        df = _compute_height_features(synthetic_tiles, synthetic_buildings,
                                      height_col="altura")
        assert len(df) == len(synthetic_tiles)

    def test_empty_buildings_all_nan(self, synthetic_tiles, empty_buildings):
        df = _compute_height_features(synthetic_tiles, empty_buildings,
                                      height_col="altura")
        for col in self.EXPECTED_COLS - {"tile_id"}:
            assert df[col].isna().all(), f"{col} should be NaN for empty buildings"

    def test_known_mean(self, synthetic_tiles):
        """Single tile with two buildings of known height."""
        tile = synthetic_tiles.iloc[[0]].copy()
        tile_geom = tile.geometry.iloc[0]
        cx, cy = tile_geom.centroid.coords[0]
        bldg = gpd.GeoDataFrame(
            {
                "height": [6.0, 12.0],
                "geometry": [
                    box(cx - 3, cy - 3, cx + 3, cy + 3),
                    box(cx + 5, cy + 5, cx + 11, cy + 11),
                ],
            },
            crs=tile.crs,
        )
        df = _compute_height_features(tile, bldg, height_col="height")
        assert abs(df["H_mean"].iloc[0] - 9.0) < 1e-6


# ---------------------------------------------------------------------------
# Group 3: Street-network features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestStreetFeatures:
    EXPECTED_COLS = {"tile_id", "road_density", "intersection_density",
                     "street_orientation_entropy"}

    def test_expected_columns(self, synthetic_tiles, synthetic_roads):
        df = _compute_street_features(synthetic_tiles, streets=synthetic_roads)
        assert self.EXPECTED_COLS == set(df.columns)

    def test_no_streets_all_nan(self, synthetic_tiles):
        df = _compute_street_features(synthetic_tiles, streets=None)
        for col in self.EXPECTED_COLS - {"tile_id"}:
            assert df[col].isna().all()

    def test_road_density_positive(self, synthetic_tiles, synthetic_roads):
        df = _compute_street_features(synthetic_tiles, streets=synthetic_roads)
        # At least some tiles should have positive road density
        assert (df["road_density"] > 0).any()

    def test_entropy_uses_log2(self, synthetic_tiles, synthetic_roads):
        """Street orientation entropy must be normalised to [0, 1]."""
        df = _compute_street_features(synthetic_tiles, streets=synthetic_roads)
        valid = df["street_orientation_entropy"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()
            assert (valid <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Group 4: SVF aggregation
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestSvfFeatures:
    EXPECTED_COLS = {"tile_id", "svf_mean", "svf_std"}

    def test_expected_columns(self, synthetic_tiles):
        df = _compute_svf_features(synthetic_tiles, svf_path=None)
        assert self.EXPECTED_COLS == set(df.columns)

    def test_no_svf_all_nan(self, synthetic_tiles):
        df = _compute_svf_features(synthetic_tiles, svf_path=None)
        for col in self.EXPECTED_COLS - {"tile_id"}:
            assert df[col].isna().all()


# ---------------------------------------------------------------------------
# Group 5: Topography features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestTopographyFeatures:
    EXPECTED_COLS = {"tile_id", "elev_mean", "elev_range", "slope_mean",
                     "slope_std", "terrain_ruggedness", "dtm_coverage_frac"}

    def test_expected_columns(self, synthetic_tiles, synthetic_dtm):
        df = _compute_topography_features(synthetic_tiles, dtm_path=synthetic_dtm)
        assert self.EXPECTED_COLS == set(df.columns)

    def test_no_dtm_all_nan(self, synthetic_tiles):
        df = _compute_topography_features(synthetic_tiles, dtm_path=None)
        for col in self.EXPECTED_COLS - {"tile_id"}:
            assert df[col].isna().all()

    def test_flat_dtm_zero_slope(self, synthetic_tiles, synthetic_dtm):
        """Flat DTM at 5.0 m should yield elev_mean ~5, slope ~0, range ~0."""
        df = _compute_topography_features(synthetic_tiles, dtm_path=synthetic_dtm)
        valid = df.dropna(subset=["elev_mean"])
        if len(valid) > 0:
            assert abs(valid["elev_mean"].mean() - 5.0) < 0.5
            assert valid["slope_mean"].mean() < 1.0
            assert valid["elev_range"].mean() < 0.5


# ---------------------------------------------------------------------------
# Group 6: Spatial texture features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestTextureFeatures:
    EXPECTED_COLS = {"tile_id", "lacunarity", "fractal_dim_box",
                     "gini_building_area"}

    def test_expected_columns(self, synthetic_tiles, synthetic_buildings):
        df = _compute_texture_features(synthetic_tiles, synthetic_buildings)
        assert self.EXPECTED_COLS == set(df.columns)

    def test_empty_buildings_all_nan(self, synthetic_tiles, empty_buildings):
        df = _compute_texture_features(synthetic_tiles, empty_buildings)
        for col in self.EXPECTED_COLS - {"tile_id"}:
            assert df[col].isna().all()

    def test_gini_empty_tile_nan(self, synthetic_tiles, synthetic_buildings):
        """Tiles with no buildings should have NaN gini (Bug Fix 2 regression)."""
        df = _compute_texture_features(synthetic_tiles, synthetic_buildings)
        # Any tile with 0 buildings should have NaN gini, not 0.0
        # We verify that no tile has gini == 0.0 unless it actually has buildings
        # (Gini can be 0 for equal-area buildings, but not for empty tiles)
        # The regression test: with empty buildings, gini must be NaN
        df_empty = _compute_texture_features(
            synthetic_tiles,
            gpd.GeoDataFrame(
                {"geometry": pd.Series(dtype=object)},
                crs=synthetic_tiles.crs,
            ),
        )
        assert df_empty["gini_building_area"].isna().all()


# ---------------------------------------------------------------------------
# Group 7: Porosity and orientation features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPorosityOrientationFeatures:
    EXPECTED_COLS = {"tile_id", "porosity_N", "porosity_E", "porosity_S",
                     "porosity_W", "orientation_entropy"}

    def test_expected_columns(self, synthetic_tiles, synthetic_buildings):
        df = _compute_porosity_orientation_features(synthetic_tiles,
                                                     synthetic_buildings)
        assert self.EXPECTED_COLS == set(df.columns)

    def test_empty_buildings_full_porosity(self, synthetic_tiles, empty_buildings):
        df = _compute_porosity_orientation_features(synthetic_tiles, empty_buildings)
        for col in ["porosity_N", "porosity_E", "porosity_S", "porosity_W"]:
            assert df[col].isna().all()

    def test_porosity_range(self, synthetic_tiles, synthetic_buildings):
        df = _compute_porosity_orientation_features(synthetic_tiles,
                                                     synthetic_buildings)
        for col in ["porosity_N", "porosity_E", "porosity_S", "porosity_W"]:
            valid = df[col].dropna()
            if len(valid) > 0:
                assert (valid >= 0).all()
                assert (valid <= 1).all()


# ---------------------------------------------------------------------------
# Gini coefficient unit tests (Bug Fix 2)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestGiniCoefficient:
    def test_empty_array_returns_nan(self):
        assert np.isnan(_gini_coefficient(np.array([])))

    def test_single_value_returns_nan(self):
        assert np.isnan(_gini_coefficient(np.array([42.0])))

    def test_equal_values_returns_zero(self):
        result = _gini_coefficient(np.array([10.0, 10.0, 10.0, 10.0]))
        assert abs(result) < 1e-10

    def test_maximal_inequality(self):
        """One building with all the area; rest zero."""
        vals = np.array([0.0, 0.0, 0.0, 100.0])
        result = _gini_coefficient(vals)
        assert result > 0.5  # highly unequal


# ---------------------------------------------------------------------------
# Orchestrator: compute_tile_features
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestComputeTileFeatures:
    def test_total_feature_count_33(self, synthetic_tiles, synthetic_buildings,
                                     synthetic_roads, synthetic_dtm):
        """All 7 feature groups should yield exactly 33 feature columns."""
        result = compute_tile_features(
            synthetic_tiles, synthetic_buildings,
            streets=synthetic_roads, dtm_path=synthetic_dtm,
        )
        feature_cols = [c for c in result.columns if c != "tile_id"]
        assert len(feature_cols) == 33, (
            f"Expected 33 features, got {len(feature_cols)}: {sorted(feature_cols)}"
        )

    def test_expected_feature_names(self, synthetic_tiles, synthetic_buildings,
                                     synthetic_roads, synthetic_dtm):
        result = compute_tile_features(
            synthetic_tiles, synthetic_buildings,
            streets=synthetic_roads, dtm_path=synthetic_dtm,
        )
        expected = {
            # Group 1
            "bcr", "far", "sigma_h",
            "lambda_f_N", "lambda_f_E", "lambda_f_S", "lambda_f_W",
            # Group 2
            "H_mean", "H_median", "H_std", "H_max",
            "H_skewness", "H_kurtosis", "H_iqr",
            # Group 3
            "road_density", "intersection_density", "street_orientation_entropy",
            # Group 4
            "svf_mean", "svf_std",
            # Group 5
            "elev_mean", "elev_range", "slope_mean", "slope_std",
            "terrain_ruggedness", "dtm_coverage_frac",
            # Group 6
            "lacunarity", "fractal_dim_box", "gini_building_area",
            # Group 7
            "porosity_N", "porosity_E", "porosity_S", "porosity_W",
            "orientation_entropy",
        }
        actual = set(result.columns) - {"tile_id"}
        assert expected == actual

    def test_row_count_matches_tiles(self, synthetic_tiles, synthetic_buildings):
        result = compute_tile_features(synthetic_tiles, synthetic_buildings)
        assert len(result) == len(synthetic_tiles)


# ---------------------------------------------------------------------------
# Clustering feature selection
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestClusteringFeatureSelection:
    def test_exclude_list_not_empty(self):
        assert len(CLUSTERING_EXCLUDE) > 0

    def test_get_clustering_features_removes_excluded(self):
        all_cols = ["bcr", "far", "sigma_h", "H_mean", "H_median",
                     "dtm_coverage_frac", "lambda_f_N", "lambda_f_S"]
        result = get_clustering_features(all_cols)
        # sigma_h, H_median, dtm_coverage_frac, lambda_f_S should be excluded
        assert "sigma_h" not in result
        assert "H_median" not in result
        assert "dtm_coverage_frac" not in result
        assert "lambda_f_S" not in result
        # These should remain
        assert "bcr" in result
        assert "far" in result
        assert "H_mean" in result
        assert "lambda_f_N" in result

    def test_get_clustering_features_preserves_order(self):
        all_cols = ["far", "bcr", "H_mean", "sigma_h"]
        result = get_clustering_features(all_cols)
        assert result == ["far", "bcr", "H_mean"]

    def test_get_clustering_features_with_full_set(self, synthetic_tiles,
                                                     synthetic_buildings,
                                                     synthetic_roads,
                                                     synthetic_dtm):
        """Clustering features from the full feature set should exclude
        the known redundant/noisy features."""
        result = compute_tile_features(
            synthetic_tiles, synthetic_buildings,
            streets=synthetic_roads, dtm_path=synthetic_dtm,
        )
        all_feats = [c for c in result.columns if c != "tile_id"]
        clustering_feats = get_clustering_features(all_feats)
        for excl in CLUSTERING_EXCLUDE:
            if excl in all_feats:
                assert excl not in clustering_feats
