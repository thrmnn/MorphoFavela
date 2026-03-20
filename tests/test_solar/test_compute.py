"""Tests for src/solar/compute.py — ray-casting solar access computation."""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.solar.compute import (
    compute_sunlit_matrix,
    compute_solar_access_streets,
    compute_solar_access_grid,
    _build_obb_tree,
    _ray_hits,
)
from src.solar.sun import sun_position_to_direction


# ===================================================================
# Tests: low-level ray helpers
# ===================================================================


class TestBuildObbTree:
    """Test OBB tree construction."""

    def test_builds_without_error(self, flat_ground_mesh):
        obb = _build_obb_tree(flat_ground_mesh)
        assert obb is not None

    def test_builds_from_box(self, enclosed_mesh):
        obb = _build_obb_tree(enclosed_mesh)
        assert obb is not None


class TestRayHits:
    """Test single-ray intersection."""

    def test_downward_ray_hits_ground(self, flat_ground_mesh):
        obb = _build_obb_tree(flat_ground_mesh)
        origin = np.array([5.0, 5.0, 5.0])
        endpoint = np.array([5.0, 5.0, -5.0])
        assert _ray_hits(obb, origin, endpoint) is True

    def test_upward_ray_misses(self, flat_ground_mesh):
        obb = _build_obb_tree(flat_ground_mesh)
        origin = np.array([5.0, 5.0, 1.0])
        endpoint = np.array([5.0, 5.0, 100.0])
        assert _ray_hits(obb, origin, endpoint) is False

    def test_ray_into_box_hits(self, enclosed_mesh):
        obb = _build_obb_tree(enclosed_mesh)
        origin = np.array([5.0, 5.0, 5.0])
        endpoint = np.array([5.0, 5.0, 15.0])  # toward ceiling
        assert _ray_hits(obb, origin, endpoint) is True


# ===================================================================
# Tests: compute_sunlit_matrix
# ===================================================================


def _make_sun_dirs(*alt_az_pairs):
    """Convert (altitude, azimuth) pairs to an Nx3 direction array."""
    return np.array(
        [sun_position_to_direction(alt, az) for alt, az in alt_az_pairs],
        dtype=np.float64,
    )


class TestSunlitMatrix:
    """Test sunlit boolean matrix computation."""

    def test_open_sky_all_sunlit(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        # Sun directions pointing upward (should not hit flat ground)
        dirs = _make_sun_dirs((90.0, 0.0), (45.0, 90.0))
        result = compute_sunlit_matrix(obs, dirs, flat_ground_mesh)
        assert result.shape == (1, 2)
        assert result.all()  # all True — no obstructions

    def test_enclosed_all_shaded(self, enclosed_mesh):
        obs = np.array([[5.0, 5.0, 5.0]])
        dirs = _make_sun_dirs(
            (45.0, 0.0), (45.0, 90.0), (45.0, 180.0), (45.0, 270.0)
        )
        result = compute_sunlit_matrix(obs, dirs, enclosed_mesh)
        assert not result.any()  # all False — completely enclosed

    def test_matrix_shape(self, flat_ground_mesh):
        obs = np.array([[5, 5, 1.5], [3, 3, 1.5], [8, 8, 1.5]])
        dirs = _make_sun_dirs((45.0, 0.0), (45.0, 90.0), (60.0, 180.0))
        result = compute_sunlit_matrix(obs, dirs, flat_ground_mesh)
        assert result.shape == (3, 3)
        assert result.dtype == bool

    def test_single_point_single_direction(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        dirs = _make_sun_dirs((90.0, 0.0))  # straight up
        result = compute_sunlit_matrix(obs, dirs, flat_ground_mesh)
        assert result.shape == (1, 1)
        assert result[0, 0] is np.bool_(True)


# ===================================================================
# Tests: compute_solar_access_streets
# ===================================================================


class TestComputeSolarAccessStreets:
    """Test street-based solar access function."""

    def test_adds_columns(self, sample_street_gdf, flat_ground_mesh):
        result = compute_solar_access_streets(sample_street_gdf, flat_ground_mesh)
        assert "solar_hours" in result.columns
        assert "n_sun_positions" in result.columns

    def test_preserves_original_columns(self, sample_street_gdf, flat_ground_mesh):
        result = compute_solar_access_streets(sample_street_gdf, flat_ground_mesh)
        assert "street_id" in result.columns
        assert "svf" in result.columns

    def test_empty_gdf(self, flat_ground_mesh):
        empty = gpd.GeoDataFrame(
            columns=["geometry", "z_observer"],
            crs="EPSG:31983",
        )
        result = compute_solar_access_streets(empty, flat_ground_mesh)
        assert len(result) == 0
        assert "solar_hours" in result.columns

    def test_row_count_preserved(self, sample_street_gdf, flat_ground_mesh):
        result = compute_solar_access_streets(sample_street_gdf, flat_ground_mesh)
        assert len(result) == len(sample_street_gdf)

    def test_solar_hours_nonnegative(self, sample_street_gdf, flat_ground_mesh):
        result = compute_solar_access_streets(sample_street_gdf, flat_ground_mesh)
        assert (result["solar_hours"] >= 0).all()

    def test_does_not_mutate_input(self, sample_street_gdf, flat_ground_mesh):
        original_cols = set(sample_street_gdf.columns)
        compute_solar_access_streets(sample_street_gdf, flat_ground_mesh)
        assert set(sample_street_gdf.columns) == original_cols
