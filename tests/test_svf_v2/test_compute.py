"""Tests for src/svf_v2/compute.py."""

import numpy as np
import pytest

from src.svf_v2.compute import (
    generate_sky_directions,
    compute_svf_raycasting,
)


class TestGenerateSkyDirections:
    def test_all_upper_hemisphere(self):
        dirs = generate_sky_directions(145)
        assert dirs.shape[1] == 3
        assert np.all(dirs[:, 2] > 0), "All directions should point upward"

    def test_approximately_unit_vectors(self):
        dirs = generate_sky_directions(50)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_count_close_to_requested(self):
        dirs = generate_sky_directions(145)
        # Should be within ~30% of requested
        assert 100 < len(dirs) < 250


class TestRaycastingSVF:
    def test_empty_scene_full_sky(self, empty_scene, sky_directions_small):
        """No obstructions -> SVF ~ 1.0."""
        obs = np.array([[5.0, 5.0, 1.5]])
        svf = compute_svf_raycasting(obs, empty_scene, sky_directions_small)
        assert svf[0] == pytest.approx(1.0, abs=0.05)

    def test_under_building_low_svf(self, single_building_scene, sky_directions_small):
        """Point directly under a tall building should have low SVF."""
        # Point at building edge, looking up -- many directions blocked
        obs = np.array([[4.1, 5.0, 0.1]])
        svf = compute_svf_raycasting(obs, single_building_scene, sky_directions_small)
        assert svf[0] < 0.9  # should be partially blocked

    def test_far_from_building_high_svf(self, single_building_scene, sky_directions_small):
        """Point far from building should have high SVF."""
        obs = np.array([[0.5, 0.5, 1.5]])
        svf = compute_svf_raycasting(obs, single_building_scene, sky_directions_small)
        assert svf[0] > 0.5

    def test_with_normals_facade(self, single_building_scene, sky_directions_small):
        """Facade point with normal should only consider forward hemisphere."""
        obs = np.array([[3.9, 5.0, 2.5]])  # just outside west wall
        normals = np.array([[-1.0, 0.0, 0.0]])  # facing west (away from building)
        svf = compute_svf_raycasting(
            obs, single_building_scene, sky_directions_small, normals=normals,
        )
        # Should have good sky view facing away from building
        assert svf[0] > 0.3

    def test_multiple_points(self, empty_scene, sky_directions_small):
        obs = np.array([
            [2.0, 2.0, 1.5],
            [5.0, 5.0, 1.5],
            [8.0, 8.0, 1.5],
        ])
        svf = compute_svf_raycasting(obs, empty_scene, sky_directions_small)
        assert len(svf) == 3
        assert np.all((svf >= 0) & (svf <= 1))
