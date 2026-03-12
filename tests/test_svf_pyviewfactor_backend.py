"""
Tests for the experimental PyViewFactor SVF backend.

For now, the `backend="pyviewfactor"` branch in `src.svf_compute.compute_svf`
delegates to the existing CPU ray-tracing implementation while we incrementally
wire in the true PyViewFactor-based algorithm.

These tests simply ensure that:
- The backend flag is recognized and does not crash.
- Behaviour matches the existing CPU backend on simple synthetic scenes.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_compute import generate_sky_patches, compute_svf  # noqa: E402
from tests.utils.test_helpers import (  # noqa: E402
    create_empty_scene,
    create_single_building_scene,
    generate_test_points_grid,
    assert_svf_valid,
)


@pytest.mark.skipif(
    "pyviewfactor" not in sys.modules and __import__("importlib").util.find_spec("pyviewfactor") is None,  # type: ignore[comparison-overlap]
    reason="pyviewfactor not installed",
)
class TestPyViewFactorBackend:
    """Basic correctness checks for the PyViewFactor backend flag."""

    def test_empty_scene_pyviewfactor_backend(self):
        """In an empty scene, backend='pyviewfactor' should give SVF ≈ 1.0."""
        mesh = create_empty_scene()
        sky_patches, _ = generate_sky_patches(145)

        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0,
        )

        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            backend="pyviewfactor",
        )

        assert_svf_valid(svf_values)
        assert np.allclose(svf_values, 1.0, atol=0.05)

    def test_single_building_pyviewfactor_backend(self):
        """
        With a single building, backend='pyviewfactor' should behave like the CPU backend:
        - Far point: high SVF
        - Under building: low SVF
        - Edge point: partial SVF
        """
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0),
        )
        sky_patches, _ = generate_sky_patches(145)

        test_points = np.array(
            [
                [30, 30, 0],  # Far from building
                [0, 0, 0],  # Under building
                [6, 0, 0],  # At edge
            ]
        )

        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            backend="pyviewfactor",
        )

        assert_svf_valid(svf_values)
        assert svf_values[0] > 0.8
        assert svf_values[1] < 0.1
        assert 0.3 < svf_values[2] < 0.8

