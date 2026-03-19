"""
Integration tests for end-to-end SVF computation.
"""

import numpy as np
import pytest
import torch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_compute import generate_sky_patches, compute_svf  # noqa: E402
from src.svf_gpu_compute import compute_svf_gpu  # noqa: E402
from src.svf_gpu_utils import (  # noqa: E402
    pv_mesh_to_pytorch3d,
    prepare_observer_points,
    prepare_sky_patches,
)
from tests.utils.test_helpers import (  # noqa: E402
    create_single_building_scene,
    generate_test_points_grid,
    assert_svf_valid,
)


class TestEndToEndCPU:
    """Test end-to-end CPU computation pipeline."""

    def test_full_pipeline_cpu(self):
        """Test complete CPU pipeline."""
        # Create scene
        mesh = create_single_building_scene()

        # Generate sky patches
        sky_patches, _ = generate_sky_patches(145)
        assert len(sky_patches) > 0

        # Generate test points
        test_points = generate_test_points_grid(
            bounds=(-30, 30, -30, 30), spacing=5.0, height=0.0
        )
        assert len(test_points) > 0

        # Compute SVF
        svf_values = compute_svf(test_points, sky_patches, mesh, evaluation_height=1.5)

        # Validate results
        assert len(svf_values) == len(test_points)
        assert_svf_valid(svf_values)

        # Check that results are reasonable
        assert np.mean(svf_values) > 0, "Mean SVF should be positive"
        assert np.mean(svf_values) < 1, (
            "Mean SVF should be less than 1 (obstructions present)"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEndToEndGPU:
    """Test end-to-end GPU computation pipeline."""

    def test_full_pipeline_gpu(self):
        """Test complete GPU pipeline."""
        # Create scene
        mesh = create_single_building_scene()

        # Generate sky patches
        sky_patches, _ = generate_sky_patches(145)
        assert len(sky_patches) > 0

        # Generate test points
        test_points = generate_test_points_grid(
            bounds=(-30, 30, -30, 30), spacing=5.0, height=0.0
        )
        assert len(test_points) > 0

        # Convert to GPU format
        device = torch.device("cuda")
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_points, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)

        # Compute SVF
        svf_values = compute_svf_gpu(
            observer_points, sky_patches_torch, pytorch3d_mesh, batch_size=100
        )

        # Convert back to numpy
        svf_values = svf_values.cpu().numpy()

        # Validate results
        assert len(svf_values) == len(test_points)
        assert_svf_valid(svf_values)

        # Check that results are reasonable
        assert np.mean(svf_values) > 0, "Mean SVF should be positive"
        assert np.mean(svf_values) < 1, (
            "Mean SVF should be less than 1 (obstructions present)"
        )


class TestOutputConsistency:
    """Test that outputs are consistent and reproducible."""

    def test_reproducibility_cpu(self):
        """Test that CPU computation is reproducible."""
        mesh = create_single_building_scene()
        sky_patches, _ = generate_sky_patches(145)

        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20), spacing=5.0, height=0.0
        )

        # Run twice
        svf_1 = compute_svf(test_points, sky_patches, mesh, evaluation_height=1.5)

        svf_2 = compute_svf(test_points, sky_patches, mesh, evaluation_height=1.5)

        # Results should be identical
        assert np.allclose(svf_1, svf_2, atol=1e-6), (
            "CPU computation should be reproducible"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reproducibility_gpu(self):
        """Test that GPU computation is reproducible."""
        mesh = create_single_building_scene()
        sky_patches, _ = generate_sky_patches(145)

        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20), spacing=5.0, height=0.0
        )

        device = torch.device("cuda")
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_points, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)

        # Run twice
        svf_1 = compute_svf_gpu(
            observer_points, sky_patches_torch, pytorch3d_mesh, batch_size=100
        )
        svf_1 = svf_1.cpu().numpy()

        svf_2 = compute_svf_gpu(
            observer_points, sky_patches_torch, pytorch3d_mesh, batch_size=100
        )
        svf_2 = svf_2.cpu().numpy()

        # Results should be identical
        assert np.allclose(svf_1, svf_2, atol=1e-5), (
            "GPU computation should be reproducible"
        )
