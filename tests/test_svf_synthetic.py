"""
Synthetic scene tests with known ground truth.
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_compute import generate_sky_patches, compute_svf  # noqa: E402
from src.svf_gpu_compute import compute_svf_gpu  # noqa: E402
from src.svf_gpu_utils import pv_mesh_to_pytorch3d, prepare_observer_points, prepare_sky_patches  # noqa: E402
from tests.utils.test_helpers import (  # noqa: E402
    create_empty_scene,
    create_single_building_scene,
    create_two_buildings_scene,
    generate_test_points_grid,
    assert_svf_valid
)
import torch  # noqa: E402


class TestEmptyScene:
    """Test SVF computation on empty scene (no obstructions)."""
    
    def test_empty_scene_cpu(self):
        """Test CPU computation on empty scene."""
        mesh = create_empty_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        # Test points on ground
        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0
        )
        
        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5
        )
        
        # All points should have SVF ≈ 1.0 (no obstructions)
        assert_svf_valid(svf_values)
        assert np.allclose(svf_values, 1.0, atol=0.05), \
            "SVF should be ~1.0 in empty scene"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_empty_scene_gpu(self):
        """Test GPU computation on empty scene."""
        mesh = create_empty_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0
        )
        
        # Convert to GPU format
        device = torch.device('cuda')
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_points, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
        
        svf_values = compute_svf_gpu(
            observer_points,
            sky_patches_torch,
            pytorch3d_mesh,
            batch_size=100
        )
        
        svf_values = svf_values.cpu().numpy()
        
        assert_svf_valid(svf_values)
        assert np.allclose(svf_values, 1.0, atol=0.05), \
            "SVF should be ~1.0 in empty scene"


class TestSingleBuilding:
    """Test SVF computation with single building."""
    
    def test_point_far_from_building_cpu(self):
        """Test point far from building should have high SVF."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Point far from building
        test_points = np.array([[30, 30, 0]])
        
        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5
        )
        
        assert_svf_valid(svf_values)
        # Should have high SVF (building is far away)
        assert svf_values[0] > 0.8, \
            f"Point far from building should have high SVF, got {svf_values[0]:.3f}"
    
    def test_point_under_building_cpu(self):
        """Test point directly under building should have low SVF."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Point directly under building center
        test_points = np.array([[0, 0, 0]])
        
        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5
        )
        
        assert_svf_valid(svf_values)
        # Should have very low SVF (completely blocked)
        assert svf_values[0] < 0.1, \
            f"Point under building should have low SVF, got {svf_values[0]:.3f}"
    
    def test_point_at_building_edge_cpu(self):
        """Test point at building edge should have partial SVF."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Point at building edge (just outside)
        test_points = np.array([[6, 0, 0]])  # Just outside 5m half-width
        
        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5
        )
        
        assert_svf_valid(svf_values)
        # Should have partial SVF (some sky visible, some blocked)
        assert 0.3 < svf_values[0] < 0.8, \
            f"Point at edge should have partial SVF, got {svf_values[0]:.3f}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_building_gpu(self):
        """Test GPU computation with single building."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Multiple test points
        test_points = np.array([
            [30, 30, 0],  # Far from building
            [0, 0, 0],    # Under building
            [6, 0, 0]     # At edge
        ])
        
        device = torch.device('cuda')
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_points, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
        
        svf_values = compute_svf_gpu(
            observer_points,
            sky_patches_torch,
            pytorch3d_mesh,
            batch_size=100
        )
        
        svf_values = svf_values.cpu().numpy()
        
        assert_svf_valid(svf_values)
        # Far point should have high SVF
        assert svf_values[0] > 0.8
        # Under building should have low SVF
        assert svf_values[1] < 0.1
        # Edge point should have partial SVF
        assert 0.3 < svf_values[2] < 0.8


class TestTwoBuildings:
    """Test SVF computation with two buildings."""
    
    def test_point_between_buildings_cpu(self):
        """Test point between two buildings."""
        mesh = create_two_buildings_scene(
            building1_center=(-15, 0),
            building2_center=(15, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Point between buildings
        test_points = np.array([[0, 0, 0]])
        
        svf_values = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5
        )
        
        assert_svf_valid(svf_values)
        # Should have partial SVF (sky visible above, blocked on sides)
        assert 0.2 < svf_values[0] < 0.9, \
            f"Point between buildings should have partial SVF, got {svf_values[0]:.3f}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_two_buildings_gpu(self):
        """Test GPU computation with two buildings."""
        mesh = create_two_buildings_scene(
            building1_center=(-15, 0),
            building2_center=(15, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = np.array([
            [0, 0, 0],    # Between buildings
            [-15, 0, 0],  # Under building 1
            [15, 0, 0]    # Under building 2
        ])
        
        device = torch.device('cuda')
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_points, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
        
        svf_values = compute_svf_gpu(
            observer_points,
            sky_patches_torch,
            pytorch3d_mesh,
            batch_size=100
        )
        
        svf_values = svf_values.cpu().numpy()
        
        assert_svf_valid(svf_values)
        # Points under buildings should have very low SVF
        assert svf_values[1] < 0.1
        assert svf_values[2] < 0.1
