"""
CPU vs GPU consistency tests.
"""

import numpy as np
import pytest
import torch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_compute import generate_sky_patches, compute_svf
# GPU utilities no longer needed - using unified interface
from tests.utils.test_helpers import (
    create_empty_scene,
    create_single_building_scene,
    create_two_buildings_scene,
    generate_test_points_grid,
    generate_test_points_random,
    compare_cpu_gpu_results,
    assert_svf_valid
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCPUGPUConsistency:
    """Test that CPU and GPU produce similar results."""
    
    def test_empty_scene_consistency(self):
        """Test CPU vs GPU on empty scene."""
        mesh = create_empty_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0
        )
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for different implementations)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.1  # Allow 10% max difference
        )
        print(f"Empty scene comparison: {stats}")
    
    def test_single_building_consistency(self):
        """Test CPU vs GPU on single building scene."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_grid(
            bounds=(-30, 30, -30, 30),
            spacing=5.0,
            height=0.0
        )
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for different implementations)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.1  # Allow 10% max difference
        )
        print(f"Single building comparison: {stats}")
    
    def test_two_buildings_consistency(self):
        """Test CPU vs GPU on two buildings scene."""
        mesh = create_two_buildings_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_grid(
            bounds=(-30, 30, -30, 30),
            spacing=5.0,
            height=0.0
        )
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for different implementations)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.1  # Allow 10% max difference
        )
        print(f"Two buildings comparison: {stats}")
    
    def test_random_points_consistency(self):
        """Test CPU vs GPU on random points."""
        mesh = create_single_building_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_random(
            bounds=(-30, 30, -30, 30, 0, 0),
            n_points=50,
            seed=42
        )
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for different implementations)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.1  # Allow 10% max difference
        )
        print(f"Random points comparison: {stats}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestParameterSensitivity:
    """Test consistency across different parameters."""
    
    @pytest.mark.parametrize("n_patches", [36, 72, 145, 290])
    def test_sky_patch_count_consistency(self, n_patches):
        """Test consistency with different sky patch counts."""
        mesh = create_single_building_scene()
        sky_patches, _ = generate_sky_patches(n_patches)
        
        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0
        )
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for different patch counts)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.1  # Allow 10% max difference
        )
        print(f"Patch count {n_patches} comparison: {stats}")
    
    @pytest.mark.parametrize("height", [0.5, 1.5, 2.0])
    def test_evaluation_height_consistency(self, height):
        """Test consistency with different evaluation heights."""
        mesh = create_single_building_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0
        )
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=height
        )
        
        # GPU computation
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=height,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for different implementations)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.1  # Allow 10% max difference
        )
        print(f"Height {height}m comparison: {stats}")
    
    @pytest.mark.parametrize("batch_size", [50, 100, 200])
    def test_batch_size_consistency(self, batch_size):
        """Test that different batch sizes produce same results."""
        mesh = create_single_building_scene()
        sky_patches, _ = generate_sky_patches(145)
        
        test_points = generate_test_points_grid(
            bounds=(-20, 20, -20, 20),
            spacing=5.0,
            height=0.0
        )
        
        # Compute with different batch sizes (using unified interface)
        gpu_svf_1 = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True,
            gpu_batch_size=batch_size
        )
        
        gpu_svf_2 = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True,
            gpu_batch_size=batch_size * 2  # Different batch size
        )
        
        # Results should be identical regardless of batch size
        assert np.allclose(gpu_svf_1, gpu_svf_2, atol=1e-5), \
            "Different batch sizes should produce same results"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEdgeCases:
    """Test edge cases for CPU vs GPU consistency."""
    
    def test_points_at_building_boundary(self):
        """Test points at building boundaries."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Points at building boundary
        test_points = np.array([
            [5, 0, 0],   # At edge
            [-5, 0, 0],  # At opposite edge
            [0, 5, 0],   # At perpendicular edge
            [5.1, 0, 0], # Just outside
            [4.9, 0, 0]  # Just inside
        ])
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (slightly relaxed tolerance for edge cases)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,  # Allow 3% mean difference
            max_abs_tolerance=0.15  # Allow 15% max difference for edge cases
        )
        print(f"Boundary points comparison: {stats}")
    
    def test_points_close_to_surface(self):
        """Test points very close to building surfaces."""
        mesh = create_single_building_scene(
            building_size=(10, 10, 20),
            building_center=(0, 0)
        )
        sky_patches, _ = generate_sky_patches(145)
        
        # Points very close to building
        test_points = np.array([
            [5.01, 0, 0],   # Very close to edge
            [0, 5.01, 0],   # Very close to perpendicular edge
            [0, 0, 0.01]    # Very close to building base
        ])
        
        # CPU computation
        cpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=False
        )
        
        # GPU computation (using unified interface)
        gpu_svf = compute_svf(
            test_points,
            sky_patches,
            mesh,
            evaluation_height=1.5,
            use_gpu=True
        )
        
        # Compare results (may have slightly higher differences near boundaries)
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.02,
            max_abs_tolerance=0.1  # More lenient for edge cases
        )
        print(f"Close to surface comparison: {stats}")
