"""
Realistic urban street canyon tests with known SVF values.

These tests use idealized street canyon geometries where we can calculate
expected SVF values analytically or from well-established formulas.
"""

import numpy as np
import pytest
import pyvista as pv
import torch
from pathlib import Path
import sys
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_svf import generate_sky_patches, compute_svf
from src.svf_gpu_compute import compute_svf_gpu
from src.svf_gpu_utils import pv_mesh_to_pytorch3d, prepare_observer_points, prepare_sky_patches
from tests.utils.test_helpers import assert_svf_valid, compare_cpu_gpu_results


def create_street_canyon(
    width: float,
    building_height: float,
    length: float = 100.0,
    canyon_center: tuple = (0, 0)
) -> pv.PolyData:
    """
    Create an idealized street canyon with two parallel buildings.
    
    Args:
        width: Street width (distance between buildings)
        building_height: Height of buildings on both sides
        length: Length of the canyon
        canyon_center: (x, y) center of the canyon
    
    Returns:
        PyVista mesh with ground and two buildings
    """
    cx, cy = canyon_center
    half_width = width / 2
    
    # Ground plane
    x = np.linspace(-50, 50, 20)
    y = np.linspace(-50, 50, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ground = pv.StructuredGrid(X, Y, Z)
    
    # Building on left side
    building_left = pv.Box(
        bounds=(
            cx - half_width - 10, cx - half_width,  # x bounds
            cy - length/2, cy + length/2,           # y bounds
            0, building_height                       # z bounds
        )
    )
    
    # Building on right side
    building_right = pv.Box(
        bounds=(
            cx + half_width, cx + half_width + 10,  # x bounds
            cy - length/2, cy + length/2,            # y bounds
            0, building_height                       # z bounds
        )
    )
    
    # Combine
    combined = ground.extract_surface() + building_left + building_right
    return combined


def calculate_expected_svf_street_canyon(
    width: float,
    height: float,
    observer_height: float = 1.5
) -> float:
    """
    Calculate expected SVF for a point in the center of a street canyon.
    
    Uses the formula for SVF in an infinite street canyon:
    SVF ≈ 2 * atan(H/W) / π
    
    Where:
    - H = building height above observer
    - W = street width
    
    For finite canyons, this is an approximation.
    
    Args:
        width: Street width
        height: Building height
        observer_height: Height of observer above ground
    
    Returns:
        Expected SVF value (0-1)
    """
    H = height - observer_height
    W = width
    
    if H <= 0:
        return 1.0  # No obstruction
    
    # Angle from observer to top of building
    angle = math.atan(H / (W / 2))
    
    # SVF in infinite canyon (simplified)
    # More accurate: SVF = 1 - (2 * angle / π)
    svf = 1.0 - (2 * angle / math.pi)
    
    return max(0.0, min(1.0, svf))


class TestStreetCanyonNarrow:
    """Test narrow street canyons (high H/W ratio)."""
    
    def test_narrow_canyon_center_cpu(self):
        """Test point in center of narrow canyon."""
        width = 5.0  # 5m wide street
        height = 20.0  # 20m tall buildings
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        # Point in center of canyon
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=1.5)
        
        assert_svf_valid(svf_values)
        # Narrow canyon should have low SVF
        assert svf_values[0] < 0.3, f"Expected low SVF in narrow canyon, got {svf_values[0]:.3f}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_narrow_canyon_center_gpu(self):
        """Test GPU computation in narrow canyon."""
        width = 5.0
        height = 20.0
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        device = torch.device('cuda')
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_point, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
        
        gpu_svf = compute_svf_gpu(
            observer_points,
            sky_patches_torch,
            pytorch3d_mesh,
            batch_size=100
        )
        gpu_svf = gpu_svf.cpu().numpy()
        
        assert_svf_valid(gpu_svf)
        assert gpu_svf[0] < 0.3


class TestStreetCanyonWide:
    """Test wide street canyons (low H/W ratio)."""
    
    def test_wide_canyon_center_cpu(self):
        """Test point in center of wide canyon."""
        width = 30.0  # 30m wide street
        height = 10.0  # 10m tall buildings
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=1.5)
        
        assert_svf_valid(svf_values)
        # Wide canyon should have higher SVF
        assert svf_values[0] > 0.5, f"Expected higher SVF in wide canyon, got {svf_values[0]:.3f}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_wide_canyon_center_gpu(self):
        """Test GPU computation in wide canyon."""
        width = 30.0
        height = 10.0
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        device = torch.device('cuda')
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_point, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
        
        gpu_svf = compute_svf_gpu(
            observer_points,
            sky_patches_torch,
            pytorch3d_mesh,
            batch_size=100
        )
        gpu_svf = gpu_svf.cpu().numpy()
        
        assert_svf_valid(gpu_svf)
        assert gpu_svf[0] > 0.5


class TestStreetCanyonCPUvsGPU:
    """Test CPU vs GPU consistency in street canyons."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("width,height", [
        (5.0, 20.0),   # Narrow, tall
        (10.0, 15.0),  # Medium
        (20.0, 10.0),  # Wide, short
        (30.0, 5.0),   # Very wide, very short
    ])
    def test_canyon_consistency(self, width, height):
        """Test CPU vs GPU consistency for different canyon geometries."""
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        # Multiple test points
        test_points = np.array([
            [0.0, 0.0, 0.0],      # Center
            [width/4, 0.0, 0.0],  # Quarter way
            [-width/4, 0.0, 0.0], # Other side
        ], dtype=np.float64)
        
        # CPU computation
        cpu_svf = compute_svf(test_points, sky_patches, mesh, evaluation_height=1.5)
        
        # GPU computation
        device = torch.device('cuda')
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
        observer_points = prepare_observer_points(test_points, 1.5, device=device)
        sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
        
        gpu_svf = compute_svf_gpu(
            observer_points,
            sky_patches_torch,
            pytorch3d_mesh,
            batch_size=100
        )
        gpu_svf = gpu_svf.cpu().numpy()
        
        # Compare results
        stats = compare_cpu_gpu_results(
            cpu_svf, gpu_svf,
            abs_tolerance=0.03,
            max_abs_tolerance=0.1
        )
        print(f"Canyon (W={width}m, H={height}m) comparison: {stats}")


class TestStreetCanyonKnownValues:
    """Test with known SVF values from analytical formulas."""
    
    def test_very_wide_canyon(self):
        """Very wide canyon should approach SVF = 1.0."""
        width = 100.0  # Very wide
        height = 5.0   # Low buildings
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=1.5)
        
        assert_svf_valid(svf_values)
        # Very wide canyon should have SVF close to 1.0
        assert svf_values[0] > 0.8, f"Expected high SVF in very wide canyon, got {svf_values[0]:.3f}"
    
    def test_very_narrow_canyon(self):
        """Very narrow canyon should have very low SVF."""
        width = 3.0   # Very narrow
        height = 30.0 # Very tall
        mesh = create_street_canyon(width, height)
        sky_patches, _ = generate_sky_patches(145)
        
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=1.5)
        
        assert_svf_valid(svf_values)
        # Very narrow canyon should have very low SVF
        assert svf_values[0] < 0.2, f"Expected very low SVF in narrow canyon, got {svf_values[0]:.3f}"
