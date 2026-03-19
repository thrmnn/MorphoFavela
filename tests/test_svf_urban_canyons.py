"""
Realistic urban street canyon tests with known SVF values.

These tests use idealized street canyon geometries where we can calculate
expected SVF values analytically or from well-established formulas.
"""

import numpy as np
import pytest
import pyvista as pv
from pathlib import Path
import sys
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional GPU imports
try:
    import torch
    from src.svf_gpu_compute import compute_svf_gpu
    from src.svf_gpu_utils import pv_mesh_to_pytorch3d, prepare_observer_points, prepare_sky_patches
    GPU_AVAILABLE = True
except ImportError:
    torch = None
    GPU_AVAILABLE = False

from src.svf_compute import generate_sky_patches, compute_svf  # noqa: E402
from tests.utils.test_helpers import assert_svf_valid, compare_cpu_gpu_results  # noqa: E402


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


def create_analytical_canyon(
    width: float,
    building_height: float,
    length: float = 200.0
) -> pv.PolyData:
    """
    Create an idealized street canyon geometry for analytical SVF testing.
    
    This creates a 2D canyon approximation with:
    - Two parallel vertical walls at x = ±W/2
    - Buildings infinitely long in the street direction (approximated with long length)
    - Flat ground plane
    - Buildings have identical height H
    
    Args:
        width: Street width W (distance between building façades)
        building_height: Building height H
        length: Length of canyon in y-direction (should be long to approximate infinite)
    
    Returns:
        PyVista mesh with ground and two building walls
    """
    half_width = width / 2
    
    # Create a large ground plane
    x = np.linspace(-100, 100, 50)
    y = np.linspace(-length/2, length/2, int(length/2))
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ground = pv.StructuredGrid(X, Y, Z)
    
    # Building on left side (wall at x = -W/2)
    # Make it a thin wall extending infinitely in y-direction
    building_left = pv.Box(
        bounds=(
            -half_width - 0.1, -half_width,  # x bounds (thin wall)
            -length/2, length/2,              # y bounds (long)
            0, building_height                # z bounds
        )
    )
    
    # Building on right side (wall at x = +W/2)
    building_right = pv.Box(
        bounds=(
            half_width, half_width + 0.1,  # x bounds (thin wall)
            -length/2, length/2,            # y bounds (long)
            0, building_height              # z bounds
        )
    )
    
    # Combine
    combined = ground.extract_surface() + building_left + building_right
    return combined


def calculate_analytical_svf(
    width: float,
    building_height: float,
    observer_height: float = 1.5
) -> float:
    """
    Calculate expected SVF using the analytical formula for urban canyons.
    
    Formula:
    - H_eff = H - h_obs
    - θ = atan(H_eff / (W/2))
    - SVF_expected = cos(θ)
    
    Or equivalently:
    - SVF_expected = 1 / sqrt(1 + (2*H_eff/W)^2)
    
    Args:
        width: Street width W (distance between building façades)
        building_height: Building height H
        observer_height: Observation height h_obs
    
    Returns:
        Expected SVF value (0-1)
    """
    H_eff = building_height - observer_height
    
    if H_eff <= 0:
        return 1.0  # No obstruction above observer
    
    # Calculate horizon elevation angle
    theta = math.atan(H_eff / (width / 2))
    
    # Calculate SVF using cos(θ)
    svf_expected = math.cos(theta)
    
    return svf_expected


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
    
    @pytest.mark.skipif(not GPU_AVAILABLE or not torch.cuda.is_available(), reason="CUDA not available")
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
    
    @pytest.mark.skipif(not GPU_AVAILABLE or not torch.cuda.is_available(), reason="CUDA not available")
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
    
    @pytest.mark.skipif(not GPU_AVAILABLE or not torch.cuda.is_available(), reason="CUDA not available")
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


class TestAnalyticalCanyonSVF:
    """
    Deterministic unit tests for SVF using analytically solvable urban canyon geometries.
    
    These tests verify that the SVF algorithm returns values close to known analytical
    solutions using the formula:
    - H_eff = H - h_obs
    - θ = atan(H_eff / (W/2))
    - SVF_expected = cos(θ)
    
    All tests use:
    - Observation height: h_obs = 1.5 m
    - Tolerance: |SVF_numeric - SVF_expected| < 0.10
      (Note: The analytical formula assumes infinite 2D canyon, while numerical
       computation uses finite 3D geometry, so some difference is expected)
    - Deterministic sky patches (no randomness, fixed number of patches)
    """
    
    # Use a fixed number of sky patches for deterministic results
    # Increased from 145 to 290 for better accuracy in analytical tests
    NUM_SKY_PATCHES = 290
    
    def test_wide_canyon(self):
        """
        Test 1: Wide canyon
        H = 10 m, W = 40 m, h_obs = 1.5 m
        Expected SVF ≈ 0.92
        """
        H = 10.0
        W = 40.0
        h_obs = 1.5
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Calculate expected SVF
        svf_expected = calculate_analytical_svf(W, H, h_obs)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=500.0)
        
        # Observation point at street centerline
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        # Compute SVF
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate
        assert_svf_valid(svf_values)
        svf_numeric = svf_values[0]
        
        # Check against expected value
        # Note: Tolerance is 0.10 because analytical formula assumes infinite 2D canyon,
        # while numerical computation uses finite 3D geometry
        abs_error = abs(svf_numeric - svf_expected)
        assert abs_error < 0.10, \
            f"Wide canyon: SVF_numeric={svf_numeric:.4f}, SVF_expected={svf_expected:.4f}, " \
            f"error={abs_error:.4f} > 0.10"
    
    def test_moderate_canyon(self):
        """
        Test 2: Moderate canyon
        H = 10 m, W = 20 m, h_obs = 1.5 m
        Expected SVF ≈ 0.76
        """
        H = 10.0
        W = 20.0
        h_obs = 1.5
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Calculate expected SVF
        svf_expected = calculate_analytical_svf(W, H, h_obs)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=500.0)
        
        # Observation point at street centerline
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        # Compute SVF
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate
        assert_svf_valid(svf_values)
        svf_numeric = svf_values[0]
        
        # Check against expected value
        abs_error = abs(svf_numeric - svf_expected)
        assert abs_error < 0.10, \
            f"Moderate canyon: SVF_numeric={svf_numeric:.4f}, SVF_expected={svf_expected:.4f}, " \
            f"error={abs_error:.4f} > 0.10"
    
    def test_hw_ratio_one_canyon(self):
        """
        Test 3: H/W = 1 canyon
        H = 10 m, W = 10 m, h_obs = 1.5 m
        Expected SVF ≈ 0.51
        """
        H = 10.0
        W = 10.0
        h_obs = 1.5
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Calculate expected SVF
        svf_expected = calculate_analytical_svf(W, H, h_obs)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=500.0)
        
        # Observation point at street centerline
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        # Compute SVF
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate
        assert_svf_valid(svf_values)
        svf_numeric = svf_values[0]
        
        # Check against expected value
        abs_error = abs(svf_numeric - svf_expected)
        assert abs_error < 0.10, \
            f"H/W=1 canyon: SVF_numeric={svf_numeric:.4f}, SVF_expected={svf_expected:.4f}, " \
            f"error={abs_error:.4f} > 0.10"
    
    def test_deep_canyon(self):
        """
        Test 4: Deep canyon
        H = 20 m, W = 10 m, h_obs = 1.5 m
        Expected SVF ≈ 0.26
        """
        H = 20.0
        W = 10.0
        h_obs = 1.5
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Calculate expected SVF
        svf_expected = calculate_analytical_svf(W, H, h_obs)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=500.0)
        
        # Observation point at street centerline
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        # Compute SVF
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate
        assert_svf_valid(svf_values)
        svf_numeric = svf_values[0]
        
        # Check against expected value
        abs_error = abs(svf_numeric - svf_expected)
        assert abs_error < 0.10, \
            f"Deep canyon: SVF_numeric={svf_numeric:.4f}, SVF_expected={svf_expected:.4f}, " \
            f"error={abs_error:.4f} > 0.10"
    
    def test_very_deep_canyon(self):
        """
        Test 5: Very deep canyon
        H = 40 m, W = 10 m, h_obs = 1.5 m
        Expected SVF ≈ 0.13
        """
        H = 40.0
        W = 10.0
        h_obs = 1.5
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Calculate expected SVF
        svf_expected = calculate_analytical_svf(W, H, h_obs)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=500.0)
        
        # Observation point at street centerline
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        # Compute SVF
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate
        assert_svf_valid(svf_values)
        svf_numeric = svf_values[0]
        
        # Check against expected value
        abs_error = abs(svf_numeric - svf_expected)
        assert abs_error < 0.10, \
            f"Very deep canyon: SVF_numeric={svf_numeric:.4f}, SVF_expected={svf_expected:.4f}, " \
            f"error={abs_error:.4f} > 0.10"
    
    def test_open_sky(self):
        """
        Sanity check: Open sky test
        No geometry, observation height = 1.5 m
        Expected SVF = 1.0
        """
        h_obs = 1.5
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Create empty scene (flat ground only, no buildings)
        from tests.utils.test_helpers import create_empty_scene
        mesh = create_empty_scene()
        
        # Observation point
        test_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        
        # Compute SVF
        svf_values = compute_svf(test_point, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate
        assert_svf_valid(svf_values)
        svf_numeric = svf_values[0]
        
        # Open sky should have SVF very close to 1.0
        # Allow small tolerance due to discretization
        assert abs(svf_numeric - 1.0) < 0.02, \
            f"Open sky: SVF={svf_numeric:.4f} not close to expected 1.0"


def calculate_analytical_svf_at_position(
    x: float,
    width: float,
    building_height: float,
    observer_height: float = 1.5
) -> float:
    """
    Calculate analytical SVF at horizontal position x within an urban canyon.
    
    The SVF at position x is determined by the maximum elevation angle of the two walls:
    - θ_left = atan(H_eff / (x + W/2))
    - θ_right = atan(H_eff / (W/2 - x))
    - θ = max(θ_left, θ_right)
    - SVF(x) = cos(θ)
    
    Args:
        x: Horizontal position within canyon (x=0 is center, negative is left, positive is right)
        width: Street width W (distance between building façades)
        building_height: Building height H
        observer_height: Observation height h_obs
    
    Returns:
        Expected SVF value (0-1) at position x
    """
    H_eff = building_height - observer_height
    
    if H_eff <= 0:
        return 1.0  # No obstruction
    
    half_width = width / 2
    
    # Calculate elevation angles to both walls
    # Left wall is at x = -W/2, so distance from observer at x is (x + W/2)
    # Right wall is at x = +W/2, so distance from observer at x is (W/2 - x)
    theta_left = math.atan(H_eff / (x + half_width))
    theta_right = math.atan(H_eff / (half_width - x))
    
    # The visible sky is limited by the maximum elevation angle
    theta = max(theta_left, theta_right)
    
    # SVF is the cosine of the limiting elevation angle
    svf = math.cos(theta)
    
    return svf


class TestSVFInterpolation:
    """
    Unit tests for spatial interpolation of SVF values between sampling points.
    
    These tests validate that interpolated SVF values are consistent with the
    analytical variation of SVF across an urban canyon. The tests ensure that:
    - SVF field varies correctly across the canyon
    - Interpolation does not introduce significant bias
    - Gradients near façades are handled correctly
    
    Note: Tolerance is 0.15 (larger than base SVF tests) because interpolation
    introduces additional error on top of the numerical SVF computation error.
    This accounts for:
    - Base SVF computation error (~0.06-0.10 from finite 3D vs infinite 2D)
    - Linear interpolation error of a non-linear function (~0.03-0.05)
    """
    
    # Use a fixed number of sky patches for deterministic results
    # Increased for better accuracy in interpolation tests
    NUM_SKY_PATCHES = 580
    INTERPOLATION_TOLERANCE = 0.13  # Tolerance accounts for base SVF error (~0.06-0.10) + interpolation error (~0.02-0.04)
    # Note: Deep canyons with strong gradients may have larger errors due to 3D vs 2D geometry differences
    
    def test_linear_interpolation_moderate_canyon(self):
        """
        Test 1: Linear SVF interpolation in a moderate canyon.
        
        Geometry: H = 20 m, W = 20 m, h_obs = 1.5 m
        Sampling points: x = -5 m, 0 m, +5 m
        Interpolation points: x = -2.5 m, +2.5 m
        """
        H = 20.0
        W = 20.0
        h_obs = 1.5
        
        # Sampling points
        sampling_x = np.array([-5.0, 0.0, 5.0])
        
        # Interpolation points
        interp_x = np.array([-2.5, 2.5])
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=1000.0)
        
        # Compute SVF at sampling points
        sampling_points = np.array([[x, 0.0, 0.0] for x in sampling_x], dtype=np.float64)
        svf_sampled = compute_svf(sampling_points, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate sampled values
        assert_svf_valid(svf_sampled)
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        interp_func = interp1d(sampling_x, svf_sampled, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        
        # Interpolate at intermediate positions
        svf_interpolated = interp_func(interp_x)
        
        # Calculate analytical SVF at interpolation points
        for i, x_interp in enumerate(interp_x):
            svf_expected = calculate_analytical_svf_at_position(x_interp, W, H, h_obs)
            svf_interp = svf_interpolated[i]
            
            abs_error = abs(svf_interp - svf_expected)
            assert abs_error < self.INTERPOLATION_TOLERANCE, \
                f"Moderate canyon interpolation at x={x_interp:.1f}m: " \
                f"SVF_interpolated={svf_interp:.4f}, SVF_expected={svf_expected:.4f}, " \
                f"error={abs_error:.4f} > {self.INTERPOLATION_TOLERANCE}"
    
    def test_strong_gradient_near_building(self):
        """
        Test 2: Strong SVF gradient near building façade.
        
        Geometry: H = 30 m, W = 10 m, h_obs = 1.5 m
        This deep canyon produces strong SVF variation near façades.
        Sampling points: x = -4 m, 0 m, +4 m
        Interpolation points: x = -2 m, +2 m
        """
        H = 30.0
        W = 10.0
        h_obs = 1.5
        
        # Sampling points
        sampling_x = np.array([-4.0, 0.0, 4.0])
        
        # Interpolation points
        interp_x = np.array([-2.0, 2.0])
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=1000.0)
        
        # Compute SVF at sampling points
        sampling_points = np.array([[x, 0.0, 0.0] for x in sampling_x], dtype=np.float64)
        svf_sampled = compute_svf(sampling_points, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate sampled values
        assert_svf_valid(svf_sampled)
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        interp_func = interp1d(sampling_x, svf_sampled, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        
        # Interpolate at intermediate positions
        svf_interpolated = interp_func(interp_x)
        
        # Calculate analytical SVF at interpolation points
        for i, x_interp in enumerate(interp_x):
            svf_expected = calculate_analytical_svf_at_position(x_interp, W, H, h_obs)
            svf_interp = svf_interpolated[i]
            
            abs_error = abs(svf_interp - svf_expected)
            assert abs_error < self.INTERPOLATION_TOLERANCE, \
                f"Deep canyon interpolation at x={x_interp:.1f}m: " \
                f"SVF_interpolated={svf_interp:.4f}, SVF_expected={svf_expected:.4f}, " \
                f"error={abs_error:.4f} > {self.INTERPOLATION_TOLERANCE}"
    
    def test_wide_canyon_smooth_variation(self):
        """
        Test 3: Wide canyon with smooth variation.
        
        Geometry: H = 10 m, W = 40 m, h_obs = 1.5 m
        Sampling points: x = -10 m, 0 m, +10 m
        Interpolation points: x = -5 m, +5 m
        Because the canyon is wide, SVF variation should be small.
        Ensure interpolation remains accurate even when gradients are low.
        """
        H = 10.0
        W = 40.0
        h_obs = 1.5
        
        # Sampling points
        sampling_x = np.array([-10.0, 0.0, 10.0])
        
        # Interpolation points
        interp_x = np.array([-5.0, 5.0])
        
        # Generate deterministic sky patches
        sky_patches, _ = generate_sky_patches(self.NUM_SKY_PATCHES)
        
        # Create canyon geometry (use longer length to better approximate infinite canyon)
        mesh = create_analytical_canyon(W, H, length=1000.0)
        
        # Compute SVF at sampling points
        sampling_points = np.array([[x, 0.0, 0.0] for x in sampling_x], dtype=np.float64)
        svf_sampled = compute_svf(sampling_points, sky_patches, mesh, evaluation_height=h_obs)
        
        # Validate sampled values
        assert_svf_valid(svf_sampled)
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        interp_func = interp1d(sampling_x, svf_sampled, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        
        # Interpolate at intermediate positions
        svf_interpolated = interp_func(interp_x)
        
        # Calculate analytical SVF at interpolation points
        for i, x_interp in enumerate(interp_x):
            svf_expected = calculate_analytical_svf_at_position(x_interp, W, H, h_obs)
            svf_interp = svf_interpolated[i]
            
            abs_error = abs(svf_interp - svf_expected)
            assert abs_error < self.INTERPOLATION_TOLERANCE, \
                f"Wide canyon interpolation at x={x_interp:.1f}m: " \
                f"SVF_interpolated={svf_interp:.4f}, SVF_expected={svf_expected:.4f}, " \
                f"error={abs_error:.4f} > {self.INTERPOLATION_TOLERANCE}"
