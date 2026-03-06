"""
Test utilities for SVF algorithm testing.
"""

import numpy as np
import pyvista as pv
import torch
from pathlib import Path
from typing import Tuple, Optional
import json


def create_empty_scene() -> pv.PolyData:
    """Create an empty scene (flat ground plane only)."""
    # Create a simple ground plane
    x = np.linspace(-50, 50, 10)
    y = np.linspace(-50, 50, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Create mesh
    mesh = pv.StructuredGrid(X, Y, Z)
    return mesh.extract_surface()


def create_single_building_scene(
    building_size: Tuple[float, float, float] = (10, 10, 20),
    building_center: Tuple[float, float] = (0, 0)
) -> pv.PolyData:
    """
    Create a scene with a single building (box) on flat ground.
    
    Args:
        building_size: (width, depth, height) of building
        building_center: (x, y) center of building
    
    Returns:
        PyVista mesh with ground and building
    """
    w, d, h = building_size
    cx, cy = building_center
    
    # Ground plane
    x = np.linspace(-50, 50, 10)
    y = np.linspace(-50, 50, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ground = pv.StructuredGrid(X, Y, Z)
    
    # Building (box)
    building = pv.Box(
        bounds=(
            cx - w/2, cx + w/2,
            cy - d/2, cy + d/2,
            0, h
        )
    )
    
    # Combine
    combined = ground.extract_surface() + building
    return combined


def create_two_buildings_scene(
    building1_size: Tuple[float, float, float] = (10, 10, 20),
    building1_center: Tuple[float, float] = (-15, 0),
    building2_size: Tuple[float, float, float] = (10, 10, 15),
    building2_center: Tuple[float, float] = (15, 0)
) -> pv.PolyData:
    """Create a scene with two buildings on flat ground."""
    # Ground plane
    x = np.linspace(-50, 50, 10)
    y = np.linspace(-50, 50, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ground = pv.StructuredGrid(X, Y, Z)
    
    # Building 1
    w1, d1, h1 = building1_size
    cx1, cy1 = building1_center
    building1 = pv.Box(
        bounds=(
            cx1 - w1/2, cx1 + w1/2,
            cy1 - d1/2, cy1 + d1/2,
            0, h1
        )
    )
    
    # Building 2
    w2, d2, h2 = building2_size
    cx2, cy2 = building2_center
    building2 = pv.Box(
        bounds=(
            cx2 - w2/2, cx2 + w2/2,
            cy2 - d2/2, cy2 + d2/2,
            0, h2
        )
    )
    
    # Combine
    combined = ground.extract_surface() + building1 + building2
    return combined


def generate_test_points_grid(
    bounds: Tuple[float, float, float, float],
    spacing: float = 5.0,
    height: float = 0.0
) -> np.ndarray:
    """
    Generate a grid of test points.
    
    Args:
        bounds: (x_min, x_max, y_min, y_max)
        spacing: Grid spacing
        height: Z coordinate for all points
    
    Returns:
        Array of shape (N, 3) with point coordinates
    """
    x_min, x_max, y_min, y_max = bounds
    x_coords = np.arange(x_min, x_max + spacing, spacing)
    y_coords = np.arange(y_min, y_max + spacing, spacing)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        np.full(X.size, height)
    ])
    
    return points


def generate_test_points_random(
    bounds: Tuple[float, float, float, float, float, float],
    n_points: int = 100,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random test points.
    
    Args:
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
        n_points: Number of points
        seed: Random seed
    
    Returns:
        Array of shape (N, 3) with point coordinates
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    points = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(n_points, 3)
    )
    
    return points


def assert_svf_valid(svf_values: np.ndarray, name: str = "SVF"):
    """
    Assert that SVF values are in valid range [0, 1].
    
    Args:
        svf_values: Array of SVF values
        name: Name for error message
    """
    assert np.all(svf_values >= 0), f"{name} contains negative values"
    assert np.all(svf_values <= 1), f"{name} contains values > 1"
    assert not np.any(np.isnan(svf_values)), f"{name} contains NaN values"
    assert not np.any(np.isinf(svf_values)), f"{name} contains Inf values"


def compare_cpu_gpu_results(
    cpu_svf: np.ndarray,
    gpu_svf: np.ndarray,
    abs_tolerance: float = 0.01,
    max_abs_tolerance: float = 0.05,
    correlation_threshold: float = 0.95
) -> dict:
    """
    Compare CPU and GPU SVF results.
    
    Args:
        cpu_svf: CPU-computed SVF values
        gpu_svf: GPU-computed SVF values
        abs_tolerance: Mean absolute difference tolerance
        max_abs_tolerance: Maximum absolute difference tolerance
        correlation_threshold: Minimum correlation coefficient
    
    Returns:
        Dictionary with comparison statistics
    
    Raises:
        AssertionError: If results don't meet thresholds
    """
    assert len(cpu_svf) == len(gpu_svf), "CPU and GPU results have different lengths"
    
    # Calculate differences
    diff = cpu_svf - gpu_svf
    abs_diff = np.abs(diff)
    
    # Statistics
    mean_diff = np.mean(diff)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)
    
    # Correlation
    if len(cpu_svf) > 1 and np.std(cpu_svf) > 0 and np.std(gpu_svf) > 0:
        correlation = np.corrcoef(cpu_svf, gpu_svf)[0, 1]
    else:
        correlation = 1.0 if np.allclose(cpu_svf, gpu_svf) else 0.0
    
    stats = {
        'mean_diff': mean_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'correlation': correlation,
        'n_points': len(cpu_svf)
    }
    
    # Assertions
    assert mean_abs_diff < abs_tolerance, \
        f"Mean absolute difference ({mean_abs_diff:.4f}) exceeds tolerance ({abs_tolerance})"
    
    assert max_abs_diff < max_abs_tolerance, \
        f"Maximum absolute difference ({max_abs_diff:.4f}) exceeds tolerance ({max_abs_tolerance})"
    
    assert correlation >= correlation_threshold, \
        f"Correlation ({correlation:.4f}) below threshold ({correlation_threshold})"
    
    return stats


def save_reference_results(
    test_name: str,
    svf_values: np.ndarray,
    output_dir: Path
):
    """Save reference results for regression testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'test_name': test_name,
        'svf_values': svf_values.tolist(),
        'mean': float(np.mean(svf_values)),
        'std': float(np.std(svf_values)),
        'min': float(np.min(svf_values)),
        'max': float(np.max(svf_values)),
        'n_points': len(svf_values)
    }
    
    output_path = output_dir / f"{test_name}_reference.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return output_path


def load_reference_results(test_name: str, reference_dir: Path) -> dict:
    """Load reference results for regression testing."""
    reference_path = reference_dir / f"{test_name}_reference.json"
    
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference results not found: {reference_path}")
    
    with open(reference_path, 'r') as f:
        results = json.load(f)
    
    return results


def assert_reference_match(
    current_svf: np.ndarray,
    reference_results: dict,
    tolerance: float = 0.01
):
    """Assert current results match reference results."""
    ref_svf = np.array(reference_results['svf_values'])
    
    assert len(current_svf) == len(ref_svf), \
        f"Point count mismatch: {len(current_svf)} vs {len(ref_svf)}"
    
    abs_diff = np.abs(current_svf - ref_svf)
    max_diff = np.max(abs_diff)
    
    assert max_diff < tolerance, \
        f"Results differ from reference: max difference = {max_diff:.4f} (tolerance = {tolerance})"
