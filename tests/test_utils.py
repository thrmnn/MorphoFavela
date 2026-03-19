"""
Test utilities for SVF testing.
"""

import numpy as np
from scipy import stats


def assert_svf_valid(svf_values, tolerance=1e-6):
    """
    Assert that SVF values are valid (in range [0, 1] and not NaN).

    Args:
        svf_values: Array of SVF values
        tolerance: Tolerance for boundary checks
    """
    assert np.all(np.isfinite(svf_values)), "SVF values must be finite"
    assert np.all(svf_values >= -tolerance), (
        f"SVF values must be >= 0, found min: {np.min(svf_values)}"
    )
    assert np.all(svf_values <= 1 + tolerance), (
        f"SVF values must be <= 1, found max: {np.max(svf_values)}"
    )


def compare_cpu_gpu_results(
    cpu_svf, gpu_svf, tolerance=0.01, correlation_threshold=0.95
):
    """
    Compare CPU and GPU SVF results and return statistics.

    Args:
        cpu_svf: CPU-computed SVF values
        gpu_svf: GPU-computed SVF values
        tolerance: Maximum acceptable mean absolute difference
        correlation_threshold: Minimum acceptable correlation

    Returns:
        Dictionary with comparison statistics

    Raises:
        AssertionError: If results don't meet thresholds
    """
    cpu_svf = np.asarray(cpu_svf).flatten()
    gpu_svf = np.asarray(gpu_svf).flatten()

    assert len(cpu_svf) == len(gpu_svf), "CPU and GPU results must have same length"

    # Calculate differences
    diff = cpu_svf - gpu_svf
    abs_diff = np.abs(diff)

    mean_diff = np.mean(diff)
    mean_abs_diff = np.mean(abs_diff)
    max_abs_diff = np.max(abs_diff)

    # Calculate correlation
    if len(cpu_svf) > 1 and np.std(cpu_svf) > 0 and np.std(gpu_svf) > 0:
        correlation = np.corrcoef(cpu_svf, gpu_svf)[0, 1]
    else:
        correlation = 1.0 if np.allclose(cpu_svf, gpu_svf) else 0.0

    # Kolmogorov-Smirnov test for distribution similarity
    if len(cpu_svf) > 10:
        ks_statistic, ks_pvalue = stats.ks_2samp(cpu_svf, gpu_svf)
    else:
        ks_statistic, ks_pvalue = np.nan, np.nan

    stats_dict = {
        "mean_diff": mean_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "correlation": correlation,
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "n_points": len(cpu_svf),
    }

    # Assertions
    assert mean_abs_diff < tolerance, (
        f"Mean absolute difference ({mean_abs_diff:.4f}) exceeds tolerance ({tolerance})"
    )

    assert correlation >= correlation_threshold, (
        f"Correlation ({correlation:.4f}) below threshold ({correlation_threshold})"
    )

    return stats_dict


def create_synthetic_mesh(mesh_type="empty", **kwargs):
    """
    Create synthetic meshes for testing.

    Args:
        mesh_type: Type of mesh ('empty', 'single_building', 'two_buildings')
        **kwargs: Additional parameters

    Returns:
        PyVista PolyData mesh
    """
    import pyvista as pv

    if mesh_type == "empty":
        # Simple flat plane
        x = np.linspace(-10, 10, 21)
        y = np.linspace(-10, 10, 21)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        n = len(x)
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                faces.extend(
                    [[3, idx, idx + 1, idx + n], [3, idx + 1, idx + n + 1, idx + n]]
                )

        return pv.PolyData(points, faces)

    elif mesh_type == "single_building":
        # Terrain + single building
        size = kwargs.get("size", 20)
        building_size = kwargs.get("building_size", 4)
        building_height = kwargs.get("building_height", 5)

        x = np.linspace(-size, size, int(2 * size) + 1)
        y = np.linspace(-size, size, int(2 * size) + 1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        terrain_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Building corners
        half = building_size / 2
        building_verts = [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
            [-half, -half, building_height],
            [half, -half, building_height],
            [half, half, building_height],
            [-half, half, building_height],
        ]

        # Terrain faces
        n = len(x)
        terrain_faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                terrain_faces.extend(
                    [[3, idx, idx + 1, idx + n], [3, idx + 1, idx + n + 1, idx + n]]
                )

        # Building faces
        n_terrain = len(terrain_points)
        building_faces = [
            [4, n_terrain + 0, n_terrain + 1, n_terrain + 2, n_terrain + 3],  # bottom
            [4, n_terrain + 4, n_terrain + 7, n_terrain + 6, n_terrain + 5],  # top
            [4, n_terrain + 0, n_terrain + 4, n_terrain + 5, n_terrain + 1],  # front
            [4, n_terrain + 2, n_terrain + 6, n_terrain + 7, n_terrain + 3],  # back
            [4, n_terrain + 0, n_terrain + 3, n_terrain + 7, n_terrain + 4],  # left
            [4, n_terrain + 1, n_terrain + 5, n_terrain + 6, n_terrain + 2],  # right
        ]

        all_points = np.vstack([terrain_points, np.array(building_verts)])
        all_faces = terrain_faces + building_faces

        return pv.PolyData(all_points, all_faces)

    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")


def generate_test_points(n_points=10, bounds=(-10, 10), seed=42):
    """
    Generate random test points.

    Args:
        n_points: Number of points to generate
        bounds: Tuple of (min, max) for x, y coordinates
        seed: Random seed

    Returns:
        Array of shape (n_points, 3) with (x, y, z) coordinates
    """
    np.random.seed(seed)
    x = np.random.uniform(bounds[0], bounds[1], n_points)
    y = np.random.uniform(bounds[0], bounds[1], n_points)
    z = np.zeros(n_points)  # On ground level
    return np.column_stack([x, y, z])
