"""
Pytest configuration and shared fixtures for SVF tests.
"""

import numpy as np
import pyvista as pv
import pytest
import torch
from pathlib import Path


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons."""
    return 0.01


@pytest.fixture
def strict_tolerance():
    """Strict tolerance for critical comparisons."""
    return 0.001


@pytest.fixture
def correlation_threshold():
    """Minimum correlation for CPU vs GPU comparisons."""
    return 0.95


@pytest.fixture
def empty_mesh():
    """Create an empty flat terrain mesh (no buildings)."""
    # Create a simple flat plane
    x = np.linspace(-10, 10, 21)
    y = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Create PyVista mesh
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    
    # Create triangular faces
    n = len(x)
    for i in range(n - 1):
        for j in range(n - 1):
            # Two triangles per quad
            idx = i * n + j
            faces.extend([
                [3, idx, idx + 1, idx + n],
                [3, idx + 1, idx + n + 1, idx + n]
            ])
    
    mesh = pv.PolyData(points, faces)
    return mesh


@pytest.fixture
def single_building_mesh():
    """Create a mesh with a single building (box) in the center."""
    # Create terrain
    x = np.linspace(-20, 20, 41)
    y = np.linspace(-20, 20, 41)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Create building (box from -2 to 2 in x, y, height 5)
    building_points = []
    building_faces = []
    
    # Building base (z=0)
    base_corners = [
        [-2, -2, 0],
        [2, -2, 0],
        [2, 2, 0],
        [-2, 2, 0]
    ]
    
    # Building top (z=5)
    top_corners = [
        [-2, -2, 5],
        [2, -2, 5],
        [2, 2, 5],
        [-2, 2, 5]
    ]
    
    # Combine terrain and building
    terrain_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Create terrain mesh
    n = len(x)
    terrain_faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            terrain_faces.extend([
                [3, idx, idx + 1, idx + n],
                [3, idx + 1, idx + n + 1, idx + n]
            ])
    
    # Create building mesh (box with 6 faces)
    building_verts = base_corners + top_corners
    building_face_indices = [
        [4, 0, 1, 2, 3],  # bottom
        [4, 4, 7, 6, 5],  # top
        [4, 0, 4, 5, 1],  # front
        [4, 2, 6, 7, 3],  # back
        [4, 0, 3, 7, 4],  # left
        [4, 1, 5, 6, 2]   # right
    ]
    
    # Offset building indices by terrain point count
    n_terrain = len(terrain_points)
    building_faces = []
    for face in building_face_indices:
        offset_face = [face[0]] + [v + n_terrain for v in face[1:]]
        building_faces.append(offset_face)
    
    # Combine
    all_points = np.vstack([terrain_points, np.array(building_verts)])
    all_faces = terrain_faces + building_faces
    
    mesh = pv.PolyData(all_points, all_faces)
    return mesh


@pytest.fixture
def two_buildings_mesh():
    """Create a mesh with two buildings."""
    # Similar to single_building_mesh but with two boxes
    x = np.linspace(-30, 30, 61)
    y = np.linspace(-30, 30, 61)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    terrain_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Building 1: from -5 to -1 in x, -2 to 2 in y, height 5
    # Building 2: from 1 to 5 in x, -2 to 2 in y, height 5
    
    building_verts = []
    # Building 1 base
    building_verts.extend([
        [-5, -2, 0], [1, -2, 0], [1, 2, 0], [-5, 2, 0]  # Actually make it -5 to -1
    ])
    # Building 1 top
    building_verts.extend([
        [-5, -2, 5], [1, -2, 5], [1, 2, 5], [-5, 2, 5]
    ])
    # Building 2 base
    building_verts.extend([
        [-1, -2, 0], [5, -2, 0], [5, 2, 0], [-1, 2, 0]
    ])
    # Building 2 top
    building_verts.extend([
        [-1, -2, 5], [5, -2, 5], [5, 2, 5], [-1, 2, 5]
    ])
    
    # Create terrain mesh
    n = len(x)
    terrain_faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            terrain_faces.extend([
                [3, idx, idx + 1, idx + n],
                [3, idx + 1, idx + n + 1, idx + n]
            ])
    
    # Create building meshes
    n_terrain = len(terrain_points)
    building_faces = []
    
    # Building 1 (indices 0-7)
    for base_idx in range(2):
        offset = base_idx * 8
        building_faces.extend([
            [4, n_terrain + offset + 0, n_terrain + offset + 1, n_terrain + offset + 2, n_terrain + offset + 3],  # bottom
            [4, n_terrain + offset + 4, n_terrain + offset + 7, n_terrain + offset + 6, n_terrain + offset + 5],  # top
            [4, n_terrain + offset + 0, n_terrain + offset + 4, n_terrain + offset + 5, n_terrain + offset + 1],  # front
            [4, n_terrain + offset + 2, n_terrain + offset + 6, n_terrain + offset + 7, n_terrain + offset + 3],  # back
            [4, n_terrain + offset + 0, n_terrain + offset + 3, n_terrain + offset + 7, n_terrain + offset + 4],  # left
            [4, n_terrain + offset + 1, n_terrain + offset + 5, n_terrain + offset + 6, n_terrain + offset + 2]   # right
        ])
    
    all_points = np.vstack([terrain_points, np.array(building_verts)])
    all_faces = terrain_faces + building_faces
    
    mesh = pv.PolyData(all_points, all_faces)
    return mesh


@pytest.fixture
def test_points():
    """Generate test points for SVF computation."""
    # Points at various locations
    points = np.array([
        [0, 0, 0],      # Origin
        [10, 10, 0],   # Far from center
        [-10, -10, 0], # Far in opposite direction
        [0, 0, 0],     # Under building (for single_building_mesh)
        [5, 5, 0],     # Mid-distance
    ])
    return points


@pytest.fixture
def sky_patches_36():
    """Generate 36 sky patches."""
    from scripts.compute_svf import generate_sky_patches
    patches, _ = generate_sky_patches(36)
    return patches


@pytest.fixture
def sky_patches_145():
    """Generate 145 sky patches."""
    from scripts.compute_svf import generate_sky_patches
    patches, _ = generate_sky_patches(145)
    return patches


@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture
def device(gpu_available):
    """Get appropriate device for tests."""
    return torch.device("cuda" if gpu_available else "cpu")
