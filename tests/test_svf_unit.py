"""
Unit tests for SVF algorithm components.
"""

import numpy as np
import pytest
import torch
import pyvista as pv
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_svf import generate_sky_patches
from src.svf_gpu_utils import (
    pv_mesh_to_pytorch3d,
    prepare_observer_points,
    prepare_sky_patches,
    check_gpu_availability
)
from tests.utils.test_helpers import assert_svf_valid


class TestSkyPatchGeneration:
    """Test sky patch generation."""
    
    def test_patch_count(self):
        """Test that requested number of patches is generated."""
        for n_patches in [36, 72, 145, 290]:
            patches, _ = generate_sky_patches(n_patches)
            # Allow some flexibility due to grid discretization
            assert len(patches) > 0.8 * n_patches, \
                f"Too few patches generated: {len(patches)} < {0.8 * n_patches}"
            assert len(patches) <= n_patches * 1.2, \
                f"Too many patches generated: {len(patches)} > {1.2 * n_patches}"
    
    def test_patches_in_hemisphere(self):
        """Test that all patches are in upper hemisphere."""
        patches, _ = generate_sky_patches(145)
        
        # Convert to spherical coordinates
        r = np.linalg.norm(patches, axis=1)
        elevation = np.arcsin(patches[:, 2] / r)
        
        # All patches should be in upper hemisphere (elevation > 0)
        assert np.all(elevation > 0), "Some patches are below horizon"
        assert np.all(elevation <= np.pi / 2), "Some patches are above zenith"
    
    def test_patch_radius(self):
        """Test that patches are at expected radius."""
        patches, _ = generate_sky_patches(145, radius=1000.0)
        
        distances = np.linalg.norm(patches, axis=1)
        # Patches should be approximately at the specified radius
        assert np.allclose(distances, 1000.0, rtol=0.1), \
            "Patches are not at expected radius"
    
    def test_patch_distribution(self):
        """Test that patches are well-distributed."""
        patches, _ = generate_sky_patches(145)
        
        # Check that patches cover full azimuth range
        azimuth = np.arctan2(patches[:, 1], patches[:, 0])
        azimuth_range = np.max(azimuth) - np.min(azimuth)
        assert azimuth_range > np.pi, "Azimuth range too small"
        
        # Check that patches cover elevation range
        r = np.linalg.norm(patches, axis=1)
        elevation = np.arcsin(patches[:, 2] / r)
        elevation_range = np.max(elevation) - np.min(elevation)
        assert elevation_range > 0.5, "Elevation range too small"


class TestMeshConversion:
    """Test mesh conversion utilities."""
    
    def test_pv_to_pytorch3d_conversion(self):
        """Test PyVista to PyTorch3D conversion."""
        # Create a simple mesh using extract_surface to get proper face format
        box = pv.Box(bounds=(-5, 5, -5, 5, 0, 10))
        mesh = box.extract_surface()
        
        # Convert to PyTorch3D
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=torch.device('cpu'))
        
        # Check structure
        assert len(pytorch3d_mesh.verts_list()) == 1
        assert len(pytorch3d_mesh.faces_list()) == 1
        
        verts = pytorch3d_mesh.verts_list()[0]
        faces = pytorch3d_mesh.faces_list()[0]
        
        # Check dimensions
        assert verts.shape[1] == 3, "Vertices should have 3 coordinates"
        assert faces.shape[1] == 3, "Faces should have 3 vertices"
        
        # Check that all face indices are valid
        assert torch.all(faces >= 0), "Face indices should be non-negative"
        assert torch.all(faces < len(verts)), "Face indices should be within vertex range"
    
    def test_mesh_with_nan_vertices(self):
        """Test handling of NaN vertices."""
        # Create mesh with some NaN vertices
        # Use VTK format for faces: [n_verts, v0, v1, v2, ...]
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [np.nan, np.nan, np.nan],  # Invalid vertex
            [2, 2, 2]
        ])
        # VTK format: [3, 0, 1, 2, 3, 1, 2, 4]
        faces = np.array([3, 0, 1, 2, 3, 1, 2, 4], dtype=np.int32)
        
        # Create PyVista mesh
        mesh = pv.PolyData(points, faces=faces)
        
        # Convert should filter out NaN vertices
        pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=torch.device('cpu'))
        
        verts = pytorch3d_mesh.verts_list()[0]
        # Should have filtered out NaN vertex
        assert len(verts) <= len(points), "NaN vertices should be filtered"
        assert not torch.any(torch.isnan(verts)), "No NaN vertices in converted mesh"
    
    def test_device_placement(self):
        """Test device placement for GPU."""
        box = pv.Box(bounds=(-5, 5, -5, 5, 0, 10))
        mesh = box.extract_surface()
        
        # Test CPU device
        pytorch3d_mesh_cpu = pv_mesh_to_pytorch3d(mesh, device=torch.device('cpu'))
        assert pytorch3d_mesh_cpu.verts_list()[0].device.type == 'cpu'
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            pytorch3d_mesh_gpu = pv_mesh_to_pytorch3d(mesh, device=torch.device('cuda'))
            assert pytorch3d_mesh_gpu.verts_list()[0].device.type == 'cuda'


class TestObserverPointPreparation:
    """Test observer point preparation."""
    
    def test_height_offset(self):
        """Test that evaluation height is correctly added."""
        ground_points = np.array([
            [0, 0, 0],
            [1, 1, 5],
            [2, 2, 10]
        ], dtype=np.float32)
        evaluation_height = 1.5
        
        observer_points = prepare_observer_points(
            ground_points,
            evaluation_height,
            device=torch.device('cpu')
        )
        
        # Check that height was added
        expected_z = ground_points[:, 2] + evaluation_height
        assert torch.allclose(observer_points[:, 2], torch.tensor(expected_z)), \
            "Evaluation height not correctly added"
    
    def test_tensor_format(self):
        """Test tensor format and dtype."""
        ground_points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        observer_points = prepare_observer_points(
            ground_points,
            1.5,
            device=torch.device('cpu')
        )
        
        assert isinstance(observer_points, torch.Tensor), "Should return torch.Tensor"
        assert observer_points.dtype == torch.float32, "Should be float32"
        assert observer_points.shape == (2, 3), "Should have shape (N, 3)"
    
    def test_device_consistency(self):
        """Test device consistency."""
        ground_points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        
        # CPU device
        observer_points_cpu = prepare_observer_points(
            ground_points,
            1.5,
            device=torch.device('cpu')
        )
        assert observer_points_cpu.device.type == 'cpu'
        
        # GPU device if available
        if torch.cuda.is_available():
            observer_points_gpu = prepare_observer_points(
                ground_points,
                1.5,
                device=torch.device('cuda')
            )
            assert observer_points_gpu.device.type == 'cuda'


class TestSkyPatchPreparation:
    """Test sky patch preparation for GPU."""
    
    def test_tensor_conversion(self):
        """Test conversion to tensor."""
        sky_patches = np.random.rand(145, 3) * 1000
        patches_tensor = prepare_sky_patches(sky_patches, device=torch.device('cpu'))
        
        assert isinstance(patches_tensor, torch.Tensor), "Should return torch.Tensor"
        assert patches_tensor.dtype == torch.float32, "Should be float32"
        assert patches_tensor.shape == (145, 3), "Should preserve shape"
        # Compare with float32 tensor to avoid dtype mismatch
        assert torch.allclose(patches_tensor, torch.tensor(sky_patches, dtype=torch.float32)), \
            "Values should be preserved"
    
    def test_device_placement(self):
        """Test device placement."""
        sky_patches = np.random.rand(145, 3) * 1000
        
        patches_cpu = prepare_sky_patches(sky_patches, device=torch.device('cpu'))
        assert patches_cpu.device.type == 'cpu'
        
        if torch.cuda.is_available():
            patches_gpu = prepare_sky_patches(sky_patches, device=torch.device('cuda'))
            assert patches_gpu.device.type == 'cuda'


class TestGPUAvailability:
    """Test GPU availability checking."""
    
    def test_gpu_check(self):
        """Test GPU availability check."""
        info = check_gpu_availability()
        
        assert 'available' in info
        assert 'device_name' in info
        assert 'device_count' in info
        assert 'cuda_version' in info
        
        if info['available']:
            assert info['device_count'] > 0
            assert info['device_name'] is not None
