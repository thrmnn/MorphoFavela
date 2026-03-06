#!/usr/bin/env python3
"""
Debug script to test GPU SVF computation on a small subset.
"""

import numpy as np
import pyvista as pv
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_utils import load_mesh
from scripts.compute_svf import generate_sky_patches
from src.svf_gpu_utils import (
    pv_mesh_to_pytorch3d,
    prepare_observer_points,
    prepare_sky_patches,
    check_gpu_availability
)
from src.svf_gpu_compute import compute_svf_gpu, _check_points_visible

def test_small_subset():
    """Test GPU SVF on a small subset of points."""
    
    print("=" * 60)
    print("GPU SVF DEBUG TEST")
    print("=" * 60)
    
    # Load mesh
    stl_path = Path("data/rocinha/rocinha.stl")
    print(f"\n1. Loading mesh from {stl_path}...")
    mesh = load_mesh(stl_path)
    print(f"   Mesh loaded: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # Check GPU
    gpu_info = check_gpu_availability()
    if not gpu_info['available']:
        print("ERROR: GPU not available!")
        return
    device = torch.device("cuda")
    print(f"   GPU: {gpu_info['device_name']}")
    
    # Convert to PyTorch3D
    print("\n2. Converting mesh to PyTorch3D...")
    pytorch3d_mesh = pv_mesh_to_pytorch3d(mesh, device=device)
    verts = pytorch3d_mesh.verts_list()[0]
    print(f"   Vertices: {len(verts)}")
    print(f"   Vertex bounds: min={verts.min(dim=0)[0]}, max={verts.max(dim=0)[0]}")
    
    # Generate sky patches
    print("\n3. Generating sky patches...")
    sky_patches, _ = generate_sky_patches(145)
    print(f"   Generated {len(sky_patches)} sky patches")
    print(f"   Sky patch bounds: min={sky_patches.min(axis=0)}, max={sky_patches.max(axis=0)}")
    
    # Create a few test observer points (above ground level)
    print("\n4. Creating test observer points...")
    # Get mesh bounds
    mesh_bounds = mesh.bounds
    center_x = (mesh_bounds[0] + mesh_bounds[1]) / 2
    center_y = (mesh_bounds[2] + mesh_bounds[3]) / 2
    
    # Sample a few points at different locations
    test_points = np.array([
        [center_x, center_y, 5.0],  # Center, 5m high
        [center_x + 50, center_y, 5.0],  # 50m east
        [center_x, center_y + 50, 5.0],  # 50m north
        [center_x - 50, center_y, 5.0],  # 50m west
        [center_x, center_y - 50, 5.0],  # 50m south
    ])
    print(f"   Created {len(test_points)} test points")
    print(f"   Test points:\n{test_points}")
    
    # Prepare for GPU
    print("\n5. Preparing data for GPU...")
    observer_points_torch = prepare_observer_points(test_points, evaluation_height=1.5, device=device)
    sky_patches_torch = prepare_sky_patches(sky_patches, device=device)
    print(f"   Observer points shape: {observer_points_torch.shape}")
    print(f"   Sky patches shape: {sky_patches_torch.shape}")
    print(f"   Observer points:\n{observer_points_torch}")
    
    # Test visibility checking on a single point
    print("\n6. Testing visibility checking...")
    test_point = observer_points_torch[0:1]  # First observer point
    print(f"   Testing point: {test_point}")
    
    # Check a few sample points along a ray
    first_patch = sky_patches_torch[0:1]  # First sky patch
    ray_dir = first_patch - test_point
    ray_length = torch.norm(ray_dir)
    ray_dir_norm = ray_dir / ray_length
    
    # Sample a few points along the ray
    sample_dists = torch.linspace(0.1, 1.0, 10, device=device) * ray_length
    sample_points = test_point + ray_dir_norm * sample_dists.unsqueeze(-1)
    print(f"   Sample points along ray:\n{sample_points}")
    
    # Check visibility
    visible = _check_points_visible(sample_points, pytorch3d_mesh, threshold=1.0)
    print(f"   Visibility results: {visible}")
    print(f"   Number visible: {visible.sum()}/{len(visible)}")
    
    # Check distances
    verts = pytorch3d_mesh.verts_list()[0]
    distances = torch.cdist(sample_points, verts)
    min_distances = distances.min(dim=1)[0]
    print(f"   Min distances to mesh: {min_distances}")
    print(f"   Min distance: {min_distances.min():.3f}, Max: {min_distances.max():.3f}")
    
    # Test full SVF computation on just 1 point
    print("\n7. Testing full SVF computation on 1 point...")
    single_observer = observer_points_torch[0:1]
    svf_result = compute_svf_gpu(
        single_observer,
        sky_patches_torch,
        pytorch3d_mesh,
        batch_size=1,
        num_samples_per_ray=20
    )
    print(f"   SVF result: {svf_result}")
    print(f"   SVF value: {svf_result[0].item():.4f}")
    
    # Compare with a point that should definitely be visible (very high up)
    print("\n8. Testing with a point high above the mesh...")
    high_point = torch.tensor([[center_x, center_y, 100.0]], device=device, dtype=torch.float32)
    svf_high = compute_svf_gpu(
        high_point,
        sky_patches_torch,
        pytorch3d_mesh,
        batch_size=1,
        num_samples_per_ray=20
    )
    print(f"   High point SVF: {svf_high[0].item():.4f}")
    
    print("\n" + "=" * 60)
    print("DEBUG TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_small_subset()
