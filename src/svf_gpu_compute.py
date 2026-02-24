"""
GPU-accelerated SVF computation using PyTorch3D.

This module implements GPU-accelerated ray-casting for Sky View Factor computation.
Uses PyTorch3D for mesh operations and PyTorch for parallel computation.
"""

import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from typing import Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_svf_gpu(
    observer_points: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    batch_size: int = 1000,
    max_ray_length: float = 1000.0,
    num_samples_per_ray: int = 100
) -> torch.Tensor:
    """
    Compute SVF using GPU-accelerated batch ray-casting.
    
    Uses a sampling-based approach: samples points along each ray and checks
    if they intersect the mesh. This is faster than exact ray-mesh intersection
    for large meshes.
    
    Args:
        observer_points: Tensor of shape (N, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch centroids
        mesh: PyTorch3D Meshes object
        batch_size: Number of rays to process in parallel
        max_ray_length: Maximum ray length (meters)
        num_samples_per_ray: Number of points to sample along each ray
    
    Returns:
        Tensor of shape (N,) with SVF values
    """
    return compute_svf_gpu_batch(
        observer_points, sky_patches, mesh, batch_size, max_ray_length, num_samples_per_ray
    )


def compute_svf_gpu_batch(
    observer_points: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    batch_size: int = 1000,
    max_ray_length: float = 1000.0,
    num_samples_per_ray: int = 100
) -> torch.Tensor:
    """
    Compute SVF using GPU-accelerated batch ray-casting.
    
    Uses a sampling-based approach for ray-mesh intersection checking.
    
    Args:
        observer_points: Tensor of shape (N, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch centroids
        mesh: PyTorch3D Meshes object
        batch_size: Number of observer points to process in parallel
        max_ray_length: Maximum ray length (meters)
        num_samples_per_ray: Number of points to sample along each ray
    
    Returns:
        Tensor of shape (N,) with SVF values
    """
    device = observer_points.device
    n_points = observer_points.shape[0]
    n_patches = sky_patches.shape[0]
    
    logger.info(f"Computing SVF on GPU for {n_points} points with {n_patches} sky patches")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Samples per ray: {num_samples_per_ray}")
    
    svf_values = torch.zeros(n_points, device=device, dtype=torch.float32)
    
    # Process observer points in batches to manage memory
    with torch.no_grad():
        for i in tqdm(range(0, n_points, batch_size), desc="Computing SVF (GPU)"):
            batch_end = min(i + batch_size, n_points)
            batch_observers = observer_points[i:batch_end]
            
            # Compute SVF for this batch
            batch_svf = _compute_batch_svf(
                batch_observers, sky_patches, mesh, max_ray_length, num_samples_per_ray
            )
            
            svf_values[i:batch_end] = batch_svf
    
    return svf_values


def _compute_batch_svf(
    observers: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    max_ray_length: float,
    num_samples_per_ray: int
) -> torch.Tensor:
    """
    Compute SVF for a batch of observers using sampling-based ray-mesh intersection.
    
    For each observer and sky patch:
    1. Sample points along the ray from observer to sky patch
    2. Check if sampled points are inside the mesh (using distance queries)
    3. If any point is inside, the ray is blocked
    4. SVF = visible patches / total patches
    
    Args:
        observers: Tensor of shape (B, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch centroids
        mesh: PyTorch3D Meshes object
        max_ray_length: Maximum ray length
        num_samples_per_ray: Number of points to sample along each ray
    
    Returns:
        Tensor of shape (B,) with SVF values
    """
    device = observers.device
    batch_size = observers.shape[0]
    n_patches = sky_patches.shape[0]
    
    # Expand observers and patches for all combinations
    # Shape: (batch_size, n_patches, 3)
    observers_exp = observers.unsqueeze(1).expand(-1, n_patches, -1)  # (B, M, 3)
    patches_exp = sky_patches.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, 3)
    
    # Compute ray directions and lengths
    ray_directions = patches_exp - observers_exp  # (B, M, 3)
    ray_lengths = torch.norm(ray_directions, dim=-1, keepdim=True)  # (B, M, 1)
    ray_directions_norm = ray_directions / (ray_lengths + 1e-8)  # (B, M, 3)
    
    # Sample points along rays
    # Create sampling distances from 0 to ray_length
    sample_distances = torch.linspace(
        0.1, 1.0, num_samples_per_ray, device=device
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, num_samples)
    
    # Scale by actual ray lengths
    sample_distances = sample_distances * ray_lengths  # (B, M, num_samples)
    
    # Compute sample points along rays
    # Shape: (B, M, num_samples, 3)
    sample_points = (
        observers_exp.unsqueeze(2) + 
        ray_directions_norm.unsqueeze(2) * sample_distances.unsqueeze(-1)
    )
    
    # Flatten for batch processing: (B * M * num_samples, 3)
    sample_points_flat = sample_points.reshape(-1, 3)
    
    # Check which sample points are inside the mesh
    # Use distance to mesh surface - if distance is very small, point is on/near surface
    # This is an approximation - for exact intersection, we'd need proper ray-mesh intersection
    visible_mask = _check_points_visible(sample_points_flat, mesh)
    
    # Reshape back: (B, M, num_samples)
    visible_mask = visible_mask.reshape(batch_size, n_patches, num_samples_per_ray)
    
    # A patch is visible if all sampled points along the ray are visible
    # (i.e., none are inside/on the mesh)
    patches_visible = visible_mask.all(dim=-1)  # (B, M)
    
    # Compute SVF: number of visible patches / total patches
    svf = patches_visible.float().mean(dim=1)  # (B,)
    
    return svf


def _check_points_visible(
    points: torch.Tensor,
    mesh: Meshes,
    threshold: float = 1.0
) -> torch.Tensor:
    """
    Check if points are visible (not inside/on mesh surface).
    
    Uses point-to-mesh distance as approximation. For accurate results,
    this should use proper point-in-mesh tests, but for performance we
    use distance to nearest mesh surface point.
    
    Args:
        points: Tensor of shape (N, 3) with points to check
        mesh: PyTorch3D Meshes object
        threshold: Distance threshold (meters) for considering point blocked
    
    Returns:
        Boolean tensor of shape (N,) indicating visibility (True = visible)
    """
    # Get mesh vertices
    verts = mesh.verts_list()[0]  # (V, 3)
    
    # For each point, find distance to nearest mesh vertex
    # This is a fast approximation - for more accuracy, we'd compute
    # distance to mesh surface (point-to-triangle distance)
    distances = torch.cdist(points, verts)  # (N, V)
    min_distances = distances.min(dim=1)[0]  # (N,)
    
    # Point is visible if distance > threshold
    # (i.e., point is far enough from mesh surface to be considered in free space)
    visible = min_distances > threshold
    
    return visible
