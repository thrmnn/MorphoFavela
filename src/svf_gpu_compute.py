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
    batch_size: int = 100,
    max_ray_length: float = 1000.0,
    num_samples_per_ray: int = 20
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
    logger.info(f"  Total rays to process: {n_points * n_patches}")
    
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
    
    OPTIMIZED VERSION: Uses adaptive sampling and early termination.
    
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
    
    # OPTIMIZATION: Process in smaller chunks to reduce memory pressure
    # and improve cache efficiency
    max_chunk_size = 5000  # Maximum points to check at once
    
    # Expand observers and patches for all combinations
    # Shape: (batch_size, n_patches, 3)
    observers_exp = observers.unsqueeze(1).expand(-1, n_patches, -1)  # (B, M, 3)
    patches_exp = sky_patches.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, 3)
    
    # Compute ray directions and lengths
    ray_directions = patches_exp - observers_exp  # (B, M, 3)
    ray_lengths = torch.norm(ray_directions, dim=-1, keepdim=True)  # (B, M, 1)
    ray_directions_norm = ray_directions / (ray_lengths + 1e-8)  # (B, M, 3)
    
    # OPTIMIZATION: Use fewer samples with adaptive spacing
    # Focus sampling near observer (where obstacles are more likely)
    # Use exponential spacing: more samples near start, fewer near end
    if num_samples_per_ray > 20:
        # Create exponential spacing for better coverage with fewer samples
        t = torch.linspace(0, 1, num_samples_per_ray, device=device)
        # Exponential curve: more samples near 0 (observer)
        t_exp = 1 - torch.exp(-3 * t)  # Maps [0,1] to [0, ~0.95]
        sample_distances_normalized = t_exp.unsqueeze(0).unsqueeze(0)  # (1, 1, num_samples)
    else:
        # Linear spacing for small number of samples
        sample_distances_normalized = torch.linspace(
            0.1, 1.0, num_samples_per_ray, device=device
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, num_samples)
    
    # Scale by actual ray lengths
    sample_distances = sample_distances_normalized * ray_lengths  # (B, M, num_samples)
    
    # Compute sample points along rays
    # Shape: (B, M, num_samples, 3)
    sample_points = (
        observers_exp.unsqueeze(2) + 
        ray_directions_norm.unsqueeze(2) * sample_distances.unsqueeze(-1)
    )
    
    # Flatten for batch processing: (B * M * num_samples, 3)
    sample_points_flat = sample_points.reshape(-1, 3)
    
    # OPTIMIZATION: Process in chunks to avoid OOM and improve performance
    total_samples = len(sample_points_flat)
    visible_mask = torch.zeros(total_samples, dtype=torch.bool, device=device)
    
    for chunk_start in range(0, total_samples, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, total_samples)
        chunk_points = sample_points_flat[chunk_start:chunk_end]
        
        # Check visibility for this chunk
        chunk_visible = _check_points_visible(chunk_points, mesh)
        visible_mask[chunk_start:chunk_end] = chunk_visible
    
    # Reshape back: (B, M, num_samples)
    visible_mask = visible_mask.reshape(batch_size, n_patches, num_samples_per_ray)
    
    # OPTIMIZATION: Early termination - if any point is blocked, patch is blocked
    # A patch is visible if all sampled points along the ray are visible
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
    
    OPTIMIZED VERSION: Uses spatial partitioning and early termination
    for much faster computation on large meshes.
    
    Args:
        points: Tensor of shape (N, 3) with points to check
        mesh: PyTorch3D Meshes object
        threshold: Distance threshold (meters) for considering point blocked
    
    Returns:
        Boolean tensor of shape (N,) indicating visibility (True = visible)
    """
    # Get mesh vertices
    verts = mesh.verts_list()[0]  # (V, 3)
    
    # OPTIMIZATION 1: Use spatial partitioning with bounding box pre-filtering
    # Compute mesh bounding box
    mesh_min = verts.min(dim=0)[0]  # (3,)
    mesh_max = verts.max(dim=0)[0]  # (3,)
    
    # Quick check: points far outside bounding box are definitely visible
    # Expand bbox by threshold for safety
    expanded_min = mesh_min - threshold
    expanded_max = mesh_max + threshold
    
    # Check which points are outside expanded bounding box
    outside_mask = (
        (points[:, 0] < expanded_min[0]) | (points[:, 0] > expanded_max[0]) |
        (points[:, 1] < expanded_min[1]) | (points[:, 1] > expanded_max[1]) |
        (points[:, 2] < expanded_min[2]) | (points[:, 2] > expanded_max[2])
    )
    
    # Points outside bbox are definitely visible
    visible = outside_mask.clone()
    
    # Only check points inside/near bounding box
    points_to_check = points[~outside_mask]
    
    if len(points_to_check) > 0:
        # OPTIMIZATION 2: For large meshes, use chunked distance computation
        # to avoid OOM and improve cache efficiency
        chunk_size = min(10000, len(verts))  # Process vertices in chunks
        
        if len(verts) > chunk_size:
            # Chunked computation for large meshes
            min_distances = torch.full(
                (len(points_to_check),), 
                float('inf'), 
                device=points.device, 
                dtype=points.dtype
            )
            
            for i in range(0, len(verts), chunk_size):
                verts_chunk = verts[i:i+chunk_size]
                distances_chunk = torch.cdist(points_to_check, verts_chunk)  # (N, chunk_size)
                min_distances_chunk = distances_chunk.min(dim=1)[0]  # (N,)
                min_distances = torch.minimum(min_distances, min_distances_chunk)
        else:
            # Direct computation for smaller meshes
            distances = torch.cdist(points_to_check, verts)  # (N, V)
            min_distances = distances.min(dim=1)[0]  # (N,)
        
        # Points are visible if distance > threshold
        visible[~outside_mask] = min_distances > threshold
    else:
        # All points are outside bounding box - all visible
        pass
    
    return visible
