"""
GPU-accelerated SVF computation using PyTorch3D.

This module implements GPU-accelerated ray-casting for Sky View Factor computation.
Uses PyTorch3D for mesh operations and PyTorch for parallel computation.
"""

import torch
from pytorch3d.structures import Meshes
from typing import Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_svf_gpu(
    observer_points: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    batch_size: int = 100,
    max_ray_length: float = 1000.0,
    num_samples_per_ray: int = 20,
    ray_chunk_size: int = 1024,
    tri_chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute SVF using GPU-accelerated batch ray-casting.
    
    Uses exact segment-triangle intersection (Moller-Trumbore) on GPU.
    The computation is chunked over rays and triangles for memory stability.
    
    Args:
        observer_points: Tensor of shape (N, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch centroids
        mesh: PyTorch3D Meshes object
        batch_size: Number of rays to process in parallel
        max_ray_length: Maximum ray length (meters)
        num_samples_per_ray: Kept for backward compatibility (unused in exact mode)
        ray_chunk_size: Number of rays to intersect in each kernel chunk
        tri_chunk_size: Number of triangles to intersect in each kernel chunk
    
    Returns:
        Tensor of shape (N,) with SVF values
    """
    return compute_svf_gpu_batch(
        observer_points,
        sky_patches,
        mesh,
        batch_size,
        max_ray_length,
        num_samples_per_ray,
        ray_chunk_size,
        tri_chunk_size,
    )


def compute_svf_gpu_batch(
    observer_points: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    batch_size: int = 1000,
    max_ray_length: float = 1000.0,
    num_samples_per_ray: int = 100,
    ray_chunk_size: int = 1024,
    tri_chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Compute SVF using GPU-accelerated batch ray-casting.
    
    Uses exact segment-triangle intersections.
    
    Args:
        observer_points: Tensor of shape (N, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch centroids
        mesh: PyTorch3D Meshes object
        batch_size: Number of observer points to process in parallel
        max_ray_length: Maximum ray length (meters)
        num_samples_per_ray: Kept for backward compatibility (unused in exact mode)
        ray_chunk_size: Number of rays to intersect in each chunk
        tri_chunk_size: Number of triangles to intersect in each chunk
    
    Returns:
        Tensor of shape (N,) with SVF values
    """
    device = observer_points.device
    n_points = observer_points.shape[0]
    n_patches = sky_patches.shape[0]
    
    logger.info(f"Computing SVF on GPU for {n_points} points with {n_patches} sky patches")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info("  Intersection: exact ray-triangle")
    logger.info(f"  Ray chunk size: {ray_chunk_size}")
    logger.info(f"  Triangle chunk size: {tri_chunk_size}")
    logger.info(f"  Total rays to process: {n_points * n_patches}")
    
    svf_values = torch.zeros(n_points, device=device, dtype=torch.float32)
    mesh_triangles = _prepare_mesh_triangles(mesh)
    
    # Process observer points in batches to manage memory
    with torch.no_grad():
        for i in tqdm(range(0, n_points, batch_size), desc="Computing SVF (GPU)"):
            batch_end = min(i + batch_size, n_points)
            batch_observers = observer_points[i:batch_end]
            
            # Compute SVF for this batch
            batch_svf = _compute_batch_svf(
                batch_observers,
                sky_patches,
                mesh_triangles,
                max_ray_length,
                ray_chunk_size,
                tri_chunk_size,
            )
            
            svf_values[i:batch_end] = batch_svf
    
    return svf_values


def _compute_batch_svf(
    observers: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh_triangles: Dict[str, torch.Tensor],
    max_ray_length: float,
    ray_chunk_size: int,
    tri_chunk_size: int,
) -> torch.Tensor:
    """
    Compute SVF for a batch of observers using exact segment-triangle intersection.
    
    Args:
        observers: Tensor of shape (B, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch centroids
        mesh_triangles: Precomputed triangle data
        max_ray_length: Maximum ray length
        ray_chunk_size: Number of rays to intersect in each chunk
        tri_chunk_size: Number of triangles to intersect in each chunk
    
    Returns:
        Tensor of shape (B,) with SVF values
    """
    batch_size = observers.shape[0]
    n_patches = sky_patches.shape[0]

    observers_exp = observers.unsqueeze(1).expand(-1, n_patches, -1)
    patches_exp = sky_patches.unsqueeze(0).expand(batch_size, -1, -1)

    ray_vectors = patches_exp - observers_exp
    ray_lengths = torch.linalg.norm(ray_vectors, dim=-1)

    if max_ray_length is not None and max_ray_length > 0:
        ray_lengths = torch.clamp(ray_lengths, max=max_ray_length)

    ray_directions = ray_vectors / (torch.linalg.norm(ray_vectors, dim=-1, keepdim=True) + 1e-8)

    ray_origins_flat = observers_exp.reshape(-1, 3)
    ray_dirs_flat = ray_directions.reshape(-1, 3)
    ray_lengths_flat = ray_lengths.reshape(-1)

    blocked = _rays_blocked_by_mesh_exact(
        ray_origins_flat,
        ray_dirs_flat,
        ray_lengths_flat,
        mesh_triangles,
        ray_chunk_size=ray_chunk_size,
        tri_chunk_size=tri_chunk_size,
    )
    visible = (~blocked).reshape(batch_size, n_patches)
    return visible.float().mean(dim=1)


def _prepare_mesh_triangles(mesh: Meshes) -> Dict[str, torch.Tensor]:
    """Precompute triangle representation used in ray-triangle tests."""
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return {"v0": v0, "e1": v1 - v0, "e2": v2 - v0}


def _rays_blocked_by_mesh_exact(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    ray_lengths: torch.Tensor,
    mesh_triangles: Dict[str, torch.Tensor],
    ray_chunk_size: int = 1024,
    tri_chunk_size: int = 4096,
    eps: float = 1e-4,  # Increased epsilon to match CPU behavior better
) -> torch.Tensor:
    """
    Exact segment-triangle intersection using chunked Moller-Trumbore on GPU.
    Returns True when a ray segment intersects any triangle before its endpoint.
    """
    v0_all = mesh_triangles["v0"]
    e1_all = mesh_triangles["e1"]
    e2_all = mesh_triangles["e2"]

    n_rays = ray_origins.shape[0]
    n_tris = v0_all.shape[0]
    blocked = torch.zeros(n_rays, dtype=torch.bool, device=ray_origins.device)

    for r0 in range(0, n_rays, ray_chunk_size):
        r1 = min(r0 + ray_chunk_size, n_rays)
        o = ray_origins[r0:r1]
        d = ray_dirs[r0:r1]
        ray_len = ray_lengths[r0:r1]
        ray_hit = torch.zeros(r1 - r0, dtype=torch.bool, device=ray_origins.device)

        for t0 in range(0, n_tris, tri_chunk_size):
            if ray_hit.all():
                break

            t1 = min(t0 + tri_chunk_size, n_tris)
            v0 = v0_all[t0:t1]
            e1 = e1_all[t0:t1]
            e2 = e2_all[t0:t1]

            pvec = torch.cross(d[:, None, :], e2[None, :, :], dim=-1)
            det = torch.sum(e1[None, :, :] * pvec, dim=-1)
            det_mask = torch.abs(det) > eps
            inv_det = torch.where(det_mask, 1.0 / det, torch.zeros_like(det))

            tvec = o[:, None, :] - v0[None, :, :]
            u = torch.sum(tvec * pvec, dim=-1) * inv_det
            u_mask = (u >= 0.0) & (u <= 1.0)

            qvec = torch.cross(tvec, e1[None, :, :], dim=-1)
            v = torch.sum(d[:, None, :] * qvec, dim=-1) * inv_det
            v_mask = (v >= 0.0) & ((u + v) <= 1.0)

            t = torch.sum(e2[None, :, :] * qvec, dim=-1) * inv_det
            # Match CPU PyVista ray_trace behavior:
            # - PyVista detects intersections at t=0 when observer is on surface
            # - For corner/edge cases, PyVista may ignore some t=0 intersections
            # - Use adaptive threshold: very small for normal cases, slightly larger for edge cases
            # - Allow small negative t to catch intersections at origin, but be conservative
            t_eps_min = eps * 10  # Minimum threshold for edge cases
            t_eps_max = eps * 50  # Maximum threshold (for very long rays)
            # Adaptive threshold based on ray length (longer rays need larger threshold)
            t_eps = torch.clamp(ray_len[:, None] * 1e-6, t_eps_min, t_eps_max)
            t_mask = (t >= -t_eps) & (t <= (ray_len[:, None] + t_eps))

            hits = det_mask & u_mask & v_mask & t_mask
            ray_hit |= hits.any(dim=1)

        blocked[r0:r1] = ray_hit

    return blocked


def _check_points_visible(
    points: torch.Tensor,
    mesh: Meshes,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Check if points are visible (not inside/on mesh surface).
    
    Uses PyTorch3D's point-to-mesh face distance for accurate computation.
    This is more accurate than distance-to-vertices but slower.
    
    Args:
        points: Tensor of shape (N, 3) with points to check
        mesh: PyTorch3D Meshes object
        threshold: Distance threshold (meters) for considering point blocked
    
    Returns:
        Boolean tensor of shape (N,) indicating visibility (True = visible)
    """
    try:
        from pytorch3d.loss import point_mesh_face_distance
    except ImportError:
        # Fallback to vertex distance if PyTorch3D function not available
        logger.warning("point_mesh_face_distance not available, using vertex distance (less accurate)")
        return _check_points_visible_vertex_approx(points, mesh, threshold)
    
    # Get mesh vertices and faces
    verts = mesh.verts_list()[0]  # (V, 3)
    _faces = mesh.faces_list()[0]  # (F, 3)
    
    # OPTIMIZATION: Use bounding box pre-filtering
    mesh_min = verts.min(dim=0)[0]  # (3,)
    mesh_max = verts.max(dim=0)[0]  # (3,)
    
    # Quick check: points far outside bounding box are definitely visible
    expanded_min = mesh_min - threshold
    expanded_max = mesh_max + threshold
    
    outside_mask = (
        (points[:, 0] < expanded_min[0]) | (points[:, 0] > expanded_max[0]) |
        (points[:, 1] < expanded_min[1]) | (points[:, 1] > expanded_max[1]) |
        (points[:, 2] < expanded_min[2]) | (points[:, 2] > expanded_max[2])
    )
    
    visible = outside_mask.clone()
    
    # Only check points inside/near bounding box
    points_to_check = points[~outside_mask]
    
    if len(points_to_check) > 0:
        # Use PyTorch3D's point-to-mesh face distance
        # This computes distance to mesh surface (triangles), not just vertices
        # Process in chunks to avoid OOM
        chunk_size = 5000
        min_distances = torch.full(
            (len(points_to_check),),
            float('inf'),
            device=points.device,
            dtype=points.dtype
        )
        
        for i in range(0, len(points_to_check), chunk_size):
            chunk_end = min(i + chunk_size, len(points_to_check))
            points_chunk = points_to_check[i:chunk_end]
            
            # Compute distance to mesh faces
            # point_mesh_face_distance returns squared distances
            distances_sq = point_mesh_face_distance(mesh, points_chunk.unsqueeze(0))
            distances = torch.sqrt(distances_sq)  # Convert to actual distances
            
            min_distances[i:chunk_end] = distances.squeeze(0)
        
        # Points are visible if distance > threshold
        visible[~outside_mask] = min_distances > threshold
    else:
        # All points are outside bounding box - all visible
        pass
    
    return visible


def _check_points_visible_vertex_approx(
    points: torch.Tensor,
    mesh: Meshes,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Fallback: Check visibility using distance to vertices (less accurate).
    """
    verts = mesh.verts_list()[0]
    mesh_min = verts.min(dim=0)[0]
    mesh_max = verts.max(dim=0)[0]
    
    expanded_min = mesh_min - threshold
    expanded_max = mesh_max + threshold
    
    outside_mask = (
        (points[:, 0] < expanded_min[0]) | (points[:, 0] > expanded_max[0]) |
        (points[:, 1] < expanded_min[1]) | (points[:, 1] > expanded_max[1]) |
        (points[:, 2] < expanded_min[2]) | (points[:, 2] > expanded_max[2])
    )
    
    visible = outside_mask.clone()
    points_to_check = points[~outside_mask]
    
    if len(points_to_check) > 0:
        chunk_size = min(10000, len(verts))
        if len(verts) > chunk_size:
            min_distances = torch.full(
                (len(points_to_check),),
                float('inf'),
                device=points.device,
                dtype=points.dtype
            )
            for i in range(0, len(verts), chunk_size):
                verts_chunk = verts[i:i+chunk_size]
                distances_chunk = torch.cdist(points_to_check, verts_chunk)
                min_distances_chunk = distances_chunk.min(dim=1)[0]
                min_distances = torch.minimum(min_distances, min_distances_chunk)
        else:
            distances = torch.cdist(points_to_check, verts)
            min_distances = distances.min(dim=1)[0]
        
        visible[~outside_mask] = min_distances > threshold
    
    return visible
