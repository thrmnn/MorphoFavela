"""
Unified SVF computation module with automatic CPU/GPU detection.

This module provides a single interface for computing Sky View Factor (SVF)
that automatically detects and uses GPU acceleration when available.
"""

import numpy as np
import pyvista as pv
import logging
from typing import Optional, Tuple, Union
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional dependency: PyViewFactor (exact view factor / SVF computations)
try:  # pragma: no cover - import guard
    import pyviewfactor as pvf  # type: ignore[import]
    PYVIEWFACTOR_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    pvf = None  # type: ignore[assignment]
    PYVIEWFACTOR_AVAILABLE = False

# Try to import GPU dependencies
try:
    import torch
    from pytorch3d.structures import Meshes
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    torch = None
    Meshes = None

# Import GPU utilities if available
if GPU_AVAILABLE:
    try:
        from src.svf_gpu_utils import (
            pv_mesh_to_pytorch3d,
            prepare_observer_points,
            prepare_sky_patches,
            check_gpu_availability
        )
        from src.svf_gpu_compute import compute_svf_gpu_batch
        GPU_COMPUTE_AVAILABLE = True
    except ImportError:
        GPU_COMPUTE_AVAILABLE = False
else:
    GPU_COMPUTE_AVAILABLE = False


def generate_sky_patches(num_patches: int, radius: float = 1000.0) -> Tuple[np.ndarray, list]:
    """
    Discretize the upper hemisphere into equal-area sky patches.
    
    Uses a simple latitude-longitude subdivision approach.
    For equal-area patches, we can use a Tregenza-like subdivision
    or a simpler uniform angular grid.
    
    Args:
        num_patches: Number of sky patches (e.g., 145, 290)
        radius: Distance to sky hemisphere (meters)
        
    Returns:
        Tuple of (patch_centroids, patch_meshes)
        - patch_centroids: Array of shape (N, 3) with patch center coordinates
        - patch_meshes: List of PyVista meshes for each patch (currently None for ray casting)
    """
    logger.info(f"Generating {num_patches} sky patches...")
    
    # Simple approach: uniform angular grid
    # Calculate grid dimensions to approximate num_patches
    # For hemisphere: patches ≈ azimuth_steps * elevation_steps / 2
    azimuth_steps = int(np.sqrt(num_patches * 2))
    elevation_steps = int(num_patches / azimuth_steps)
    
    # Adjust to get close to desired number
    total_patches = azimuth_steps * elevation_steps
    if total_patches < num_patches:
        elevation_steps += 1
        total_patches = azimuth_steps * elevation_steps
    
    logger.info(f"  Using {azimuth_steps} azimuth × {elevation_steps} elevation = {total_patches} patches")
    
    patch_centroids = []
    patch_meshes = []
    
    for az_idx in range(azimuth_steps):
        azimuth = 2 * np.pi * az_idx / azimuth_steps  # 0 to 2π
        
        for el_idx in range(elevation_steps):
            # Elevation from 0 (horizon) to π/2 (zenith)
            elevation = np.pi / 2 * (el_idx + 0.5) / elevation_steps
            
            # Skip patches below horizon (elevation < 0)
            if elevation <= 0:
                continue
            
            # Calculate patch centroid direction
            dx = np.cos(elevation) * np.cos(azimuth)
            dy = np.cos(elevation) * np.sin(azimuth)
            dz = np.sin(elevation)
            
            # Patch centroid at distance radius
            centroid = np.array([dx, dy, dz]) * radius
            patch_centroids.append(centroid)
            
            # Create a small patch polygon (simplified as a point for ray casting)
            # For actual view factor computation, we'd create a proper polygon
            # Here we use the centroid for ray casting
            patch_meshes.append(None)  # Not needed for ray casting approach
    
    patch_centroids = np.array(patch_centroids)
    logger.info(f"  Generated {len(patch_centroids)} sky patches")
    
    return patch_centroids, patch_meshes


def compute_svf_cpu(
    ground_points: np.ndarray,
    sky_patches: np.ndarray,
    full_mesh: pv.PolyData,
    evaluation_height: float
) -> np.ndarray:
    """
    Compute SVF for each ground point using CPU (PyVista).
    
    For each grid point and each sky patch:
    1. Cast a ray toward the patch centroid
    2. Use PyVista ray intersection to test obstruction
    3. Compute SVF as: number of visible patches / total patches
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates
        sky_patches: Array of shape (M, 3) with sky patch centroids
        full_mesh: Full scene mesh (terrain + buildings)
        evaluation_height: Height above ground for evaluation (meters)
        
    Returns:
        Array of SVF values (0-1) for each ground point
    """
    logger.info(f"Computing SVF on CPU for {len(ground_points)} points...")
    logger.info(f"  Evaluation height: {evaluation_height}m")
    logger.info(f"  Sky patches: {len(sky_patches)}")
    
    svf_values = []
    
    # Create observer points (ground points + evaluation height)
    # Ensure float dtype to avoid casting errors
    observer_points = ground_points.astype(np.float64).copy()
    observer_points[:, 2] += evaluation_height
    
    # Compute SVF for each observer point
    pbar = tqdm(total=len(observer_points), desc="Computing SVF (CPU)", unit="points")
    
    for i, observer in enumerate(observer_points):
        visible_patches = 0
        
        for patch_centroid in sky_patches:
            # Ray direction from observer to patch centroid
            ray_direction = patch_centroid - observer
            ray_length = np.linalg.norm(ray_direction)
            ray_direction = ray_direction / ray_length  # Normalize
            
            # Cast ray and check for intersection
            # PyVista ray intersection
            ray_end = observer + ray_direction * ray_length
            intersection, cell_id = full_mesh.ray_trace(observer, ray_end)
            
            # If no intersection, patch is visible
            # If intersection exists, the ray hit something before reaching the sky
            if len(intersection) == 0:
                visible_patches += 1
            # else: ray was blocked, patch is not visible
        
        # SVF = visible patches / total patches
        svf = visible_patches / len(sky_patches)
        svf_values.append(svf)
        
        # Update progress bar with current statistics
        if len(svf_values) > 0:
            current_mean = np.mean(svf_values)
            pbar.set_postfix({
                'mean_svf': f'{current_mean:.3f}',
                'current': f'{svf:.3f}'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    return np.array(svf_values)


def compute_svf_pyvf(
    ground_points: np.ndarray,
    sky_patches: np.ndarray,
    full_mesh: pv.PolyData,
    evaluation_height: float,
) -> np.ndarray:
    """
    Compute SVF using the experimental PyViewFactor backend.

    This uses PyViewFactor to approximate SVF at each observer point by:
    - Representing the observer as a small horizontal patch (receiver)
    - Treating all above-ground mesh faces as emitting facets (obstructions)
    - Computing view factors from the patch to each obstruction facet
    - Approximating SVF ≈ 1 - sum(view_factors_to_obstructions)

    This separation allows us to:
    - Keep tests explicitly targeting the PyViewFactor backend
    - Swap in a real pvf-based implementation later without touching callers
    """
    if not PYVIEWFACTOR_AVAILABLE:
        raise RuntimeError(
            "PyViewFactor backend requested but pyviewfactor is not installed. "
            "Install it in the IVF conda environment to use this backend."
        )

    logger.info("compute_svf_pyvf: computing SVF using PyViewFactor (experimental backend)")

    # Extract mesh faces and filter to above-ground obstructions
    points = full_mesh.points
    faces = full_mesh.faces

    if len(faces) == 0:
        # No faces at all → no obstructions
        return np.ones(len(ground_points), dtype=np.float64)

    ground_z = np.min(points[:, 2])
    z_tol = 1e-3

    obstruction_polys: list[np.ndarray] = []
    idx = 0
    n_faces = len(faces)
    while idx < n_faces:
        n_verts = faces[idx]
        idx += 1
        vert_ids = faces[idx : idx + n_verts]
        idx += n_verts
        poly = points[vert_ids]
        # Skip ground plane (approximately at ground_z)
        if np.max(poly[:, 2]) <= ground_z + z_tol:
            continue
        obstruction_polys.append(poly.astype(np.float64))

    if not obstruction_polys:
        # No above-ground obstructions → full sky
        return np.ones(len(ground_points), dtype=np.float64)

    # Build emitter facet arrays for PyViewFactor (shared across all observers)
    max_n2 = max(poly.shape[0] for poly in obstruction_polys)
    n_obstructions = len(obstruction_polys)

    pts2_arr = np.zeros((n_obstructions, max_n2, 3), dtype=np.float64)
    nverts2 = np.zeros((n_obstructions,), dtype=np.int32)
    for i, poly in enumerate(obstruction_polys):
        n = poly.shape[0]
        pts2_arr[i, :n, :] = poly
        nverts2[i] = n

    # Prepare output array
    svf_values = np.zeros(len(ground_points), dtype=np.float64)

    # Patch size around each observer (meters)
    patch_size = 1.0
    half = patch_size / 2.0

    for i, gp in enumerate(ground_points):
        # Observer point at evaluation height
        obs = np.array(gp, dtype=np.float64)
        obs[2] += evaluation_height

        # Small horizontal quad centered at observer
        p1 = np.array(
            [
                [obs[0] - half, obs[1] - half, obs[2]],
                [obs[0] + half, obs[1] - half, obs[2]],
                [obs[0] + half, obs[1] + half, obs[2]],
                [obs[0] - half, obs[1] + half, obs[2]],
            ],
            dtype=np.float64,
        )

        # Receiver array: same patch repeated for each obstruction facet
        pts1_arr = np.repeat(p1[None, ...], n_obstructions, axis=0)
        nverts1 = np.full((n_obstructions,), 4, dtype=np.int32)
        const_arr = np.zeros((n_obstructions,), dtype=np.float64)

        try:
            vf_vals = pvf.batch_compute_viewfactors(
                pts1_arr, nverts1, pts2_arr, nverts2, const_arr, verbose=False
            )
            # Sum of view factors to all obstructions
            F_tot = float(np.sum(vf_vals))
            svf = 1.0 - F_tot
            # Clamp to [0, 1] to guard against numerical noise
            svf_values[i] = max(0.0, min(1.0, svf))
        except Exception as e:  # pragma: no cover - defensive guard
            logger.warning(
                "PyViewFactor computation failed at point %d, falling back to CPU SVF. Error: %s",
                i,
                e,
            )
            svf_values[i] = compute_svf_cpu(
                ground_points=ground_points[i : i + 1],
                sky_patches=sky_patches,
                full_mesh=full_mesh,
                evaluation_height=evaluation_height,
            )[0]

    return svf_values


def compute_svf(
    ground_points: np.ndarray,
    sky_patches: np.ndarray,
    full_mesh: pv.PolyData,
    evaluation_height: float,
    use_gpu: Optional[bool] = None,
    gpu_batch_size: int = 1000,
    max_ray_length: float = 1000.0,
    ray_chunk_size: int = 1024,
    tri_chunk_size: int = 4096,
    backend: str = "auto",
) -> np.ndarray:
    """
    Compute SVF for each ground point with automatic CPU/GPU selection.
    
    This is the unified interface that automatically detects GPU availability
    and routes to the appropriate implementation.

    The `backend` argument controls which algorithm to use:
    - ``\"auto\"`` (default): existing CPU/GPU ray-tracing implementation
    - ``\"pyviewfactor\"``: use PyViewFactor-based implementation (when available)
    
    Args:
        ground_points: Array of shape (N, 3) with ground point coordinates
        sky_patches: Array of shape (M, 3) with sky patch centroids
        full_mesh: Full scene mesh (terrain + buildings) as PyVista PolyData
        evaluation_height: Height above ground for evaluation (meters)
        use_gpu: If True, use GPU if available; if False, use CPU; if None, auto-detect
        gpu_batch_size: Number of observer points to process in parallel (GPU only)
        max_ray_length: Maximum ray length (meters, GPU only)
        ray_chunk_size: Number of rays to intersect in each chunk (GPU only)
        tri_chunk_size: Number of triangles to intersect in each chunk (GPU only)
        backend: \"auto\" (default) for existing implementation, or \"pyviewfactor\"
        
    Returns:
        Array of SVF values (0-1) for each ground point
    """
    # Backend selection
    backend = backend.lower()

    # PyViewFactor backend (experimental – currently delegates to CPU implementation)
    if backend == "pyviewfactor":
        return compute_svf_pyvf(
            ground_points=ground_points,
            sky_patches=sky_patches,
            full_mesh=full_mesh,
            evaluation_height=evaluation_height,
        )

    # Default: existing CPU/GPU implementation
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = GPU_COMPUTE_AVAILABLE

    # Use GPU if requested and available
    if use_gpu and GPU_COMPUTE_AVAILABLE:
        logger.info("Using GPU-accelerated SVF computation")

        # Convert mesh to PyTorch3D format
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch3d_mesh = pv_mesh_to_pytorch3d(full_mesh, device=device)

        # Prepare observer points and sky patches
        observer_points_tensor = prepare_observer_points(
            ground_points,
            evaluation_height,
            device=device
        )
        sky_patches_tensor = prepare_sky_patches(sky_patches, device=device)

        # Compute SVF on GPU
        svf_tensor = compute_svf_gpu_batch(
            observer_points_tensor,
            sky_patches_tensor,
            pytorch3d_mesh,
            batch_size=gpu_batch_size,
            max_ray_length=max_ray_length,
            num_samples_per_ray=100,  # Unused in exact mode, kept for compatibility
            ray_chunk_size=ray_chunk_size,
            tri_chunk_size=tri_chunk_size,
        )

        # Convert back to numpy
        return svf_tensor.cpu().numpy()

    # Fallback: CPU implementation
    if use_gpu and not GPU_COMPUTE_AVAILABLE:
        logger.warning("GPU requested but not available, falling back to CPU")

    return compute_svf_cpu(
        ground_points,
        sky_patches,
        full_mesh,
        evaluation_height
    )


def get_compute_backend() -> str:
    """
    Get the current compute backend (CPU or GPU).
    
    Returns:
        String indicating backend: "cpu", "gpu", or "gpu_unavailable"
    """
    if GPU_COMPUTE_AVAILABLE:
        return "gpu"
    elif GPU_AVAILABLE:
        return "gpu_unavailable"  # PyTorch available but GPU compute not
    else:
        return "cpu"
