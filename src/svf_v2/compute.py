"""
SVF computation backends: ray-casting (primary), GPU, and PyViewFactor.
"""

import numpy as np
import pyvista as pv
import logging
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional: PyViewFactor
try:
    import pyviewfactor as pvf

    PYVIEWFACTOR_AVAILABLE = True
except Exception:
    pvf = None
    PYVIEWFACTOR_AVAILABLE = False

# Optional: GPU
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    torch = None


# ---------------------------------------------------------------------------
# Sky hemisphere discretization
# ---------------------------------------------------------------------------


def generate_sky_directions(n_patches: int = 145) -> np.ndarray:
    """
    Generate unit direction vectors over the upper hemisphere.

    Uses a simple azimuth x elevation grid that approximates *n_patches*
    directions.  Each direction is a unit vector; no solid-angle weighting
    is applied (for basic SVF this is acceptable; for high-accuracy work,
    use equal-area Tregenza patches).

    Returns:
        Mx3 array of unit direction vectors (z >= 0).
    """
    az_steps = int(np.sqrt(n_patches * 2))
    el_steps = max(1, n_patches // az_steps)

    dirs = []
    for ai in range(az_steps):
        az = 2 * np.pi * ai / az_steps
        for ei in range(el_steps):
            el = np.pi / 2 * (ei + 0.5) / el_steps
            dx = np.cos(el) * np.cos(az)
            dy = np.cos(el) * np.sin(az)
            dz = np.sin(el)
            dirs.append([dx, dy, dz])

    dirs = np.array(dirs)
    logger.info(f"Generated {len(dirs)} sky directions")
    return dirs


# ---------------------------------------------------------------------------
# Ray-casting SVF (primary backend)
# ---------------------------------------------------------------------------


def compute_svf_raycasting(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    sky_directions: np.ndarray,
    max_ray_length: float = 500.0,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute SVF via PyVista ray-tracing.

    For each observer, cast a ray toward each sky direction and check for
    intersection with the scene mesh.

    Args:
        observer_points: Nx3 observer positions.
        scene_mesh: Combined terrain + buildings mesh.
        sky_directions: Mx3 unit direction vectors.
        max_ray_length: Maximum ray travel distance (m).
        normals: Optional Nx3 surface normals (for facade points).
            If provided, only sky directions in the forward hemisphere
            of each point are tested.

    Returns:
        N-length array of SVF values in [0, 1].
    """
    n_obs = len(observer_points)
    svf = np.zeros(n_obs)

    pbar = tqdm(total=n_obs, desc="SVF ray-casting", unit="pts")

    for i in range(n_obs):
        origin = observer_points[i]

        if normals is not None:
            n_vec = normals[i]
            # Only test directions in the forward hemisphere
            dots = sky_directions @ n_vec
            valid_dirs = sky_directions[dots > 0]
        else:
            valid_dirs = sky_directions

        if len(valid_dirs) == 0:
            svf[i] = 0.0
            pbar.update(1)
            continue

        visible = 0
        for d in valid_dirs:
            end = origin + d * max_ray_length
            hits, _ = scene_mesh.ray_trace(origin, end)
            if len(hits) == 0:
                visible += 1

        svf[i] = visible / len(valid_dirs)

        if (i + 1) % 50 == 0 or i == n_obs - 1:
            pbar.set_postfix(mean=f"{np.mean(svf[: i + 1]):.3f}", cur=f"{svf[i]:.3f}")
        pbar.update(1)

    pbar.close()
    return svf


# ---------------------------------------------------------------------------
# GPU backend (optional) -- delegates to existing code
# ---------------------------------------------------------------------------


def compute_svf_gpu(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    sky_directions: np.ndarray,
    batch_size: int = 1000,
    max_ray_length: float = 500.0,
    **kwargs,
) -> np.ndarray:
    """
    Compute SVF on GPU using existing Moller-Trumbore backend.

    UTM coordinates exceed float32 precision, so we subtract the scene
    centroid before GPU computation and add it back after.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    from src.svf_gpu_utils import pv_mesh_to_pytorch3d, check_gpu_availability
    from src.svf_gpu_compute import compute_svf_gpu_batch

    device = torch.device("cuda")
    check_gpu_availability()

    # Centre-shift for float32 precision
    centroid = scene_mesh.center
    shifted_points = observer_points - centroid
    shifted_mesh = scene_mesh.copy()
    shifted_mesh.points = shifted_mesh.points - centroid

    pytorch3d_mesh = pv_mesh_to_pytorch3d(shifted_mesh, device=device)

    # Sky directions are unit vectors (relative), no shift needed
    # Convert directions to "sky patch centroids" at max_ray_length
    sky_pts = sky_directions * max_ray_length
    sky_tensor = torch.tensor(sky_pts, dtype=torch.float32, device=device)

    # Observer points as tensor
    obs_tensor = torch.tensor(shifted_points, dtype=torch.float32, device=device)

    svf_tensor = compute_svf_gpu_batch(
        obs_tensor,
        sky_tensor,
        pytorch3d_mesh,
        batch_size=batch_size,
        max_ray_length=max_ray_length,
        num_samples_per_ray=100,
        **kwargs,
    )
    return svf_tensor.cpu().numpy()


# ---------------------------------------------------------------------------
# PyViewFactor backend (optional / validation)
# ---------------------------------------------------------------------------


def compute_svf_pyviewfactor(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    patch_size: float = 0.5,
) -> np.ndarray:
    """
    Compute SVF using PyViewFactor (per-facet view factor summation).

    Uses the correct API: ``pvf.compute_viewfactor(cell1, cell2)`` on
    individual PyVista cells, NOT ``batch_compute_viewfactors`` which is
    fragile.

    Pre-filters mesh: removes degenerate faces (area < 1e-4 m^2) and
    ground-facing faces.
    """
    if not PYVIEWFACTOR_AVAILABLE:
        raise RuntimeError("pyviewfactor is not installed")

    n_obs = len(observer_points)
    svf = np.zeros(n_obs)

    # Pre-filter mesh: compute face areas and normals
    mesh = scene_mesh.compute_normals(point_normals=False, cell_normals=True)
    areas = mesh.compute_cell_sizes()["Area"]
    normals_z = mesh.cell_normals[:, 2]

    # Keep only above-ground, non-degenerate faces
    keep = (areas > 1e-4) & (normals_z < 0.9)  # exclude near-horizontal ground
    obstruction_ids = np.where(keep)[0]
    logger.info(
        f"PyViewFactor: {len(obstruction_ids)} obstruction faces (of {mesh.n_cells})"
    )

    if len(obstruction_ids) == 0:
        return np.ones(n_obs)

    half = patch_size / 2

    for i in tqdm(range(n_obs), desc="SVF PyViewFactor"):
        obs = observer_points[i]

        # Create small horizontal receiver quad
        receiver_pts = np.array(
            [
                [obs[0] - half, obs[1] - half, obs[2]],
                [obs[0] + half, obs[1] - half, obs[2]],
                [obs[0] + half, obs[1] + half, obs[2]],
                [obs[0] - half, obs[1] + half, obs[2]],
            ]
        )
        receiver = pv.PolyData(receiver_pts, [4, 0, 1, 2, 3])

        total_vf = 0.0
        for cid in obstruction_ids:
            cell = mesh.extract_cells([cid])
            try:
                vf = pvf.compute_viewfactor(receiver, cell)
                if np.isfinite(vf) and vf > 0:
                    total_vf += vf
            except Exception:
                continue

        svf[i] = max(0.0, min(1.0, 1.0 - total_vf))

    return svf


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def compute_svf(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    backend: str = "raycasting",
    n_sky_patches: int = 145,
    normals: Optional[np.ndarray] = None,
    max_ray_length: float = 500.0,
    **kwargs,
) -> np.ndarray:
    """
    Unified SVF computation entry point.

    Args:
        observer_points: Nx3 observer positions.
        scene_mesh: Combined scene mesh (terrain + buildings).
        backend: ``"raycasting"`` | ``"gpu"`` | ``"pyviewfactor"``.
        n_sky_patches: Number of sky hemisphere directions.
        normals: Optional Nx3 normals for facade points.
        max_ray_length: Maximum ray distance (m).
        **kwargs: Forwarded to the chosen backend.

    Returns:
        N-length array of SVF values in [0, 1].
    """
    sky_dirs = generate_sky_directions(n_sky_patches)

    if backend == "raycasting":
        return compute_svf_raycasting(
            observer_points,
            scene_mesh,
            sky_dirs,
            max_ray_length=max_ray_length,
            normals=normals,
        )
    elif backend == "gpu":
        return compute_svf_gpu(
            observer_points,
            scene_mesh,
            sky_dirs,
            max_ray_length=max_ray_length,
            **kwargs,
        )
    elif backend == "pyviewfactor":
        return compute_svf_pyviewfactor(
            observer_points,
            scene_mesh,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
