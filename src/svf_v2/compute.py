"""
SVF computation backends: ray-casting (primary), GPU, and PyViewFactor.
"""

import numpy as np
import pyvista as pv
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional: embree-backed multi_ray_trace
# Requires trimesh with a working embree backend (pyembree or embreex).
try:
    import trimesh  # noqa: F401

    _MULTI_RAY_TRACE_AVAILABLE = bool(getattr(trimesh.ray, "has_embree", False))
except Exception:
    _MULTI_RAY_TRACE_AVAILABLE = False

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


def _svf_for_point_obb(origin, sky_directions, obb_tree, max_ray_length, normal=None):
    """Compute SVF for a single point using a pre-built VTK OBB tree.

    This avoids the PyVista wrapper overhead on each ``ray_trace`` call and
    is roughly 2-3x faster per ray than ``PolyData.ray_trace``.

    Args:
        origin: Length-3 observer position.
        sky_directions: Mx3 unit direction vectors.
        obb_tree: Pre-built ``vtkOBBTree`` locator.
        max_ray_length: Maximum ray travel distance.
        normal: Optional length-3 surface normal for hemisphere filtering.

    Returns:
        SVF value in [0, 1].
    """
    import vtk

    if normal is not None:
        dots = sky_directions @ normal
        valid_dirs = sky_directions[dots > 0]
    else:
        valid_dirs = sky_directions

    if len(valid_dirs) == 0:
        return 0.0

    visible = 0
    o = origin.tolist()
    for d in valid_dirs:
        end = (origin + d * max_ray_length).tolist()
        pts = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        obb_tree.IntersectWithLine(o, end, pts, cell_ids)
        if pts.GetNumberOfPoints() == 0:
            visible += 1

    return visible / len(valid_dirs)


def _svf_for_point_multi_ray(
    origin, sky_directions, tri_mesh, max_ray_length, normal=None
):
    """Compute SVF for a single point using ``multi_ray_trace`` (embree).

    Batches all M direction rays into a single vectorized call, which is
    significantly faster than looping in Python.

    Args:
        origin: Length-3 observer position.
        sky_directions: Mx3 unit direction vectors.
        tri_mesh: Triangulated ``pv.PolyData`` scene mesh.
        max_ray_length: Maximum ray travel distance.
        normal: Optional length-3 surface normal for hemisphere filtering.

    Returns:
        SVF value in [0, 1].
    """
    if normal is not None:
        dots = sky_directions @ normal
        valid_dirs = sky_directions[dots > 0]
    else:
        valid_dirs = sky_directions

    if len(valid_dirs) == 0:
        return 0.0

    n_dirs = len(valid_dirs)
    origins = np.tile(origin, (n_dirs, 1))
    # multi_ray_trace takes directions, not endpoints
    directions = valid_dirs * max_ray_length

    _pts, intersection_rays, _cells = tri_mesh.multi_ray_trace(
        origins, directions, first_point=True, retry=True
    )

    n_hit = len(set(intersection_rays))
    return (n_dirs - n_hit) / n_dirs


def _build_obb_tree(mesh):
    """Build a VTK OBB tree locator from a PyVista mesh.

    Returns:
        A ``vtkOBBTree`` instance ready for ``IntersectWithLine`` calls.
    """
    from vtkmodules.vtkFiltersGeneral import vtkOBBTree

    obb = vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()
    return obb


def _compute_chunk_obb(
    indices,
    observer_points,
    sky_directions,
    max_ray_length,
    normals,
    mesh_file,
):
    """Worker function for joblib: compute SVF for a chunk of indices.

    The scene mesh is loaded from a temporary VTK file so that the worker
    process does not need to pickle the PyVista object.  The OBB tree is
    built once per worker.

    Returns:
        Tuple of (indices, svf_values).
    """
    mesh = pv.read(mesh_file)
    obb_tree = _build_obb_tree(mesh)
    svf_chunk = np.empty(len(indices))
    for k, i in enumerate(indices):
        normal = normals[i] if normals is not None else None
        svf_chunk[k] = _svf_for_point_obb(
            observer_points[i], sky_directions, obb_tree, max_ray_length, normal
        )
    return indices, svf_chunk


def _compute_chunk_multi_ray(
    indices,
    observer_points,
    sky_directions,
    max_ray_length,
    normals,
    mesh_file,
):
    """Worker function for joblib using multi_ray_trace (embree).

    Returns:
        Tuple of (indices, svf_values).
    """
    mesh = pv.read(mesh_file)
    tri_mesh = mesh.triangulate()
    svf_chunk = np.empty(len(indices))
    for k, i in enumerate(indices):
        normal = normals[i] if normals is not None else None
        svf_chunk[k] = _svf_for_point_multi_ray(
            observer_points[i], sky_directions, tri_mesh, max_ray_length, normal
        )
    return indices, svf_chunk


def compute_svf_raycasting(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    sky_directions: np.ndarray,
    max_ray_length: float = 500.0,
    normals: Optional[np.ndarray] = None,
    n_jobs: int = 1,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 500,
) -> np.ndarray:
    """
    Compute SVF via ray-tracing with optional parallelization.

    Optimizations applied in order of preference:

    1. **multi_ray_trace** (requires trimesh + embree): batches all M
       direction rays per point into a single vectorized call.
    2. **VTK OBB tree**: pre-builds the OBB tree once and reuses it for
       all rays, bypassing per-call PyVista overhead (~2-3x faster).
    3. **joblib parallelization** (``n_jobs > 1``): distributes observer
       points across multiple processes.  Each worker loads its own copy
       of the mesh from a temporary VTK file and builds a local OBB tree
       (or uses ``multi_ray_trace``).

    When ``n_jobs=1`` the computation runs in-process with a progress bar
    and optional checkpointing.

    Args:
        observer_points: Nx3 observer positions.
        scene_mesh: Combined terrain + buildings mesh.
        sky_directions: Mx3 unit direction vectors.
        max_ray_length: Maximum ray travel distance (m).
        normals: Optional Nx3 surface normals (for facade points).
            If provided, only sky directions in the forward hemisphere
            of each point are tested.
        n_jobs: Number of parallel workers.  ``1`` = sequential (default),
            ``-1`` = use all available cores.
        checkpoint_path: Optional path for checkpoint file (.npz).
            If provided and the file exists, resumes from last checkpoint.
            Only effective when ``n_jobs=1``.
        checkpoint_interval: Save checkpoint every N points (default 500).

    Returns:
        N-length array of SVF values in [0, 1].
    """
    n_obs = len(observer_points)

    # Resolve n_jobs: -1 means all cores
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, n_jobs)

    # Decide ray strategy
    use_multi_ray = _MULTI_RAY_TRACE_AVAILABLE

    # -------------------------------------------------------------------
    # Parallel path (n_jobs > 1)
    # -------------------------------------------------------------------
    if n_jobs > 1:
        from joblib import Parallel, delayed

        logger.info(
            f"Parallel SVF: {n_obs} points, {n_jobs} workers, "
            f"strategy={'multi_ray_trace' if use_multi_ray else 'obb_tree'}"
        )

        if checkpoint_path is not None:
            logger.warning(
                "Checkpointing is not supported in parallel mode (n_jobs>1); "
                "checkpoint_path will be ignored"
            )

        # Write mesh to a temporary file so workers can load it
        tmp_dir = tempfile.mkdtemp(prefix="svf_parallel_")
        mesh_file = os.path.join(tmp_dir, "scene.vtk")
        scene_mesh.save(mesh_file)

        try:
            # Split indices into roughly equal chunks (one per worker)
            all_indices = np.arange(n_obs)
            chunks = np.array_split(all_indices, n_jobs)
            chunks = [c for c in chunks if len(c) > 0]

            worker = (
                _compute_chunk_multi_ray if use_multi_ray else _compute_chunk_obb
            )

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(worker)(
                    chunk,
                    observer_points,
                    sky_directions,
                    max_ray_length,
                    normals,
                    mesh_file,
                )
                for chunk in chunks
            )

            svf = np.zeros(n_obs)
            for chunk_indices, chunk_svf in results:
                svf[chunk_indices] = chunk_svf

        finally:
            # Clean up temporary mesh file
            try:
                os.remove(mesh_file)
                os.rmdir(tmp_dir)
            except OSError:
                pass

        logger.info(
            f"SVF complete: mean={svf.mean():.3f}, "
            f"min={svf.min():.3f}, max={svf.max():.3f}"
        )
        return svf

    # -------------------------------------------------------------------
    # Sequential path (n_jobs == 1) with progress bar and checkpointing
    # -------------------------------------------------------------------
    svf = np.zeros(n_obs)
    start_index = 0

    # Resume from checkpoint if it exists
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            data = np.load(checkpoint_path)
            saved_svf = data["svf"]
            saved_index = int(data["last_index"])
            if len(saved_svf) == n_obs:
                svf[:] = saved_svf
                start_index = saved_index + 1
                logger.info(
                    f"Resumed from checkpoint at index {saved_index} "
                    f"({start_index}/{n_obs} points done)"
                )
            else:
                logger.warning(
                    f"Checkpoint array length ({len(saved_svf)}) does not match "
                    f"observer count ({n_obs}); starting from scratch"
                )

    # Choose in-process strategy
    if use_multi_ray:
        tri_mesh = scene_mesh.triangulate()
        logger.info("Using multi_ray_trace (embree) for sequential SVF")
    else:
        obb_tree = _build_obb_tree(scene_mesh)
        logger.info("Using VTK OBB tree for sequential SVF")

    pbar = tqdm(total=n_obs, desc="SVF ray-casting", unit="pts", initial=start_index)

    for i in range(start_index, n_obs):
        normal = normals[i] if normals is not None else None

        if use_multi_ray:
            svf[i] = _svf_for_point_multi_ray(
                observer_points[i],
                sky_directions,
                tri_mesh,
                max_ray_length,
                normal,
            )
        else:
            svf[i] = _svf_for_point_obb(
                observer_points[i],
                sky_directions,
                obb_tree,
                max_ray_length,
                normal,
            )

        if (i + 1) % 50 == 0 or i == n_obs - 1:
            pbar.set_postfix(
                mean=f"{np.mean(svf[: i + 1]):.3f}", cur=f"{svf[i]:.3f}"
            )
        pbar.update(1)

        # Save checkpoint periodically
        if checkpoint_path is not None and (i + 1) % checkpoint_interval == 0:
            np.savez(checkpoint_path, svf=svf, last_index=i)
            logger.info(f"Checkpoint saved at index {i} ({i + 1}/{n_obs})")

    pbar.close()

    # Clean up checkpoint on successful completion
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Computation complete; checkpoint file removed")

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
    n_jobs: int = 1,
    checkpoint_path: Optional[Path] = None,
    checkpoint_interval: int = 500,
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
        n_jobs: Number of parallel workers (raycasting backend only).
            ``1`` = sequential (default), ``-1`` = all available cores.
        checkpoint_path: Optional path for checkpoint file (.npz).
            Only used by the raycasting backend.
        checkpoint_interval: Save checkpoint every N points (default 500).
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
            n_jobs=n_jobs,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
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
