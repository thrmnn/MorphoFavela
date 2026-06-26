"""
SVF computation backends: ray-casting (primary), GPU, and PyViewFactor.

Supports two sky hemisphere discretization schemes:
  - ``"uniform"``: simple azimuth x elevation grid (legacy, biased)
  - ``"tregenza"``: Tregenza 145-patch equal-area subdivision (standard
    in building science; used by Radiance, DAYSIM, etc.)
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Optional: embree-backed multi_ray_trace
# Requires trimesh with a working embree backend (pyembree or embreex).
try:
    import trimesh

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

# Tregenza sky subdivision definition (CIE standard, Tregenza 1987).
# Each row: (centre_elevation_deg, number_of_patches, azimuth_spacing_deg)
# The 7 bands + 1 zenith cap yield 145 patches of approximately equal
# solid angle (~0.0435 sr each; full hemisphere = 2*pi sr).
TREGENZA_BANDS = [
    (6.0, 30, 12.0),
    (18.0, 30, 12.0),
    (30.0, 24, 15.0),
    (42.0, 24, 15.0),
    (54.0, 18, 20.0),
    (66.0, 12, 30.0),
    (78.0, 6, 60.0),
]
# Zenith cap is a single patch at elevation 90 deg.


def generate_tregenza_patches() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the Tregenza 145-patch sky subdivision.

    Returns a tuple ``(directions, weights)`` where:
      - *directions* is a 145x3 array of unit direction vectors (z >= 0).
      - *weights* is a 145-length array of solid-angle weights (sr) that
        sum to 2*pi (the solid angle of the hemisphere).

    The solid angle of each patch in a band is computed analytically from
    the band boundaries.  Band *k* spans from ``el_k - delta/2`` to
    ``el_k + delta/2`` where ``delta = 12 deg`` (the uniform band width
    in the Tregenza scheme).  The solid angle of a band ring is::

        Omega_band = 2 * pi * (sin(el_hi) - sin(el_lo))

    divided equally among the *n* patches in that band.  The zenith cap
    covers from 84 deg to 90 deg.
    """
    band_width_deg = 12.0  # each band spans 12 degrees of elevation
    dirs = []
    weights = []

    for el_centre_deg, n_patches, az_spacing_deg in TREGENZA_BANDS:
        el_lo = np.radians(el_centre_deg - band_width_deg / 2.0)
        el_hi = np.radians(el_centre_deg + band_width_deg / 2.0)
        el_centre = np.radians(el_centre_deg)

        # Solid angle of the full azimuthal band
        band_omega = 2.0 * np.pi * (np.sin(el_hi) - np.sin(el_lo))
        patch_omega = band_omega / n_patches

        for j in range(n_patches):
            az = np.radians(az_spacing_deg * (j + 0.5))
            dx = np.cos(el_centre) * np.cos(az)
            dy = np.cos(el_centre) * np.sin(az)
            dz = np.sin(el_centre)
            dirs.append([dx, dy, dz])
            weights.append(patch_omega)

    # Zenith cap: from 84 deg to 90 deg
    el_lo_zenith = np.radians(84.0)
    cap_omega = 2.0 * np.pi * (np.sin(np.pi / 2) - np.sin(el_lo_zenith))
    dirs.append([0.0, 0.0, 1.0])
    weights.append(cap_omega)

    directions = np.array(dirs)
    weights = np.array(weights)

    assert len(directions) == 145, f"Expected 145 patches, got {len(directions)}"

    logger.info(
        f"Tregenza sky: {len(directions)} patches, "
        f"total solid angle = {weights.sum():.4f} sr "
        f"(2*pi = {2 * np.pi:.4f})"
    )
    return directions, weights


def generate_sky_directions(n_patches: int = 145) -> np.ndarray:
    """
    Generate unit direction vectors over the upper hemisphere.

    Uses a simple azimuth x elevation grid that approximates *n_patches*
    directions.  Each direction is a unit vector; no solid-angle weighting
    is applied (for basic SVF this is acceptable; for high-accuracy work,
    use equal-area Tregenza patches via ``generate_tregenza_patches``).

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
    logger.info(f"Generated {len(dirs)} sky directions (uniform grid)")
    return dirs


# ---------------------------------------------------------------------------
# Ray-casting SVF (primary backend)
# ---------------------------------------------------------------------------


def _svf_for_point_obb(
    origin,
    sky_directions,
    obb_tree,
    max_ray_length,
    normal=None,
    sky_weights=None,
):
    """Compute SVF for a single point using a pre-built VTK OBB tree.

    This avoids the PyVista wrapper overhead on each ``ray_trace`` call and
    is roughly 2-3x faster per ray than ``PolyData.ray_trace``.

    Args:
        origin: Length-3 observer position.
        sky_directions: Mx3 unit direction vectors.
        obb_tree: Pre-built ``vtkOBBTree`` locator.
        max_ray_length: Maximum ray travel distance.
        normal: Optional length-3 surface normal for hemisphere filtering.
        sky_weights: Optional M-length solid-angle weights.

    Returns:
        SVF value in [0, 1].
    """
    import vtk

    if normal is not None:
        dots = sky_directions @ normal
        mask = dots > 0
        valid_dirs = sky_directions[mask]
        valid_weights = sky_weights[mask] if sky_weights is not None else None
    else:
        valid_dirs = sky_directions
        valid_weights = sky_weights

    if len(valid_dirs) == 0:
        return 0.0

    o = origin.tolist()

    if valid_weights is not None:
        # Weighted SVF: sum of visible solid-angle weights / total weights
        visible_weight = 0.0
        total_weight = valid_weights.sum()
        for j, d in enumerate(valid_dirs):
            end = (origin + d * max_ray_length).tolist()
            pts = vtk.vtkPoints()
            cell_ids = vtk.vtkIdList()
            obb_tree.IntersectWithLine(o, end, pts, cell_ids)
            if pts.GetNumberOfPoints() == 0:
                visible_weight += valid_weights[j]
        return visible_weight / total_weight if total_weight > 0 else 0.0
    else:
        # Unweighted SVF: count ratio (legacy behaviour)
        visible = 0
        for d in valid_dirs:
            end = (origin + d * max_ray_length).tolist()
            pts = vtk.vtkPoints()
            cell_ids = vtk.vtkIdList()
            obb_tree.IntersectWithLine(o, end, pts, cell_ids)
            if pts.GetNumberOfPoints() == 0:
                visible += 1
        return visible / len(valid_dirs)


def _svf_for_point_multi_ray(
    origin,
    sky_directions,
    tri_mesh,
    max_ray_length,
    normal=None,
    sky_weights=None,
):
    """Compute SVF for a single point using ``multi_ray_trace`` (embree).

    Batches all M direction rays into a single vectorized call, which is
    significantly faster than looping in Python.

    When *sky_weights* is provided, the SVF is computed as the ratio of
    visible solid-angle weight to total tested solid-angle weight.

    Args:
        origin: Length-3 observer position.
        sky_directions: Mx3 unit direction vectors.
        tri_mesh: Triangulated ``pv.PolyData`` scene mesh.
        max_ray_length: Maximum ray travel distance.
        normal: Optional length-3 surface normal for hemisphere filtering.
        sky_weights: Optional M-length solid-angle weights.

    Returns:
        SVF value in [0, 1].
    """
    if normal is not None:
        dots = sky_directions @ normal
        mask = dots > 0
        valid_dirs = sky_directions[mask]
        valid_weights = sky_weights[mask] if sky_weights is not None else None
    else:
        valid_dirs = sky_directions
        valid_weights = sky_weights

    if len(valid_dirs) == 0:
        return 0.0

    n_dirs = len(valid_dirs)
    origins = np.tile(origin, (n_dirs, 1))
    # multi_ray_trace takes directions, not endpoints
    directions = valid_dirs * max_ray_length

    _pts, intersection_rays, _cells = tri_mesh.multi_ray_trace(
        origins, directions, first_point=True, retry=True
    )

    hit_set = set(intersection_rays)

    if valid_weights is not None:
        # Weighted SVF
        total_weight = valid_weights.sum()
        blocked_weight = sum(valid_weights[r] for r in hit_set)
        return (total_weight - blocked_weight) / total_weight if total_weight > 0 else 0.0
    else:
        # Unweighted SVF: count ratio (legacy behaviour)
        n_hit = len(hit_set)
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
    sky_weights=None,
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
            observer_points[i],
            sky_directions,
            obb_tree,
            max_ray_length,
            normal,
            sky_weights=sky_weights,
        )
    return indices, svf_chunk


def _compute_chunk_multi_ray(
    indices,
    observer_points,
    sky_directions,
    max_ray_length,
    normals,
    mesh_file,
    sky_weights=None,
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
            observer_points[i],
            sky_directions,
            tri_mesh,
            max_ray_length,
            normal,
            sky_weights=sky_weights,
        )
    return indices, svf_chunk


def compute_svf_raycasting(
    observer_points: np.ndarray,
    scene_mesh: pv.PolyData,
    sky_directions: np.ndarray,
    max_ray_length: float = 500.0,
    normals: np.ndarray | None = None,
    sky_weights: np.ndarray | None = None,
    n_jobs: int = 1,
    checkpoint_path: Path | None = None,
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
        sky_weights: Optional M-length array of solid-angle weights for
            each sky direction.  When provided, the SVF is computed as
            ``sum(visible_weights) / sum(all_tested_weights)`` instead
            of the unweighted patch count ratio.
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

            worker = _compute_chunk_multi_ray if use_multi_ray else _compute_chunk_obb

            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(worker)(
                    chunk,
                    observer_points,
                    sky_directions,
                    max_ray_length,
                    normals,
                    mesh_file,
                    sky_weights=sky_weights,
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
            f"SVF complete: mean={svf.mean():.3f}, min={svf.min():.3f}, max={svf.max():.3f}"
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
                sky_weights=sky_weights,
            )
        else:
            svf[i] = _svf_for_point_obb(
                observer_points[i],
                sky_directions,
                obb_tree,
                max_ray_length,
                normal,
                sky_weights=sky_weights,
            )

        if (i + 1) % 50 == 0 or i == n_obs - 1:
            pbar.set_postfix(mean=f"{np.mean(svf[: i + 1]):.3f}", cur=f"{svf[i]:.3f}")
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
    Compute SVF on GPU.

    .. note:: The v1 GPU modules (``svf_gpu_utils``, ``svf_gpu_compute``)
       have been retired.  This stub remains so the ``backend="gpu"``
       code-path raises a clear error instead of an import failure.
    """
    raise NotImplementedError(
        "GPU backend not yet ported to v2. Use backend='raycasting' (CPU, parallelised) instead."
    )


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
    logger.info(f"PyViewFactor: {len(obstruction_ids)} obstruction faces (of {mesh.n_cells})")

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
    sky_model: str = "tregenza",
    normals: np.ndarray | None = None,
    max_ray_length: float = 500.0,
    n_jobs: int = 1,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 500,
    **kwargs,
) -> np.ndarray:
    """
    Unified SVF computation entry point.

    Args:
        observer_points: Nx3 observer positions.
        scene_mesh: Combined scene mesh (terrain + buildings).
        backend: ``"raycasting"`` | ``"gpu"`` | ``"pyviewfactor"``.
        n_sky_patches: Number of sky hemisphere directions (used only
            when ``sky_model="uniform"``; ignored for ``"tregenza"``
            which always produces exactly 145 patches).
        sky_model: Sky discretization scheme.  ``"tregenza"`` uses the
            CIE 145-patch equal-area subdivision with solid-angle
            weighting (recommended).  ``"uniform"`` uses the legacy
            azimuth x elevation grid with equal patch counting.
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
    if sky_model == "tregenza":
        sky_dirs, sky_weights = generate_tregenza_patches()
    elif sky_model == "uniform":
        sky_dirs = generate_sky_directions(n_sky_patches)
        sky_weights = None
    else:
        raise ValueError(f"Unknown sky_model: {sky_model!r}. Use 'tregenza' or 'uniform'.")

    if backend == "raycasting":
        return compute_svf_raycasting(
            observer_points,
            scene_mesh,
            sky_dirs,
            max_ray_length=max_ray_length,
            normals=normals,
            sky_weights=sky_weights,
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
