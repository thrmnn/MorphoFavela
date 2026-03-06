"""
Utilities for converting meshes to PyTorch3D format for GPU acceleration.

This module provides functions to convert PyVista meshes to PyTorch3D format
and prepare data for GPU-accelerated SVF computation.
"""

import torch
import numpy as np
import pyvista as pv
from pytorch3d.structures import Meshes
from typing import Optional


def pv_mesh_to_pytorch3d(
    pv_mesh: pv.PolyData,
    device: Optional[torch.device] = None
) -> Meshes:
    """
    Convert PyVista PolyData to PyTorch3D Meshes.
    
    Args:
        pv_mesh: PyVista PolyData mesh
        device: Torch device (cuda/cpu). If None, uses CUDA if available.
    
    Returns:
        PyTorch3D Meshes object
    
    Raises:
        ImportError: If PyTorch3D is not installed
        RuntimeError: If CUDA is requested but not available
    """
    try:
        from pytorch3d.structures import Meshes
    except ImportError:
        raise ImportError(
            "PyTorch3D is required for GPU acceleration. "
            "Install with: pip install pytorch3d"
        )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but CUDA is not available. "
            "Falling back to CPU."
        )
    
    # Extract vertices and faces
    # Filter out NaN values from mesh points
    points = pv_mesh.points.copy()
    valid_mask = ~np.isnan(points).any(axis=1)
    
    # Extract faces - PyVista stores faces in VTK format: [n_verts, v0, v1, ..., n_verts, v0, v1, ...]
    # PyTorch3D requires triangular faces, so we need to triangulate any non-triangular faces
    faces_list = []
    if pv_mesh.faces.size > 0:
        faces_array = pv_mesh.faces
        i = 0
        while i < len(faces_array):
            n_verts = int(faces_array[i])
            if i + n_verts < len(faces_array):
                verts = faces_array[i+1:i+1+n_verts]
                if n_verts == 3:
                    # Already triangular
                    faces_list.append(verts)
                elif n_verts == 4:
                    # Quadrilateral - split into two triangles
                    # Triangle 1: v0, v1, v2
                    faces_list.append(verts[[0, 1, 2]])
                    # Triangle 2: v0, v2, v3
                    faces_list.append(verts[[0, 2, 3]])
                elif n_verts > 4:
                    # Polygon with more than 4 vertices - triangulate using fan method
                    for j in range(1, n_verts - 1):
                        faces_list.append(verts[[0, j, j+1]])
                i += n_verts + 1
            else:
                break
    
    if len(faces_list) > 0:
        faces_old = np.array(faces_list, dtype=np.int64)
    else:
        faces_old = np.empty((0, 3), dtype=np.int64)
    
    if not valid_mask.all():
        n_invalid = np.sum(~valid_mask)
        print(f"  Warning: Found {n_invalid} vertices with NaN values, filtering them out")
        points = points[valid_mask]
        
        # Need to remap face indices
        # Create mapping from old index to new index
        old_to_new = np.full(len(valid_mask), -1, dtype=np.int64)
        new_idx = 0
        for old_idx, is_valid in enumerate(valid_mask):
            if is_valid:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        
        # Filter faces that reference only valid vertices
        valid_faces_mask = valid_mask[faces_old].all(axis=1) if len(faces_old) > 0 else np.array([], dtype=bool)
        faces_old = faces_old[valid_faces_mask]
        
        # Remap face indices
        if len(faces_old) > 0:
            faces = old_to_new[faces_old]
            
            # Remove faces that reference invalid vertices (shouldn't happen after filtering)
            invalid_face_mask = (faces < 0).any(axis=1)
            if invalid_face_mask.any():
                faces = faces[~invalid_face_mask]
        else:
            faces = np.empty((0, 3), dtype=np.int64)
        
        print(f"  After filtering: {len(points)} vertices, {len(faces)} faces")
    else:
        faces = faces_old
    
    vertices = torch.tensor(points, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Create PyTorch3D mesh
    # Note: PyTorch3D expects faces to be 0-indexed (already the case)
    mesh = Meshes(verts=[vertices], faces=[faces])
    
    return mesh


def prepare_observer_points(
    ground_points: np.ndarray,
    evaluation_height: float,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Prepare observer points for GPU computation.
    
    Args:
        ground_points: Array of shape (N, 3) with ground coordinates
        evaluation_height: Height above ground (meters)
        device: Torch device
    
    Returns:
        Tensor of shape (N, 3) with observer points
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure float dtype to avoid casting errors
    observer_points = ground_points.astype(np.float32).copy()
    observer_points[:, 2] += evaluation_height
    
    return torch.tensor(observer_points, dtype=torch.float32, device=device)


def prepare_sky_patches(
    sky_patches: np.ndarray,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Prepare sky patches for GPU computation.
    
    Args:
        sky_patches: Array of shape (M, 3) with sky patch centroids
        device: Torch device
    
    Returns:
        Tensor of shape (M, 3) with sky patches
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return torch.tensor(sky_patches, dtype=torch.float32, device=device)


def check_gpu_availability() -> dict:
    """
    Check GPU availability and return status information.
    
    Returns:
        Dictionary with GPU status information:
        - available: bool
        - device_name: str or None
        - device_count: int
        - cuda_version: str or None
    """
    info = {
        "available": torch.cuda.is_available(),
        "device_name": None,
        "device_count": 0,
        "cuda_version": None,
    }
    
    if info["available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    
    return info
