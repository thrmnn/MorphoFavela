# PyTorch3D GPU Acceleration Implementation Guide

## Overview

This document provides step-by-step instructions for implementing GPU-accelerated SVF computation using PyTorch3D. This will significantly speed up ray-casting operations by leveraging GPU parallelization.

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA support (compute capability 3.5+)
- **CUDA Toolkit** (version 11.0 or later recommended)
- **cuDNN** (for optimized operations)

### Software Requirements
- Python 3.8+
- PyTorch with CUDA support
- PyTorch3D
- NumPy, PyVista (existing dependencies)

## Installation Steps

### Step 1: Verify GPU Availability

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 11.8 (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Step 3: Install PyTorch3D

```bash
# Install from source (recommended for latest features)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Or install from PyPI (may have limited features)
pip install pytorch3d
```

### Step 4: Verify Installation

```python
import torch
import pytorch3d

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch3D version: {pytorch3d.__version__}")
```

## Implementation Architecture

### Core Components

1. **Mesh Conversion Module** (`src/svf_gpu_utils.py`)
   - Convert PyVista mesh to PyTorch3D format
   - Handle coordinate transformations
   - Optimize mesh representation

2. **GPU Ray-Casting Module** (`src/svf_gpu_compute.py`)
   - Batch ray-casting operations
   - GPU-accelerated intersection tests
   - Memory-efficient processing

3. **Integration Layer** (`scripts/compute_svf_streets_gpu.py`)
   - Wrapper around existing script
   - Automatic fallback to CPU if GPU unavailable
   - Maintains same API as CPU version

## Step-by-Step Implementation

### Phase 1: Mesh Conversion (Day 1)

#### Step 1.1: Create Mesh Conversion Utility

**File**: `src/svf_gpu_utils.py`

```python
"""
Utilities for converting meshes to PyTorch3D format for GPU acceleration.
"""

import torch
import numpy as np
import pyvista as pv
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from typing import Optional, Tuple


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
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract vertices and faces
    vertices = torch.tensor(pv_mesh.points, dtype=torch.float32, device=device)
    
    # Extract faces (PyVista uses different format)
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # Remove cell count
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Create PyTorch3D mesh
    # Note: PyTorch3D expects faces to be 0-indexed
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
    
    observer_points = ground_points.copy()
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
```

#### Step 1.2: Test Mesh Conversion

Create test script to verify conversion works correctly.

### Phase 2: GPU Ray-Casting Implementation (Day 2-3)

#### Step 2.1: Implement Batch Ray-Casting

**File**: `src/svf_gpu_compute.py`

```python
"""
GPU-accelerated SVF computation using PyTorch3D.
"""

import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    look_at_view_transform,
    FoVPerspectiveCameras,
)
from typing import Optional, Tuple


def compute_svf_gpu_batch(
    observer_points: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    batch_size: int = 1000,
    max_ray_length: float = 1000.0
) -> torch.Tensor:
    """
    Compute SVF using GPU-accelerated batch ray-casting.
    
    Args:
        observer_points: Tensor of shape (N, 3) with observer positions
        sky_patches: Tensor of shape (M, 3) with sky patch directions
        mesh: PyTorch3D Meshes object
        batch_size: Number of rays to process in parallel
        max_ray_length: Maximum ray length (meters)
    
    Returns:
        Tensor of shape (N,) with SVF values
    """
    device = observer_points.device
    n_points = observer_points.shape[0]
    n_patches = sky_patches.shape[0]
    
    svf_values = torch.zeros(n_points, device=device)
    
    # Process in batches to manage memory
    for i in range(0, n_points, batch_size):
        batch_end = min(i + batch_size, n_points)
        batch_observers = observer_points[i:batch_end]
        
        # Compute SVF for this batch
        batch_svf = _compute_batch_svf(
            batch_observers, sky_patches, mesh, max_ray_length
        )
        
        svf_values[i:batch_end] = batch_svf
    
    return svf_values


def _compute_batch_svf(
    observers: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    max_ray_length: float
) -> torch.Tensor:
    """
    Compute SVF for a batch of observers.
    
    This uses PyTorch3D's ray-mesh intersection capabilities.
    """
    device = observers.device
    n_observers = observers.shape[0]
    n_patches = sky_patches.shape[0]
    
    # Expand observers and patches for batch processing
    # Shape: (n_observers, n_patches, 3)
    observers_expanded = observers.unsqueeze(1).expand(-1, n_patches, -1)
    patches_expanded = sky_patches.unsqueeze(0).expand(n_observers, -1, -1)
    
    # Compute ray directions
    ray_directions = patches_expanded - observers_expanded
    ray_lengths = torch.norm(ray_directions, dim=-1, keepdim=True)
    ray_directions = ray_directions / (ray_lengths + 1e-8)
    
    # Cast rays and check intersections
    # Note: This is a simplified version - full implementation would use
    # PyTorch3D's ray-mesh intersection functions
    visible_patches = _check_ray_intersections(
        observers_expanded, ray_directions, ray_lengths, mesh, max_ray_length
    )
    
    # Compute SVF: visible patches / total patches
    svf = visible_patches.sum(dim=1) / n_patches
    
    return svf


def _check_ray_intersections(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    ray_lengths: torch.Tensor,
    mesh: Meshes,
    max_length: float
) -> torch.Tensor:
    """
    Check which rays intersect the mesh.
    
    Returns:
        Boolean tensor of shape (n_observers, n_patches) indicating visibility
    """
    # This is a placeholder - actual implementation would use:
    # - pytorch3d.ops.sample_points_from_meshes for sampling
    # - Custom ray-mesh intersection kernel
    # - Or rasterization-based approach
    
    # For now, return a simplified version
    # Full implementation requires custom CUDA kernel or PyTorch3D extensions
    
    raise NotImplementedError(
        "Ray-mesh intersection needs to be implemented using "
        "PyTorch3D's ray casting utilities or custom CUDA kernel"
    )
```

#### Step 2.2: Alternative Approach - Rasterization-Based

Since direct ray-mesh intersection in PyTorch3D is complex, consider a rasterization-based approach:

```python
def compute_svf_rasterization(
    observer_points: torch.Tensor,
    sky_patches: torch.Tensor,
    mesh: Meshes,
    image_size: int = 512
) -> torch.Tensor:
    """
    Compute SVF using rasterization (faster but slightly less accurate).
    
    This approach renders the scene from each observer point and counts
    visible sky pixels.
    """
    device = observer_points.device
    n_points = observer_points.shape[0]
    
    # Setup rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # Create renderer
    lights = PointLights(device=device, location=[[0.0, 0.0, 10.0]])
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader = HardPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    
    svf_values = torch.zeros(n_points, device=device)
    
    for i, observer in enumerate(observer_points):
        # Position camera at observer point, looking up
        R, T = look_at_view_transform(
            eye=observer.unsqueeze(0),
            at=observer.unsqueeze(0) + torch.tensor([[0, 0, 1]], device=device),
            up=torch.tensor([[0, 1, 0]], device=device),
        )
        
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        # Render scene
        images = renderer(mesh, cameras=cameras)
        
        # Count visible sky (pixels with no geometry)
        # SVF = (transparent pixels) / (total pixels)
        # This is a simplified version - full implementation would
        # properly identify sky vs. geometry
        
        # Placeholder
        svf_values[i] = 0.5  # Would compute actual SVF from rendered image
    
    return svf_values
```

### Phase 3: Integration (Day 4)

#### Step 3.1: Create GPU-Enabled Script

**File**: `scripts/compute_svf_streets_gpu.py`

This will be a modified version of `compute_svf_streets.py` that:
1. Checks for GPU availability
2. Converts mesh to PyTorch3D format
3. Uses GPU computation when available
4. Falls back to CPU if GPU unavailable
5. Maintains same output format as original

#### Step 3.2: Add Command-Line Options

```python
parser.add_argument(
    '--use-gpu',
    action='store_true',
    help='Use GPU acceleration (requires CUDA-capable GPU)'
)
parser.add_argument(
    '--gpu-batch-size',
    type=int,
    default=1000,
    help='Batch size for GPU processing'
)
```

### Phase 4: Testing and Validation (Day 5)

#### Step 4.1: Unit Tests

Create tests to verify:
- Mesh conversion accuracy
- GPU vs CPU result comparison
- Memory usage
- Performance benchmarks

#### Step 4.2: Validation Script

Compare GPU results with CPU results on small dataset to ensure accuracy.

## Performance Optimization Tips

### Memory Management

1. **Batch Processing**: Process points in batches to avoid GPU memory overflow
2. **Gradient Disabling**: Use `torch.no_grad()` for inference
3. **Memory Clearing**: Explicitly clear GPU cache when needed

```python
with torch.no_grad():
    # Computation here
    pass

torch.cuda.empty_cache()  # Clear GPU cache if needed
```

### Batch Size Tuning

- Start with batch_size=1000
- Increase if GPU memory allows
- Monitor with `nvidia-smi` during execution

### Mixed Precision (Optional)

For further speedup, consider using FP16:

```python
with torch.cuda.amp.autocast():
    # Computation here
    pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size
   - Process fewer points at once
   - Use gradient checkpointing

2. **Slow Performance**
   - Verify GPU is being used: `torch.cuda.is_available()`
   - Check GPU utilization: `nvidia-smi`
   - Ensure data is on GPU: `tensor.device`

3. **Installation Issues**
   - Ensure CUDA version matches PyTorch version
   - Reinstall PyTorch3D from source if needed
   - Check PyTorch3D compatibility with your PyTorch version

## Expected Performance

### Benchmarks (Estimated)

- **CPU (current)**: ~7 seconds per point
- **GPU (PyTorch3D)**: ~0.1-0.5 seconds per point
- **Speedup**: 10-70× depending on GPU and batch size

### For 33,387 points with 300 patches:

- **CPU**: ~65 hours
- **GPU (RTX 3090)**: ~1-3 hours
- **GPU (RTX 4090)**: ~0.5-1.5 hours

## Implementation Checklist

- [ ] Install PyTorch with CUDA support
- [ ] Install PyTorch3D
- [ ] Create `src/svf_gpu_utils.py` (mesh conversion)
- [ ] Create `src/svf_gpu_compute.py` (GPU computation)
- [ ] Implement ray-mesh intersection or rasterization approach
- [ ] Create `scripts/compute_svf_streets_gpu.py` (integration)
- [ ] Add GPU fallback to CPU
- [ ] Write unit tests
- [ ] Validate GPU vs CPU results
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Update requirements.txt

## Next Steps After Implementation

1. **Profile GPU code** to identify further optimizations
2. **Compare with CPU version** to ensure accuracy
3. **Optimize batch sizes** for your specific GPU
4. **Consider hybrid approach** (GPU for large batches, CPU for small)

## References

- [PyTorch3D Documentation](https://pytorch3d.readthedocs.io/)
- [PyTorch3D GitHub](https://github.com/facebookresearch/pytorch3d)
- [PyTorch CUDA Guide](https://pytorch.org/docs/stable/cuda.html)
