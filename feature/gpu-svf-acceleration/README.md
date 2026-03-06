# GPU-Accelerated SVF Computation

This branch implements GPU acceleration for Sky View Factor (SVF) computation using PyTorch3D.

## Status

🚧 **Under Development** - Initial structure and utilities created.

## Implementation Plan

Following the guide in `docs/PYTORCH3D_GPU_IMPLEMENTATION.md`:

### Phase 1: Mesh Conversion ✅
- [x] Create `src/svf_gpu_utils.py` with mesh conversion functions
- [x] Implement `pv_mesh_to_pytorch3d()`
- [x] Implement `prepare_observer_points()`
- [x] Implement `prepare_sky_patches()`
- [x] Add GPU availability checking

### Phase 2: GPU Ray-Casting (In Progress)
- [ ] Implement ray-mesh intersection using PyTorch3D
- [ ] Create batch processing for memory efficiency
- [ ] Optimize ray-casting algorithm
- [ ] Add fallback to CPU if GPU unavailable

### Phase 3: Integration
- [ ] Create `scripts/compute_svf_streets_gpu.py`
- [ ] Integrate with existing `compute_svf_streets.py`
- [ ] Add command-line options for GPU usage
- [ ] Maintain backward compatibility

### Phase 4: Testing and Validation
- [ ] Unit tests for mesh conversion
- [ ] Compare GPU vs CPU results
- [ ] Performance benchmarking
- [ ] Memory usage profiling

## Prerequisites

- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- PyTorch3D

See `docs/PYTORCH3D_GPU_IMPLEMENTATION.md` for installation instructions.

## Usage

Once implemented, usage will be:

```bash
python scripts/compute_svf_streets_gpu.py \
    --stl data/rocinha/rocinha.stl \
    --roads data/rocinha/raw/roads_rocinha.shp \
    --use-gpu \
    --spacing 1.5 \
    --sky-patches 300
```

## Files

- `src/svf_gpu_utils.py` - Mesh conversion and data preparation utilities
- `src/svf_gpu_compute.py` - GPU-accelerated SVF computation (placeholder)
- `docs/PYTORCH3D_GPU_IMPLEMENTATION.md` - Implementation guide

## Expected Performance

- **Current (CPU)**: ~7 seconds per point → ~65 hours for 33,387 points
- **Target (GPU)**: ~0.1-0.5 seconds per point → ~1-3 hours for 33,387 points
- **Speedup**: 10-70× depending on GPU
