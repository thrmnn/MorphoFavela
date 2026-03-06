# SVF Computation Optimization Proposal

## Current Bottleneck Analysis

### Current Implementation
- **Method**: Sequential ray-casting using PyVista's `ray_trace()` (CPU-based VTK)
- **Complexity**: O(N × M) where N = points, M = sky patches
- **Current Performance**: ~7 seconds per point with 300 sky patches
- **Bottleneck**: Each `ray_trace()` call processes the entire mesh sequentially

### Performance Breakdown (for 33,387 points, 300 patches)
- Total ray casts: 33,387 × 300 = **10,016,100 ray casts**
- Current time: ~65 hours
- Target: 2 hours (36× speedup needed)

---

## Optimization Strategies

### Strategy 1: CPU Parallelization (EASIEST, HIGHEST IMPACT)

**Approach**: Use multiprocessing to parallelize point processing

**Implementation**:
- Split points into chunks
- Process chunks in parallel using `multiprocessing.Pool`
- Each worker processes a subset of points independently

**Expected Speedup**: 4-8× (depending on CPU cores)
- 8 cores → ~8-16 hours (still not meeting 2-hour target alone)
- 16 cores → ~4-8 hours

**Pros**:
- ✅ Easy to implement (minimal code changes)
- ✅ No new dependencies
- ✅ Works on any system
- ✅ Maintains exact same algorithm/accuracy

**Cons**:
- ❌ Limited by CPU cores
- ❌ Won't reach 2-hour target alone
- ❌ Memory overhead (each worker loads mesh)

**Code Changes**: ~50 lines
**Risk**: Low
**Effort**: 2-4 hours

---

### Strategy 2: Spatial Indexing / Early Termination (MEDIUM EFFORT, GOOD SPEEDUP)

**Approach**: Use spatial data structures to accelerate ray-mesh intersection

**Options**:
1. **Octree/BVH**: Build spatial index of mesh, use for faster ray intersection
2. **Early termination**: Stop at first intersection (already doing this, but can optimize)
3. **Mesh simplification**: Reduce mesh complexity for ray-casting (LOD)

**Expected Speedup**: 2-5× per ray cast
- Combined with parallelization: ~1-4 hours total

**Implementation**:
- Use `trimesh` or `pymeshlab` for spatial indexing
- Or implement custom BVH/Octree
- Or use PyVista's built-in spatial locator

**Pros**:
- ✅ Significant speedup per ray
- ✅ Works with existing code
- ✅ Can combine with other strategies

**Cons**:
- ❌ Requires spatial index construction (one-time cost)
- ❌ More complex implementation
- ❌ May need mesh preprocessing

**Code Changes**: ~100-200 lines
**Risk**: Medium
**Effort**: 1-2 days

---

### Strategy 3: GPU Acceleration (HIGH EFFORT, HIGHEST POTENTIAL)

#### Option 3A: CUDA/OpenCL Custom Ray-Tracing

**Approach**: Implement custom GPU ray-tracer using CUDA or OpenCL

**Libraries**:
- **CuPy**: NumPy-like API for CUDA
- **PyCUDA**: Direct CUDA access
- **PyOpenCL**: OpenCL (works on AMD/Intel too)
- **Numba CUDA**: JIT compilation to CUDA

**Expected Speedup**: 10-100× (depending on GPU)
- Modern GPU (RTX 3090/4090): ~0.5-2 hours total
- Older GPU: ~2-8 hours

**Pros**:
- ✅ Massive parallelization (thousands of cores)
- ✅ Can process all rays simultaneously
- ✅ Best potential speedup

**Cons**:
- ❌ Requires GPU hardware
- ❌ Complex implementation (500-1000+ lines)
- ❌ GPU memory constraints
- ❌ Requires CUDA/OpenCL knowledge
- ❌ Portability issues (CUDA vs OpenCL)

**Code Changes**: 500-1000+ lines (new module)
**Risk**: High
**Effort**: 1-2 weeks

#### Option 3B: GPU-Accelerated Libraries

**Libraries to Consider**:
1. **PyTorch3D**: GPU-accelerated 3D operations
   - Has ray-mesh intersection functions
   - Requires converting mesh to PyTorch format
   - Good documentation

2. **Kaolin** (NVIDIA): 3D deep learning library
   - GPU ray-casting utilities
   - More specialized

3. **OptiX** (NVIDIA): Professional ray-tracing engine
   - Requires OptiX SDK
   - Most powerful but complex

**Expected Speedup**: 5-50×
- PyTorch3D: ~1-4 hours total

**Pros**:
- ✅ Uses existing optimized libraries
- ✅ Less code than custom CUDA
- ✅ Better maintained than custom code

**Cons**:
- ❌ Requires GPU
- ❌ May need mesh format conversion
- ❌ Learning curve for library APIs

**Code Changes**: 200-400 lines
**Risk**: Medium-High
**Effort**: 3-5 days

---

### Strategy 4: Algorithmic Improvements (MEDIUM EFFORT, MODERATE SPEEDUP)

#### Option 4A: Adaptive Sky Patch Sampling

**Approach**: Use fewer patches in areas with low variation, more in complex areas

**Expected Speedup**: 2-3× (fewer total rays)

#### Option 4B: Hierarchical Ray-Casting

**Approach**: Start with coarse patches, refine only where needed

**Expected Speedup**: 3-5×

#### Option 4C: Rasterization-Based Approximation

**Approach**: Use depth buffer rendering instead of ray-casting
- Render scene from observer point
- Count visible sky pixels
- Much faster but less accurate

**Expected Speedup**: 10-20×
**Accuracy**: Slightly lower (but often acceptable)

**Pros**:
- ✅ Very fast
- ✅ Can use GPU rendering (OpenGL/Vulkan)
- ✅ Good for visualization

**Cons**:
- ❌ Less accurate than ray-casting
- ❌ Requires rendering pipeline

---

### Strategy 5: Hybrid Approach (RECOMMENDED)

**Combination**: CPU Parallelization + Spatial Indexing + Optimized Parameters

**Implementation**:
1. Build spatial index (octree/BVH) of mesh once
2. Use multiprocessing to parallelize points
3. Use spatial index for faster ray intersection
4. Optimize sky patch count (use 200-250 instead of 300)

**Expected Speedup**: 10-20× combined
- **Result**: ~3-6 hours → Could meet 2-hour target with good hardware

**Code Changes**: ~200-300 lines
**Risk**: Low-Medium
**Effort**: 3-5 days

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (1-2 days)
1. **Add multiprocessing** (Strategy 1)
   - Expected: 4-8× speedup
   - Low risk, high reward
   - Can implement today

2. **Optimize parameters**
   - Reduce sky patches to 200-250
   - Use adaptive spacing if needed
   - Expected: 1.2-1.5× speedup

**Combined Phase 1**: ~5-12× speedup → **5-13 hours** (closer to target)

### Phase 2: Spatial Optimization (2-3 days)
3. **Add spatial indexing** (Strategy 2)
   - Use PyVista's built-in locator or trimesh
   - Expected: 2-3× additional speedup

**Combined Phase 1+2**: ~10-36× speedup → **1.8-6.5 hours**

### Phase 3: GPU Acceleration (if Phase 1+2 insufficient) (1-2 weeks)
4. **Implement GPU ray-casting** (Strategy 3B - PyTorch3D recommended)
   - Only if still not meeting target
   - Requires GPU hardware
   - Expected: 5-50× additional speedup

**Combined All**: Could achieve **<1 hour** on good GPU

---

## Detailed Recommendations

### Immediate Action (Today)
✅ **Implement multiprocessing** - This alone will give 4-8× speedup with minimal effort

### Short-term (This Week)
✅ **Add spatial indexing** - Use PyVista's `vtkStaticCellLocator` or similar
✅ **Profile code** - Use `cProfile` to identify exact bottlenecks

### Medium-term (If Needed)
⚠️ **Consider GPU acceleration** - Only if CPU optimizations insufficient
   - Start with PyTorch3D (easiest GPU option)
   - Fallback to custom CUDA if needed

### Long-term (Future)
💡 **Research alternative algorithms** - Rasterization-based methods for real-time applications

---

## Risk Assessment

| Strategy | Risk | Effort | Speedup | Priority |
|----------|------|--------|---------|----------|
| Multiprocessing | Low | Low | 4-8× | **HIGH** |
| Spatial Indexing | Medium | Medium | 2-5× | **HIGH** |
| GPU (PyTorch3D) | Medium | High | 5-50× | Medium |
| GPU (Custom CUDA) | High | Very High | 10-100× | Low |
| Algorithmic | Medium | Medium | 2-5× | Medium |

---

## Code Structure Proposal

```
src/
  svf_utils.py (existing)
  svf_compute.py (new - optimized compute functions)
    - compute_svf_parallel()  # Multiprocessing version
    - compute_svf_spatial()   # With spatial indexing
    - compute_svf_gpu()       # GPU version (optional)
```

---

## Validation Plan

1. **Benchmark current implementation** on small subset (100 points)
2. **Implement Phase 1** (multiprocessing)
3. **Validate** - Run on same subset, compare results
4. **Implement Phase 2** (spatial indexing)
5. **Full validation** - Compare all results with original
6. **Performance testing** - Measure actual speedup

---

## Questions to Answer Before Implementation

1. **Do you have GPU available?** (NVIDIA GPU with CUDA support?)
2. **How many CPU cores?** (Determines multiprocessing benefit)
3. **Accuracy requirements?** (Can we accept slight approximation?)
4. **Budget for development time?** (Quick fix vs. long-term solution)

---

## Next Steps

1. **Profile current code** to confirm bottlenecks
2. **Implement multiprocessing** (Phase 1)
3. **Test and validate** results
4. **Decide on Phase 2** based on results

Would you like me to:
- A) Start with multiprocessing implementation (Phase 1)?
- B) Create a detailed implementation plan for a specific strategy?
- C) Profile the current code first to identify exact bottlenecks?
