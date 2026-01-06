# SVF Computation Migration Plan: Ray-Casting to pyviewfactor

## Overview

This document outlines the migration from the current ray-casting approach to using the `pyviewfactor` library for Sky View Factor (SVF) computation. The `pyviewfactor` library uses a more accurate double-contour integration method for computing view factors between planar polygons.

## Why pyviewfactor?

### Current Approach (Ray-Casting)
- **Method**: Discrete ray sampling with intersection checks
- **Accuracy**: Limited by sampling resolution (azimuth/elevation steps)
- **Performance**: Can be slow for high-resolution sampling
- **Limitations**: Approximate method, requires fine-tuning of step sizes

### New Approach (pyviewfactor)
- **Method**: Analytical double-contour integration for view factors
- **Accuracy**: More accurate for planar polygon surfaces
- **Performance**: Optimized with numba JIT compilation
- **Advantages**: 
  - More physically accurate
  - Better handling of complex geometries
  - Built-in visibility checking
  - Well-tested library

## Library Information

**pyviewfactor** (v1.0.0)
- **Purpose**: Computes radiation view factors between planar polygons
- **Method**: Double-contour integration (analytical approach)
- **Dependencies**: numpy, scipy, numba, pyvista
- **Documentation**: https://arep-dev.gitlab.io/pyViewFactor/

## Migration Strategy

### Phase 1: Preparation & Setup ✅

- [x] Install `pyviewfactor` library
- [x] Update `requirements.txt`
- [x] Create migration plan document
- [ ] Review pyviewfactor API and examples
- [ ] Understand geometry conversion requirements

### Phase 2: Core Implementation

#### Step 2.1: Geometry Conversion Module
**File**: `src/svf_geometry.py` (new)

**Tasks**:
- [ ] Create function to convert building footprints to 3D pyvista meshes
  - Extract building polygons
  - Get ground elevation from DTM
  - Extrude from `base_height` to `max_height`
  - Convert to pyvista `PolyData` objects
- [ ] Create function to generate sky hemisphere discretization
  - Create hemisphere polygon mesh
  - Discretize into patches (azimuth × elevation grid)
  - Each patch is a planar polygon for view factor computation
- [ ] Create function to convert sampling points to pyvista geometry
  - Represent observer points as small polygons or points
  - Position at pedestrian height (1.5m above ground)

**Key Functions**:
```python
def buildings_to_pyvista_meshes(buildings, dtm_array, dtm_src) -> List[pv.PolyData]
def create_sky_hemisphere(center, radius, azimuth_steps, elevation_steps) -> pv.PolyData
def point_to_observer_surface(point, height, size=0.1) -> pv.PolyData
```

#### Step 2.2: SVF Computation with pyviewfactor
**File**: `src/svf.py` (modify)

**Tasks**:
- [ ] Replace `calculate_svf_ray()` with `calculate_svf_viewfactor()`
- [ ] Implement view factor computation:
  1. Convert buildings to pyvista meshes (once, cached)
  2. Create discretized sky hemisphere
  3. For each sampling point:
     - Create observer surface at pedestrian height
     - For each sky patch:
       - Check visibility using `pyviewfactor.get_visibility()`
       - If visible, compute view factor using `pyviewfactor.compute_viewfactor()`
     - Sum view factors to get total SVF
- [ ] Handle building obstructions:
  - Use pyviewfactor's visibility checking
  - Account for building surfaces blocking sky patches
- [ ] Optimize performance:
  - Cache building meshes
  - Batch process where possible
  - Use spatial indexing for building queries

**Key Function Signature**:
```python
def calculate_svf_viewfactor(
    point: Point,
    point_height: float,
    building_meshes: List[pv.PolyData],
    sky_hemisphere: pv.PolyData,
    pedestrian_height: float = 1.5,
    azimuth_steps: int = 36,
    elevation_steps: int = 9
) -> float:
    """
    Compute SVF using pyviewfactor library.
    
    Returns:
        SVF value (0-1)
    """
```

#### Step 2.3: Integration with Main Pipeline
**File**: `src/svf.py` - `compute_svf_raster()` (modify)

**Tasks**:
- [ ] Update `compute_svf_raster()` to use new approach:
  1. Load and prepare buildings (existing)
  2. Convert buildings to pyvista meshes (new)
  3. Create sky hemisphere (new)
  4. For each grid point, use `calculate_svf_viewfactor()` instead of `calculate_svf_ray()`
- [ ] Maintain same interface (backward compatibility)
- [ ] Update progress monitoring
- [ ] Preserve existing output format

### Phase 3: Testing & Validation

#### Step 3.1: Unit Tests
- [ ] Test geometry conversion functions
- [ ] Test SVF computation for simple cases (known results)
- [ ] Test edge cases (no buildings, single building, etc.)

#### Step 3.2: Comparison Testing
- [ ] Run both old and new methods on same dataset
- [ ] Compare results (should be similar but more accurate)
- [ ] Validate performance improvements
- [ ] Document differences

#### Step 3.3: Integration Testing
- [ ] Test full pipeline with real data
- [ ] Verify output raster format
- [ ] Check visualization compatibility
- [ ] Validate statistics output

### Phase 4: Documentation & Cleanup

#### Step 4.1: Code Documentation
- [ ] Update docstrings with pyviewfactor methodology
- [ ] Add comments explaining view factor approach
- [ ] Document geometry conversion process
- [ ] Update type hints

#### Step 4.2: User Documentation
- [ ] Update README.md with new methodology
- [ ] Update ROADMAP.md to reflect completion
- [ ] Add pyviewfactor to dependencies list
- [ ] Document any parameter changes

#### Step 4.3: Code Cleanup
- [ ] Remove old ray-casting functions (or mark as deprecated)
- [ ] Clean up unused imports
- [ ] Optimize imports
- [ ] Update configuration if needed

## Technical Details

### Geometry Requirements

**Building Meshes**:
- Each building surface (walls, roof) as separate `pv.PolyData`
- Or combined building mesh with all surfaces
- Must be planar polygons (pyviewfactor requirement)

**Sky Hemisphere**:
- Discretized into patches (azimuth × elevation grid)
- Each patch is a planar polygon
- Positioned at appropriate distance from observer
- Radius should be large enough to represent "sky at infinity"

**Observer Surface**:
- Small polygon or point at pedestrian height
- Represents the observer location
- Size should be small relative to building dimensions

### View Factor Computation

The SVF is computed as:
```
SVF = Σ (view_factor(observer, sky_patch_i) for all visible sky_patch_i)
```

Where:
- `view_factor()` is computed using pyviewfactor's analytical method
- Only visible sky patches are included (checked using `get_visibility()`)
- Building obstructions are handled automatically by pyviewfactor

### Performance Considerations

1. **Caching**: Building meshes should be computed once and cached
2. **Spatial Indexing**: Use spatial index to find nearby buildings for each point
3. **Batch Processing**: Consider batching multiple points if pyviewfactor supports it
4. **Resolution**: Balance sky hemisphere discretization (accuracy vs. speed)

## Configuration Updates

### New Parameters (if needed)

```python
# src/config.py additions
SVF_SKY_HEMISPHERE_RADIUS = 1000.0  # m (distance to sky hemisphere)
SVF_OBSERVER_SURFACE_SIZE = 0.1  # m (size of observer surface)
```

### Modified Parameters

- `SVF_AZIMUTH_STEPS`: Now controls sky hemisphere discretization
- `SVF_ELEVATION_STEPS`: Now controls sky hemisphere discretization
- May need adjustment for optimal accuracy/performance balance

## Migration Timeline

1. **Week 1**: Preparation, API review, geometry conversion module
2. **Week 2**: Core SVF computation implementation
3. **Week 3**: Testing, validation, comparison
4. **Week 4**: Documentation, cleanup, final integration

## Risk Mitigation

1. **Backward Compatibility**: Keep old code as fallback initially
2. **Testing**: Extensive testing before removing old code
3. **Performance**: Monitor performance, optimize as needed
4. **Accuracy**: Validate against known test cases

## Success Criteria

- [ ] SVF computation uses pyviewfactor successfully
- [ ] Results are accurate (validated against test cases)
- [ ] Performance is acceptable (comparable or better than ray-casting)
- [ ] Documentation is updated
- [ ] Code is clean and maintainable
- [ ] All tests pass

## References

- pyviewfactor documentation: https://arep-dev.gitlab.io/pyViewFactor/
- pyviewfactor GitHub: (check for examples)
- View factor theory: (add references as needed)

