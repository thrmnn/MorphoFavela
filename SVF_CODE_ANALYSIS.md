# SVF Computation Code Analysis & Refactoring Plan

## Executive Summary

As a senior developer review, I've identified several critical issues and opportunities for simplification in the SVF computation codebase. The main concerns are:

1. **Critical Alignment Bug**: Current center-based translation is fragile and error-prone
2. **Road Rerouting Always Enabled**: Should be optional and disabled by default
3. **Code Complexity**: Alignment logic is scattered across multiple locations
4. **DTM Underutilization**: DTM should be the primary alignment reference

## Critical Issues

### 1. Alignment Bug (HIGH PRIORITY)

**Current Approach:**
- Uses center-based translation: `dx = stl_center_x - footprints_center_x`
- Fragile: Assumes STL and footprints have similar extents
- No validation that alignment is correct
- DTM bounds not used for alignment even when available

**Problems:**
- If STL and footprints have different extents, centers won't align correctly
- No way to verify alignment quality
- STL is ungeoreferenced but we're guessing its position

**Recommended Fix:**
- Use DTM bounds as the ground truth (georeferenced)
- Align STL to DTM bounds by matching their spatial extents
- Use building footprints and roads (both georeferenced) to validate alignment
- Add alignment validation checks

### 2. Road Rerouting Always Enabled

**Current Behavior:**
- Road rerouting is always executed when buildings are present
- Covered roads are always set to SVF=0
- No way to disable this behavior

**Recommended Fix:**
- Add `--redirect-roads` flag (default: False)
- Add `--set-covered-svf-zero` flag (default: False)
- Make both optional and disabled by default

### 3. Code Organization Issues

**Problems:**
- Alignment logic scattered across main script
- Transformation code duplicated
- Hard to test alignment independently
- No clear separation of concerns

**Recommended Fix:**
- Centralize alignment in `src/data_alignment_utils.py`
- Create `align_stl_to_georeferenced_data()` function
- Use DTM as primary reference when available
- Fallback to building footprints if no DTM

## Proposed Architecture

### New Alignment Strategy

```
1. Load DTM (if available) → Get georeferenced bounds
2. Load Building Footprints → Get georeferenced bounds  
3. Load Roads → Get georeferenced bounds
4. Load STL → Get ungeoreferenced bounds
5. Align STL to DTM bounds (if DTM available)
   OR Align STL to Building Footprints bounds (if no DTM)
6. Validate alignment using all georeferenced datasets
7. Transform all data to STL-local coordinates for computation
8. Transform results back to world coordinates for output
```

### Key Functions to Create

1. `align_stl_to_dtm_bounds(stl_bounds, dtm_bounds) -> transform`
2. `validate_alignment(dtm, buildings, roads, stl, transform) -> bool`
3. `apply_alignment_transform(gdf, transform) -> gdf`

## Simplifications

### 1. Remove Redundant Transformations
- Currently transforming roads, then buildings, then back
- Simplify to: align once, transform once, reverse once

### 2. Centralize Coordinate Handling
- All coordinate transformations in one module
- Clear separation: world coords vs. STL-local coords

### 3. Simplify Road Rerouting Logic
- Make it a clear preprocessing step (optional)
- Only run if explicitly requested

## Implementation Plan

1. **Phase 1**: Create new alignment functions in `data_alignment_utils.py`
2. **Phase 2**: Refactor main script to use new alignment
3. **Phase 3**: Add optional flags for road rerouting
4. **Phase 4**: Add alignment validation
5. **Phase 5**: Update tests

## Testing Strategy

- Test alignment with DTM available
- Test alignment without DTM (fallback to footprints)
- Test alignment validation catches misalignments
- Test road rerouting when enabled/disabled
- Test covered roads behavior when enabled/disabled
