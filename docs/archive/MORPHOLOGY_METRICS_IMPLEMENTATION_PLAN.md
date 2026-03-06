# Complete Morphology Metrics Implementation Plan

## Overview
This document outlines the complete, autonomous implementation plan for adding 25 morphological indicators from Morphometrics.md to the analysis pipeline. All metrics will be implemented, tested, and validated on RioDasPedras and Vidigal_TLS.

## Branch Information
- **Branch**: `feature/morphology-metrics`
- **Status**: Ready for Implementation
- **Test Areas**: RioDasPedras (informal), Vidigal_TLS (informal)

---

## Metrics to Implement (25 total)

### Category 1: Geometric Properties (3 metrics)
1. ✅ **Building Area (A)** - Already implemented as `area`
2. ✅ **Building Perimeter (P)** - Already implemented as `perimeter`
3. ⚠️ **Longest Axis Length (L_max)** - NEW: `max(L_major, L_minor)`

### Category 2: Compactness and Shape Metrics (13 metrics)
4. ⚠️ **Shape Index (SI)** - NEW: `P / sqrt(A)`
5. ⚠️ **Compactness Weighted Axis (CWA)** - NEW: `(L_major/L_minor) × (4πA/P²)`
6. ⚠️ **Convexity (C)** - NEW: `A / A_convex`
7. ⚠️ **Facade Ratio (FR)** - NEW: `L_facade / P` (requires street data - optional)
8. ⚠️ **Shared Walls (SW)** - NEW: Sum of boundary lengths shared with adjacent buildings
9. ⚠️ **Perimeter Wall (PW)** - NEW: `P - SW`
10. ⚠️ **Number of Corners (NC)** - NEW: Count of polygon vertices
11. ⚠️ **Equivalent Rectangular Index (ERI)** - NEW: `A / A_rect` (minimum bounding rectangle)
12. ⚠️ **Rectangularity (R)** - NEW: `A / (L_major × L_minor)`
13. ⚠️ **Squareness (SQ)** - NEW: `1 - (1/n) × Σ|θ_i - 90°|/90°`
14. ⚠️ **Square Compactness (SC)** - NEW: `A / (max(L_major, L_minor))²`
15. ⚠️ **Elongation (E)** - NEW: `L_major / L_minor`
16. ⚠️ **Compactness-weighted Axis of Each Tessellation (CWT)** - NEW: `L_axis / sqrt(A_tess)`

### Category 3: Spatial Distribution Metrics (4 metrics)
17. ⚠️ **Mean Distance Between Buildings (d_mean)** - NEW: Average centroid distance between all pairs
18. ✅ **Mean Interbuilding Distance (d_NN)** - Already implemented as `inter_building_distance`
19. ⚠️ **Building Adjacency (ADJ)** - NEW: Number of buildings that intersect with each building
20. ⚠️ **Covered Area Ratio (CAR)** - NEW: `ΣA_i / A_site`

### Category 4: Tessellation-related Metrics (3 metrics)
21. ⚠️ **Tessellation Areas (A_v)** - NEW: Voronoi polygon area for each building
22. ⚠️ **Cell Alignment (CA)** - NEW: Consistency of cell orientation
23. ⚠️ **Tessellation Number of Neighbors (TN)** - NEW: Count of buildings sharing edge in tessellation

### Category 5: Topological Features (2 metrics)
24. ⚠️ **Fractal Dimension (FD)** - NEW: `2 × ln(P) / ln(A)`
25. ⚠️ **Average Weighted Distance (d_w)** - NEW: Weighted centroid distances

**Total New Metrics to Implement**: 22 (3 already exist)

---

## Implementation Structure

### File Organization
```
src/
├── morphology_metrics.py          # NEW: All 25 morphology metrics
├── metrics.py                     # EXISTING: Basic metrics (extend)
└── visualize.py                   # EXISTING: Visualization (extend)

scripts/
├── calculate_morphology_metrics.py # NEW: Main script for morphology metrics
└── calculate_metrics.py           # EXISTING: Basic metrics script

tests/
└── test_morphology_metrics.py      # NEW: Unit tests
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)
**Goal**: Set up the structure and helper functions

1. **Create `src/morphology_metrics.py`**
   - Helper functions for geometric calculations:
     - `get_major_minor_axes(geometry)` - Calculate L_major, L_minor
     - `get_convex_hull_area(geometry)` - Calculate A_convex
     - `get_minimum_bounding_rectangle(geometry)` - Calculate A_rect
     - `get_interior_angles(geometry)` - Calculate angles for squareness
     - `get_shared_boundary_length(geom1, geom2)` - Calculate shared wall length
   - Spatial indexing utilities for efficient neighbor queries

2. **Create `scripts/calculate_morphology_metrics.py`**
   - Main script structure
   - Argument parsing
   - Data loading and validation
   - Integration with existing pipeline

### Phase 2: Geometric Properties (1 metric)
**Goal**: Implement L_max (Longest Axis Length)

3. **Function**: `calculate_longest_axis_length(gdf)`
   - Calculate L_major and L_minor for each building
   - Return max(L_major, L_minor)
   - Add as column `longest_axis_length`

### Phase 3: Compactness and Shape Metrics (13 metrics)
**Goal**: Implement all shape-related metrics

4. **Shape Index (SI)**: `P / sqrt(A)`
5. **Compactness Weighted Axis (CWA)**: Requires L_major, L_minor, A, P
6. **Convexity (C)**: `A / A_convex`
7. **Shared Walls (SW)**: Spatial intersection with neighbors
8. **Perimeter Wall (PW)**: `P - SW`
9. **Number of Corners (NC)**: Count vertices
10. **Equivalent Rectangular Index (ERI)**: `A / A_rect`
11. **Rectangularity (R)**: `A / (L_major × L_minor)`
12. **Squareness (SQ)**: Calculate interior angles
13. **Square Compactness (SC)**: `A / (max(L_major, L_minor))²`
14. **Elongation (E)**: `L_major / L_minor`
15. **CWT**: Requires tessellation (defer to Phase 5)

**Note**: Facade Ratio (FR) requires street data - mark as optional, implement if street data available

### Phase 4: Spatial Distribution Metrics (3 metrics)
**Goal**: Implement spatial arrangement metrics

16. **Mean Distance Between Buildings (d_mean)**: All-pairs centroid distance
17. **Building Adjacency (ADJ)**: Count intersecting neighbors
18. **Covered Area Ratio (CAR)**: Requires site boundary (use convex hull of all buildings)

### Phase 5: Tessellation Metrics (3 metrics)
**Goal**: Implement Voronoi tessellation-based metrics

19. **Tessellation Areas (A_v)**: Generate Voronoi diagram
20. **Cell Alignment (CA)**: Calculate orientation consistency
21. **Tessellation Number of Neighbors (TN)**: Count shared edges
22. **CWT**: Complete from Phase 3

**Dependencies**: `scipy.spatial.Voronoi` or `shapely.voronoi_diagram`

### Phase 6: Topological Features (2 metrics)
**Goal**: Implement advanced topological metrics

23. **Fractal Dimension (FD)**: `2 × ln(P) / ln(A)`
24. **Average Weighted Distance (d_w)**: Weighted by adjacency or size

### Phase 7: Integration and Testing
**Goal**: Integrate all metrics into pipeline

25. **Update `scripts/calculate_metrics.py`**
   - Add option to compute morphology metrics
   - Maintain backward compatibility

26. **Create comprehensive test suite**
   - Unit tests for each metric function
   - Synthetic data tests (square, rectangle, circle)
   - Edge case tests (single building, two buildings, etc.)

27. **Run on RioDasPedras**
   - Compute all metrics
   - Validate ranges
   - Check for NaN/inf values
   - Generate visualizations

28. **Run on Vidigal_TLS**
   - Compute all metrics
   - Validate ranges
   - Compare distributions with RioDasPedras

### Phase 8: Visualization
**Goal**: Create comprehensive visualizations

29. **Thematic Maps**
   - Building-level maps for each metric
   - Color schemes appropriate for each metric type
   - Legends and scale bars

30. **Statistical Visualizations**
   - Histograms for each metric
   - Box plots (RioDasPedras vs Vidigal_TLS)
   - Correlation matrices
   - Scatter plots (metric relationships)

31. **Multi-panel Summaries**
   - Grouped by category
   - Side-by-side comparisons

### Phase 9: Documentation and Validation
**Goal**: Document and validate results

32. **Update documentation**
   - README.md with new metrics
   - claude.md with metric descriptions
   - Usage examples

33. **Generate validation report**
   - Summary statistics for both areas
   - Distribution comparisons
   - Performance metrics (computation time, memory)

---

## Technical Implementation Details

### Dependencies
- **Existing**: geopandas, shapely, numpy, matplotlib
- **New**: scipy (for Voronoi tessellation)

### Data Requirements
- **Buildings**: Required (both areas have this)
- **Streets**: Optional (only for Facade Ratio)
- **Site Boundary**: Optional (use convex hull of all buildings for CAR)

### Computational Considerations
- **Spatial Indexing**: Use STRtree for efficient neighbor queries
- **Tessellation**: Use scipy.spatial.Voronoi for large datasets
- **Shared Walls**: Use spatial intersection with buffered geometries
- **Performance**: Optimize for datasets with 1000+ buildings

### Output Format
- **GeoPackage**: All metrics as columns in building footprints
- **CSV**: Summary statistics
- **Visualizations**: PNG files (300 DPI)

---

## Validation Strategy

### Unit Tests
- Square building (10m × 10m): Known values for all metrics
- Rectangle building (20m × 10m): Test elongation, rectangularity
- Circle building: Test compactness metrics
- Two adjacent buildings: Test shared walls, adjacency

### Real Data Validation
- **RioDasPedras**: ~X buildings
  - Check all metrics computed
  - Validate ranges (no negative values, reasonable bounds)
  - Check for NaN/inf
  - Spatial consistency checks

- **Vidigal_TLS**: ~X buildings
  - Same validation as RioDasPedras
  - Compare distributions

### Comparative Validation
- Compare metric distributions between areas
- Check expected differences (e.g., density, compactness)
- Validate metric relationships (e.g., SI vs CWA)

---

## Success Criteria

### Functional
- [ ] All 25 metrics implemented
- [ ] All metrics computed for RioDasPedras
- [ ] All metrics computed for Vidigal_TLS
- [ ] No computational errors
- [ ] All values within expected ranges

### Quality
- [ ] Unit tests pass
- [ ] Real data validation passes
- [ ] Visualizations generated
- [ ] Documentation complete

### Performance
- [ ] Computation time < 1 hour per area
- [ ] Memory usage reasonable
- [ ] Scalable to larger datasets

---

## Execution Plan

### Autonomous Execution Steps
1. Create `src/morphology_metrics.py` with all helper functions
2. Implement all 22 new metrics (grouped by category)
3. Create `scripts/calculate_morphology_metrics.py`
4. Create unit tests
5. Run on RioDasPedras
6. Run on Vidigal_TLS
7. Generate visualizations
8. Update documentation
9. Commit all changes

### Expected Timeline
- Phase 1-6 (Implementation): ~2-3 hours
- Phase 7 (Testing): ~1 hour
- Phase 8 (Visualization): ~1 hour
- Phase 9 (Documentation): ~30 minutes
- **Total**: ~4-5 hours

---

## Notes

- Facade Ratio (FR) will be implemented only if street data is available
- CWT requires tessellation, so it's grouped with tessellation metrics
- Some metrics may have edge cases (e.g., single building, zero area) - handle gracefully
- Use logging for progress tracking
- Follow existing code style and patterns
