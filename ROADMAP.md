# Project Roadmap

## Phase 1: Basic Morphometric Analysis ✅ COMPLETE

### Completed Features
- [x] Building footprint data loading and validation
- [x] Basic metrics calculation (height, area, volume, perimeter, h/w ratio)
- [x] Data filtering pipeline (height, area, volume, h/w ratio, outliers)
- [x] Statistical analysis and summary generation
- [x] Comprehensive visualizations (thematic maps, distributions, scatter plots)
- [x] Flexible input format support (standard and alternative column names)
- [x] Robust error handling and validation

### Current Status
- **Version**: 1.0.0
- **Status**: Production ready
- **Outputs**: Enhanced datasets, statistics, and visualizations

---

## Phase 2: Sky View Factor (SVF) Computation 🎯 NEXT STEP

### Objective
Compute Sky View Factor for outdoor/open space to analyze urban microclimate and thermal comfort in informal settlements.

### Requirements

#### Input Data
- **DTM raster**: Ground elevation in meters (e.g., `vidigal_dtm_cropped.tif`)
- **Building footprints**: Shapefile with attributes:
  - `base_height`: Height above ground (meters)
  - `max_height`: Total building height (meters)

#### Processing Steps

1. **3D Obstruction Surface Generation**
   - Combine DTM elevation with building footprints
   - Extrude buildings from `(DTM + base_height)` to `(DTM + max_height)`
   - Create 3D obstruction model for sky visibility calculations

2. **SVF Computation at Pedestrian Level**
   - Compute SVF at 1.5 m above ground level
   - Exclude building interiors (only outdoor space)
   - Use pyviewfactor library for view factor computation:
     - Convert buildings to 3D pyvista meshes
     - Create discretized sky hemisphere (azimuth × elevation patches)
     - Compute view factors between observer and sky patches
     - Account for building obstructions using visibility checking

3. **Sampling Strategy**
   - Regular grid sampling (1-2 m resolution)
   - Focus on street/open-space areas
   - Mask out building footprints

4. **Output Products**
   - Raster SVF map (values 0-1, where 1 = fully open sky)
   - SVF statistics for public/open space (mean, min, max, std)
   - Optional: Point-based SVF values for specific locations

#### Technical Approach
- **Libraries**: geopandas, rasterio, numpy, shapely, **pyviewfactor**, pyvista
- **Method**: View factor computation using pyviewfactor (double-contour integration)
  - More accurate than ray-casting
  - Analytical computation of view factors between planar polygons
  - Built-in visibility checking
- **Coordinate System**: Projected CRS (meters)
- **Assumptions**:
  - Buildings fully block sky where extruded
  - Trees ignored (focus on built environment)
  - Ground surface follows DTM
- **Migration**: See [docs/SVF_MIGRATION_PLAN.md](docs/SVF_MIGRATION_PLAN.md) for detailed migration plan

#### Implementation Plan

**Step 2.1: Data Preparation**
- [ ] Load and validate DTM raster
- [ ] Load building footprints with height attributes
- [ ] Align coordinate systems
- [ ] Create building extrusion geometry

**Step 2.2: 3D Obstruction Model**
- [ ] Generate 3D building surfaces from footprints
- [ ] Combine with DTM to create complete obstruction surface
- [ ] Validate 3D geometry

**Step 2.3: SVF Algorithm** (Updated: Using pyviewfactor)
- [x] Install and configure pyviewfactor library
- [ ] Convert buildings to pyvista meshes
- [ ] Create discretized sky hemisphere
- [ ] Implement view factor computation using pyviewfactor
- [ ] Handle building obstructions with visibility checking
- [ ] SVF calculation per sample point (sum of view factors)

**Step 2.4: Grid Sampling**
- [ ] Generate sampling grid (1-2 m resolution)
- [ ] Mask building footprints
- [ ] Compute SVF for each grid cell
- [ ] Handle edge cases and boundaries

**Step 2.5: Output Generation**
- [ ] Create SVF raster map
- [ ] Calculate statistics for open space
- [ ] Generate visualizations
- [ ] Export results

#### Expected Outputs
- `svf_map.tif` - Raster SVF map (0-1)
- `svf_statistics.csv` - Summary statistics
- `svf_visualization.png` - Thematic map
- Integration with existing pipeline

---

## Phase 3: Advanced Morphometric Analysis (Future)

### Planned Features
- [ ] Neighborhood-level metrics (BCR, FAR)
- [ ] Spatial autocorrelation analysis
- [ ] Building adjacency analysis
- [ ] Fractal dimension calculation
- [ ] Multi-site comparison framework

---

## Phase 4: Environmental Performance Analysis (Future)

### Planned Features
- [ ] Thermal comfort modeling
- [ ] Wind flow analysis
- [ ] Solar radiation mapping
- [ ] Integration with SVF results
- [ ] Policy recommendations generation

---

## Technical Debt & Improvements

### Code Quality
- [ ] Add unit tests
- [ ] Improve error messages
- [ ] Add progress bars for long operations
- [ ] Optimize performance for large datasets

### Documentation
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Methodology documentation
- [ ] Example datasets

---

## Version History

### v1.0.0 (Current)
- Basic morphometric analysis
- Comprehensive filtering pipeline
- Rich visualizations
- Production-ready codebase

### v2.0.0 (Planned)
- SVF computation
- 3D obstruction modeling
- Outdoor space analysis

---

## Notes

- SVF computation is computationally intensive - consider parallelization
- May need to handle large datasets efficiently
- Consider user-configurable sampling resolution
- Integration with existing filtering pipeline


