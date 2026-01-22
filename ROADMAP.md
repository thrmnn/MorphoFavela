# Project Roadmap

## Phase 1: Basic Morphometric Analysis ✅ COMPLETE

### Completed Features
- [x] Building footprint data loading and validation
- [x] Basic metrics calculation (height, area, volume, perimeter, h/w ratio, inter-building distance)
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

## Phase 2: Sky View Factor (SVF) Computation ✅ COMPLETE

### Completed Features
- [x] STL-based 3D scene loading and terrain extraction
- [x] Ground-level grid generation with building footprint masking
- [x] Hemispherical sky patch discretization
- [x] Ray-casting SVF computation using pyviewfactor-style visibility
- [x] Progress monitoring during computation
- [x] SVF visualization (heatmap and histogram)
- [x] Shared utilities module for code reuse

### Current Status
- **Input**: Single STL file containing terrain + buildings
- **Method**: Discretized hemispherical dome with ray-casting
- **Output**: SVF raster (0-1), CSV, and visualizations
- **Ground Masking**: Excludes building interiors using footprint shapefile
- **Street-Level Analysis**: Additional street centerline-based SVF computation (see Phase 2.1)

---

## Phase 2.1: Street-Level SVF Computation ✅ COMPLETE

### Completed Features
- [x] Point sampling along street centerlines
- [x] DTM and mesh-based elevation extraction
- [x] Street-level SVF computation using existing infrastructure
- [x] Segment-level aggregation and statistics
- [x] Street-colored visualization maps
- [x] Statistical distribution plots

### Current Status
- **Script**: `scripts/compute_svf_streets.py`
- **Input**: STL mesh, road network shapefile (LineString), optional DTM raster
- **Method**: Sample points along streets, compute SVF at pedestrian eye level (1.5m)
- **Output**: Point-level and segment-level GeoPackages, statistics CSV, visualizations
- **Complementary**: Works alongside grid-based SVF (does not replace it)

### Use Cases
- Pedestrian-level environmental assessment
- Street hierarchy comparison
- Identification of problematic street segments with low sky access

---

## Phase 2.5: Solar Access Computation ✅ COMPLETE

### Completed Features
- [x] Solar position calculation using pvlib (winter solstice)
- [x] Ray-casting solar access computation
- [x] Hours of direct sunlight calculation
- [x] Threshold-based classification (deficit vs. acceptable)
- [x] Progress monitoring during computation
- [x] Solar access visualizations (heatmap and threshold map)
- [x] Code reuse with SVF utilities

### Current Status
- **Input**: Same STL file and building footprints as SVF
- **Method**: Ray-casting toward sun positions for winter solstice
- **Output**: Solar access heatmap and threshold classification map
- **Ground Masking**: Same logic as SVF for consistency

---

## Phase 2.6: Sky Exposure Plane Exceedance Analysis ✅ COMPLETE

### Completed Features
- [x] Unified analysis script (`analyze_sky_exposure_streets.py`) combining building-level and street-level exceedance
- [x] Building-level exceedance computation (percentage of volume exceeding envelope per building)
- [x] Street-level exceedance computation (point sampling along street centerlines)
- [x] Rio de Janeiro ruleset implementation (1/5 ratio, variable setbacks)
- [x] São Paulo ruleset implementation (1/10 ratio, 10m threshold)
- [x] Building mesh extraction from STL using footprints
- [x] Sky exposure plane envelope calculation with ruleset-specific parameters
- [x] Volumetric exceedance computation per building
- [x] Street point-level and segment-level exceedance aggregation
- [x] Exceedance map visualization (building-level and street-level)
- [x] Vertical section views showing actual vs. allowed building heights
- [x] Summary statistics and CSV export
- [x] Support for analysis with or without road network (building-level always computed)

### Current Status
- **Script**: `scripts/analyze_sky_exposure_streets.py`
- **Input**: STL mesh, building footprints, optional road network shapefile
- **Method**: 
  - **Building-level**: Ruleset-based envelope calculation per building, percentage exceedance
  - **Street-level**: Point sampling along streets, pedestrian perspective (1.5m height), ruleset-based envelope calculation
- **Rulesets**: 
  - **Rio de Janeiro**: Base height from first ventilated floor, setback = max(2.5m, H/5), envelope = base + (distance × 5)
  - **São Paulo**: 10m threshold, setback = max(3.0m, (H-6)/10) for H > 10m, envelope = 10 + (distance × 10)
- **Output**: 
  - Building exceedance map (percentage per building)
  - Street exceedance maps (points, segments, colored by exceedance)
  - Section views (high, mean, low exceedance points)
  - Statistics (building and street-level)
- **Purpose**: Building code compliance evaluation and environmental performance assessment
- **Note**: Legacy `analyze_sky_exposure.py` (45° fixed envelope) is deprecated

---

## Phase 2.7: Sectional Porosity Computation ✅ COMPLETE

### Completed Features
- [x] Load and prepare building footprints (reprojection, buffering)
- [x] Generate regular 2D grid covering analysis domain
- [x] Compute sectional porosity per grid cell using vectorized operations
- [x] Generate top-down porosity heatmap visualization
- [x] Provide summary statistics (mean, 10th percentile)

### Current Status
- **Input**: Building footprints shapefile
- **Method**: Geometric calculation of void area fraction within horizontal slice
- **Output**: Porosity raster (.npy), CSV, and heatmap visualization
- **Conceptual Framing**: Geometric proxy for wind access, no airflow simulation

---

## Phase 2.8: Occupancy Density Proxy ✅ COMPLETE

### Completed Features
- [x] Compute built volume per building from STL and footprints
- [x] Automatically generate analysis units as regular grid
- [x] Aggregate built volume per analysis unit
- [x] Compute open space area per analysis unit
- [x] Compute density proxy (built volume / open space area) safely handling zero open space
- [x] Generate choropleth density map and summary statistics

### Current Status
- **Input**: STL mesh and building footprints
- **Method**: Aggregation of built volume and open space area at defined spatial unit
- **Output**: GeoDataFrame, density map, CSV summary
- **Conceptual Framing**: Proxy for occupancy pressure, relative ranking

---

## Phase 2.9: Morphological Environmental Deprivation Index (Unit-level) ✅ COMPLETE

### Completed Features
- [x] Aggregate solar access, SVF, and sectional porosity to analysis units
- [x] Compute normalized deficit scores for solar, ventilation, and occupancy pressure
- [x] Compute composite hotspot index using equal weighting
- [x] Classify analysis units into "Extreme hotspot", "High deprivation", and "Baseline"
- [x] Generate hotspot map, deficit overlap map, and ranking table

### Current Status
- **Input**: Analysis units, solar/SVF/porosity rasters, occupancy density GeoDataFrame
- **Method**: Composite index combining multiple environmental performance proxies
- **Output**: GeoDataFrame, hotspot map, deficit overlap map, ranking table
- **Conceptual Framing**: Highlights zones of compounded environmental deprivation, non-causal

---

## Phase 2.9.5: Morphological Environmental Deprivation Index (Raster-based) ✅ COMPLETE

### Completed Features
- [x] Continuous 2D raster computation at pixel level
- [x] Pixel-level occupancy pressure computation (built volume / open space per cell)
- [x] Coordinate-based resampling for different raster resolutions
- [x] Building mask application (excludes building interiors)
- [x] Continuous heatmap visualization
- [x] Classified hotspot map with thresholds (Extreme, High, Baseline)
- [x] Unit-level aggregation for policy interpretation

### Current Status
- **Input**: Solar/SVF/porosity rasters, STL mesh, building footprints, optional analysis units
- **Method**: Continuous raster with pixel-level deficit computation and composite index
- **Output**: Raster (.npy), continuous heatmap, classified map, unit-level aggregation
- **Advantages**: Higher spatial resolution, continuous gradients, precise hotspot identification

---

## Phase 3: Multi-Area Comparative Analysis 🚧 IN PROGRESS

### Phase 3.0: Data Organization Structure ✅ COMPLETE

#### Completed Features
- [x] Area-based data directory structure (`data/{area}/raw/`)
- [x] Area-based output directory structure (`outputs/{area}/`)
- [x] Configuration helpers for area paths (`get_area_data_dir()`, `get_area_output_dir()`)
- [x] Supported areas: `vidigal` (informal), `copacabana` (formal)
- [x] Documentation for data organization and file naming conventions

#### Current Status
- **Structure**: Complete - area-based data organization implemented
- **Areas**: Vidigal and Copacabana data organized and analyzed
- **Status**: Ready for comparative analysis (see Phase 3.1)

### Phase 3.1: Comparative Analysis Framework ✅ COMPLETE

#### Completed Features
- [x] Area-based script support (`--area` parameter in `calculate_metrics.py`)
- [x] Comparative analysis script (`compare_areas.py`) comparing metrics across areas
- [x] Side-by-side visualization framework for formal vs informal comparisons
- [x] Statistical comparison of morphometric metrics (mean, distributions, Mann-Whitney U tests)
- [x] Comparative environmental performance analysis (SVF, solar access, porosity, deprivation index)
- [x] Automated PDF report generation with clean Swiss design aesthetic
- [x] Area normalization for fair spatial comparisons
- [x] Aspect ratio preservation in side-by-side visualizations

#### Current Status
- **Script**: `scripts/compare_areas.py`
- **Output**: Comprehensive PDF report with statistics, visualizations, and findings
- **Features**: 
  - Morphometric metrics comparison with statistical tests
  - Environmental performance comparison (SVF, solar, porosity, deprivation)
  - Area-normalized statistics accounting for different study area sizes
  - Professional PDF report with clean design
- **Results**: `outputs/comparative/comparison_report.pdf`

#### Comparisons Included
- **Morphometric metrics**: Height, area, volume, perimeter, H/W ratio, inter-building distance
- **Environmental performance**: SVF, solar access, porosity distributions and statistics
- **Deprivation analysis**: Hotspot identification and spatial patterns
- **Statistical rigor**: Mann-Whitney U tests, significance indicators, effect sizes

### Phase 3.2: Advanced Morphometric Analysis (Future)

#### Planned Features
- [ ] Neighborhood-level metrics (BCR, FAR)
- [ ] Spatial autocorrelation analysis
- [ ] Building adjacency analysis
- [ ] Fractal dimension calculation
- [ ] Urban form typology classification

---

## Phase 4: Urban Morphology Metrics 🆕 PLANNED

### Overview
Add comprehensive urban morphology metrics for environmental analysis, including plan area density, frontal area density, height variability, street orientation entropy, and morphological typology clustering.

### Planned Features
- [ ] Plan area density (λp) - footprint area / total area per analysis unit
- [ ] Frontal area density (λf) - building frontal area perpendicular to wind
- [ ] Height variability (σh) - standard deviation of building heights per unit
- [ ] Street orientation entropy (H) - Shannon entropy of street directions
- [ ] Zone flagging (SVF < 0.3, λf > 0.4)
- [ ] Morphological typology clustering (K-means/hierarchical)
- [ ] Integration with comparative analysis framework

### New Files to Create
- `src/urban_morphology.py` - Core module with metric computation functions
- `scripts/compute_urban_morphology.py` - Main script for running the analysis
- `scripts/compute_typology_clustering.py` - Clustering on morphology features

### Implementation Phases
1. **Core Functions**: Plan density, height variability
2. **Complex Metrics**: Frontal density, entropy
3. **Main Script**: Visualizations, integration
4. **Zone Flagging**: Threshold-based flags
5. **Typology Clustering**: K-means/hierarchical analysis
6. **Comparative Analysis**: Multi-area comparison

### Status
- **Planning**: Complete - see `URBAN_MORPHOLOGY_PLAN.md` for detailed design
- **Implementation**: Not started

---

## Phase 5: Environmental Performance Analysis (Future)

### Planned Features
- [ ] Thermal comfort modeling
- [ ] Wind flow analysis
- [ ] Solar radiation mapping
- [ ] Integration with SVF results
- [ ] Policy recommendations generation

---

## Next Steps & Priorities

### High Priority - Code Quality & Robustness

#### Performance Optimization
- [ ] **Parallelize SVF/solar access computation** - Ray-casting operations can be parallelized across grid points
- [ ] **Optimize pixel-level occupancy pressure** - Vectorize building volume computation where possible
- [ ] **Cache intermediate results** - Building volumes, ground masks, and resampled rasters
- [ ] **Profile and benchmark** - Identify bottlenecks in raster-based deprivation index computation
- **Priority**: High - Current computation can be slow for large datasets

#### Error Handling & Validation
- [ ] **Validate raster alignment** - Check CRS consistency and spatial bounds before processing
- [ ] **Handle edge cases** - Empty rasters, mismatched bounds, missing data gracefully
- [ ] **Improve error messages** - Provide actionable guidance when errors occur
- [ ] **Input validation** - Comprehensive checks for all input files and parameters
- **Priority**: High - Prevents runtime failures and improves user experience

#### Testing
- [ ] **Unit tests** - Core functions (metrics calculation, deficit computation, aggregation)
- [ ] **Integration tests** - Full pipeline end-to-end tests
- [ ] **Test data fixtures** - Small representative datasets for testing
- [ ] **Regression tests** - Ensure outputs remain consistent across code changes
- **Priority**: Medium-High - Ensures reliability and maintainability

### Medium Priority - Functionality Enhancements

#### Configuration Management
- [ ] **Centralize parameters** - All thresholds, percentiles, and classification parameters in one place
- [ ] **Configuration file support** - YAML/JSON config files for easy parameter adjustment
- [ ] **Command-line argument validation** - Better validation and defaults
- **Priority**: Medium - Improves usability and reproducibility

#### Output Enhancements
- [ ] **GeoTIFF export** - Export rasters with CRS metadata for GIS integration
- [ ] **Interactive visualizations** - Plotly/Folium maps for exploration
- [ ] **Summary report generation** - PDF/HTML reports with key findings
- [ ] **Standardized output formats** - Consistent naming and structure across all scripts
- **Priority**: Medium - Improves usability and integration with other tools

#### Documentation
- [ ] **Methodology documentation** - Detailed explanation of how each metric is computed
- [ ] **Tutorial notebooks** - Step-by-step examples for each analysis
- [ ] **API documentation** - Sphinx-generated documentation for all functions
- [ ] **Example datasets** - Sample data for users to test the pipeline
- **Priority**: Medium - Improves accessibility for new users

### Lower Priority - Advanced Features

#### Advanced Morphometric Analysis (Phase 3)
- [ ] Neighborhood-level metrics (BCR, FAR)
- [ ] Spatial autocorrelation analysis
- [ ] Building adjacency analysis
- [ ] Fractal dimension calculation
- [ ] Multi-site comparison framework
- **Priority**: Low - Future research direction

#### Environmental Performance Modeling (Phase 4)
- [ ] Thermal comfort modeling
- [ ] Wind flow analysis
- [ ] Solar radiation mapping
- [ ] Integration with SVF results
- [ ] Policy recommendations generation
- **Priority**: Low - Requires domain expertise and additional data

---

## Technical Debt & Improvements

### Code Quality
- [x] Add progress bars for long operations (tqdm implemented)
- [ ] Add unit tests
- [ ] Improve error messages
- [ ] Optimize performance for large datasets

### Documentation
- [x] Basic README and ROADMAP documentation
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Methodology documentation
- [ ] Example datasets

---

## Version History

### v1.0.0
- Basic morphometric analysis
- Comprehensive filtering pipeline
- Rich visualizations
- Production-ready codebase

### v2.0.0 (January 2025)
- SVF computation (STL-based)
- Solar access computation
- Sky exposure plane exceedance analysis
- Sectional porosity computation
- Occupancy density proxy
- Morphological environmental deprivation index (unit-level and raster-based)
- Ground-level analysis with building masking
- Shared utilities for code reuse
- **Status**: All Phase 2 analyses complete and production-ready

### v3.0.0 (Current - January 2025)
- Multi-area data organization structure
- Area-based filtering policy (formal vs informal)
- Comparative analysis framework (`compare_areas.py`)
- Automated PDF report generation with clean Swiss design
- Statistical comparison tools (Mann-Whitney U tests)
- Area-normalized spatial comparisons
- Aspect ratio preservation in visualizations
- **Status**: Phase 3 complete - ready for formal vs informal settlement research

### v3.1.0 (Planned)
- Performance optimizations (parallelization, caching)
- Enhanced error handling and validation
- GeoTIFF export with CRS metadata
- Configuration file support
- Basic unit tests

### v2.2.0 (Planned)
- Interactive visualizations
- Summary report generation
- Methodology documentation
- Tutorial notebooks

---

## Immediate Action Items (Current Focus)

### Phase 3: Multi-Area Setup (This Week)
1. **Migrate Vidigal data** - Move existing data from `data/raw/` to `data/vidigal/raw/`
2. **Add Copacabana data** - Place STL mesh and building footprints in `data/copacabana/raw/`
3. **Verify data structure** - Ensure all files are properly organized following naming conventions
4. **Test area-based paths** - Verify `get_area_data_dir()` and `get_area_output_dir()` work correctly

### Next Phase: Comparative Analysis (Next 1-2 Weeks)
5. **Update scripts for area parameter** - Add `--area` flag to support area-specific processing
6. **Create comparative analysis script** - Framework for side-by-side comparisons
7. **Generate comparison visualizations** - Compare formal vs informal morphometric patterns

### Future Improvements (After Phase 3.1)
8. **Performance profiling** - Identify bottlenecks in raster-based deprivation index computation
9. **Add GeoTIFF export** - Include CRS metadata for GIS integration
10. **Configuration file** - Create `config.yaml` for all analysis parameters
11. **Basic unit tests** - Start with metrics calculation and deficit computation functions

## Notes

- SVF computation is computationally intensive - parallelization is a high priority
- Raster-based deprivation index works with native resolutions but may need optimization for very large datasets
- All Phase 2 analyses are complete and production-ready
- Focus should shift to optimization, robustness, and usability improvements


