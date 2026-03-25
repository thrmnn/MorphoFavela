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
  - **Rio de Janeiro**: Base height from first ventilated floor, setback = max(2.5m, H/5), envelope = base + (distance x 5)
  - **São Paulo**: 10m threshold, setback = max(3.0m, (H-6)/10) for H > 10m, envelope = 10 + (distance x 10)
- **Output**:
  - Building exceedance map (percentage per building)
  - Street exceedance maps (points, segments, colored by exceedance)
  - Section views (high, mean, low exceedance points)
  - Statistics (building and street-level)
- **Purpose**: Building code compliance evaluation and environmental performance assessment
- **Note**: Legacy `analyze_sky_exposure.py` (45 deg fixed envelope) is deprecated

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

## Phase 3: Multi-Area Comparative Analysis ✅ COMPLETE

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

---

## Phase 3.5: SVF v2 Engine ✅ COMPLETE

### Completed Features
- [x] Modular SVF engine (`src/svf_v2/`) with world-coordinate computation
- [x] 3D scene construction from DTM and building footprints (`src/svf_v2/scene.py`)
- [x] Joblib-based parallel raycasting (`src/svf_v2/compute.py`, `n_jobs` parameter)
- [x] Checkpoint/resume support for long-running computations
- [x] Scene optimization for faster mesh operations
- [x] Facade sampling module (`src/svf_v2/facades.py`)
- [x] I/O utilities and path management (`src/svf_v2/io.py`, `src/svf_v2/paths.py`)
- [x] SVF visualization module (`src/svf_v2/visualize.py`)
- [x] Legacy SVF code removed from active use (archived)
- [x] HPC deployment scripts (`scripts/hpc/`)

### Current Status
- **Engine**: `src/svf_v2/` - complete rewrite with modular architecture
- **Parallelization**: joblib with configurable `n_jobs` (default 1, -1 for all cores)
- **Checkpointing**: Auto-save every N points, resume from last checkpoint on restart
- **HPC**: Batch scripts for SLURM-based clusters (`scripts/hpc/run_svf_batch.sh`, `run_full_pipeline.sh`)
- **Script**: `scripts/run_svf_v2.py`

---

## Phase 4: Urban Morphology Metrics ✅ COMPLETE

### Completed Features
- [x] Plan area density (lambda_p) - footprint area / total area per analysis unit
- [x] Frontal area density (lambda_f) - building frontal area perpendicular to wind
- [x] Height variability (sigma_h) - standard deviation of building heights per unit
- [x] Street orientation entropy (H) - Shannon entropy of street directions
- [x] Zone flagging (SVF < 0.3, lambda_f > 0.4)
- [x] Morphological typology clustering (K-means/hierarchical) (`src/typology.py`)
- [x] Spatial analysis utilities (`src/spatial_analysis.py`)
- [x] Morphology visualization module (`src/visualize_morphology.py`)
- [x] Cartography module for publication-quality maps (`src/cartography.py`)

### Modules
- `src/urban_morphology.py` - Core metric computation functions
- `src/typology.py` - Clustering and typology classification
- `src/spatial_analysis.py` - Spatial statistics and analysis utilities
- `src/visualize_morphology.py` - Morphology-specific visualizations
- `src/cartography.py` - Publication-quality cartographic output
- `scripts/compute_urban_morphology.py` - Main analysis script
- `scripts/classify_typology.py` - Typology clustering script

### Tests
- `tests/test_urban_morphology.py`
- `tests/test_typology.py`
- `tests/test_visualize_morphology.py`
- `tests/test_spatial_analysis.py`
- `tests/test_cartography.py`

---

## Phase 4.5: Solar Access Package ✅ COMPLETE

### Completed Features
- [x] Full solar irradiance models (`src/solar/irradiance.py`)
- [x] Sun position calculation (`src/solar/sun.py`)
- [x] Seasonal analysis across solstices and equinoxes (`src/solar/seasonal.py`)
- [x] Ray-casting solar computation (`src/solar/compute.py`)
- [x] Solar I/O utilities (`src/solar/io.py`)
- [x] Publication-quality solar visualizations (`src/solar/visualize.py`)
- [x] Exposure module for sky exposure analysis (`src/exposure.py`)

### Modules
- `src/solar/` - Complete solar access package
  - `compute.py` - Ray-casting solar access computation
  - `irradiance.py` - Direct/diffuse irradiance models
  - `sun.py` - Sun position and path calculations
  - `seasonal.py` - Multi-season analysis (solstices, equinoxes)
  - `visualize.py` - Solar visualization (heatmaps, sun path diagrams)
  - `io.py` - Solar data I/O
- `src/exposure.py` - Sky exposure plane analysis

### Tests
- `tests/test_solar/` - Comprehensive test suite
  - `test_compute.py`, `test_irradiance.py`, `test_seasonal.py`, `test_sun.py`, `test_visualize.py`
- `tests/test_exposure.py`

---

## Phase 5: Environmental Performance Analysis (Future)

### Planned Features
- [ ] Thermal comfort modeling
- [ ] Wind flow analysis
- [ ] Integration with morphology and solar results
- [ ] Policy recommendations generation

---

## Next Steps & Priorities

### Recently Completed
- [x] Tregenza 145 equal-area patches (PR #6)
- [x] Makefile build automation (PR #5)
- [x] CFD patch selection framework — tiling, 31 features, clustering, export (PR #7)
- [x] Mare onboarding — SVF + morphology + patch selection complete
- [x] Complexo do Alemao re-extraction (21,729 buildings)

### High Priority

#### HPC SVF Computation
- [ ] Run full SVF for Complexo do Alemao (21,729 buildings)
- [ ] Run full SVF for Vidigal (full extent, not just TLS subset)

#### Cross-Area Clustering
- [ ] Cross-area patch clustering across all 5 CFD campaign areas
- [ ] Unified typology labels for OpenFOAM domain selection

#### OpenFOAM Handoff
- [ ] Test OpenFOAM mesh generation from exported patch geometries
- [ ] Validate boundary condition setup for selected patches

#### Validation
- [ ] Benchmark SVF results against RayMan, SOLWEIG, or other reference tools
- [ ] Cross-validate solar irradiance against measured data or pvlib baselines

### Medium Priority

#### CI Expansion
- [ ] Extend CI beyond svf_v2 tests to cover morphology, solar, and exposure modules
- [ ] Add integration tests for full pipeline

#### Configuration Management
- [ ] Centralize parameters in YAML/JSON config files
- [ ] Command-line argument validation improvements

#### Output Enhancements
- [ ] GeoTIFF export with CRS metadata for GIS integration
- [ ] Interactive visualizations (Plotly/Folium)

### Lower Priority

#### Advanced Morphometric Analysis (Phase 3.2)
- [ ] Neighborhood-level metrics (BCR, FAR)
- [ ] Spatial autocorrelation analysis
- [ ] Building adjacency analysis
- [ ] Fractal dimension calculation

#### Environmental Performance Modeling (Phase 5)
- [ ] Thermal comfort modeling
- [ ] Wind flow analysis (CFD campaign in progress)

#### Documentation
- [ ] API documentation (Sphinx)
- [ ] Tutorial notebooks
- [ ] Methodology documentation

---

## Technical Debt & Improvements

### Code Quality
- [x] Add progress bars for long operations (tqdm implemented)
- [x] Parallelize SVF computation (joblib)
- [x] Checkpoint/resume for long computations
- [x] Scene optimization for mesh operations
- [x] Remove legacy SVF code
- [ ] Expand unit test coverage across all modules
- [ ] Improve error messages

### Documentation
- [x] Basic README and ROADMAP documentation
- [ ] API documentation
- [ ] Tutorial notebooks
- [ ] Methodology documentation

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

### v3.0.0 (January 2025)
- Multi-area data organization structure
- Area-based filtering policy (formal vs informal)
- Comparative analysis framework (`compare_areas.py`)
- Automated PDF report generation with clean Swiss design
- Statistical comparison tools (Mann-Whitney U tests)
- Area-normalized spatial comparisons
- Aspect ratio preservation in visualizations
- **Status**: Phase 3 complete

### v3.5.0 (February 2025)
- SVF v2 engine with modular architecture (`src/svf_v2/`)
- Joblib parallel raycasting with configurable worker count
- Checkpoint/resume for long-running SVF computations
- Optimized scene construction from DTM + footprints
- HPC deployment scripts for SLURM clusters (`scripts/hpc/`)
- SVF visualization module
- Legacy SVF code archived
- **Status**: SVF v2 engine production-ready

### v4.0.0 (March 2025)
- Urban morphology metrics: plan area density, frontal area density, height variability, street orientation entropy
- Zone flagging for environmental risk areas
- Morphological typology clustering (K-means/hierarchical)
- Spatial analysis utilities
- Cartography module for publication-quality maps
- Solar access package with irradiance models, seasonal analysis, sun position
- Exposure module for sky exposure plane analysis
- Publication-quality solar visualizations
- Comprehensive test suites for morphology, typology, solar, spatial analysis, cartography, and exposure
- **Status**: Phase 4 + 4.5 complete

---

## Notes

- Tregenza 145-patch SVF is merged and production-ready
- Validation against benchmark tools is needed before publication
- All Phase 2, 3, 3.5, 4, and 4.5 analyses are complete and production-ready
- CFD campaign: 5 areas (vidigal_tls, rocinha, riodaspedras, complexo_do_alemao, mare); CDD excluded due to broken data
- Current focus: HPC SVF for remaining areas, cross-area clustering, OpenFOAM handoff
