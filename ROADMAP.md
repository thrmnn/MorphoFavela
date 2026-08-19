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
- **Version**: 6.0.0
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
      — **repaired 2026-06-02** (cell clipping, commit e2252f4): pre-June grids inflated
      λf 2-3×; all grids overwritten in place, pre-June λf values are superseded
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
- [x] **Dissolved λf canonicalised + downstream re-baseline** (2026-06-25) — party-wall-corrected
      (dissolved) λf is now the single authoritative source, pinned bit-for-bit in
      `outputs/brisa_ventilation_fix/lambda_f_canonical.json` and test-locked
      (`tests/test_lambda_f_lockfile.py`; pooled built mask n = 64,389). Oke / Grimmond-Oke
      flow-regime split ≈ 65 % skimming / 30 % wake / 5 % isolated. Regime-stratified four-state
      taxonomy supersedes the interim λf>2.75 cut; the old "typology inversion" is retired
      (hillside 42.2 % > flatland 37.3 % compound, H/W cross-check agrees). See
      `docs/brisa_round4_results_2026-06-25.md`.
- [x] **k=6 cell morphotypes + k=3 block morphotopes** re-fit on dissolved λf
      (`scripts/build_signature.py`, `src/morphometry/signature.py`).
- [x] **Typology→WHO-2 h-sun predictor — conclusion FLIPPED** (2026-06) — under the re-baseline
      the continuous fabric vector (LOSO AUC-PR ≈ 0.84) beats the discrete type (≈ 0.61);
      spatially-blocked block-bootstrap 95 % CIs overlap (so the flip is a direction, not a clean
      gap) with low VIF (1.1–3.0). Blind risk map externally validated on 3 calibration favelas
      (pooled AUC-PR 0.76). `scripts/analyze_typology_predictor.py` + `typology_predictor_extra.py`.
- [x] **New geometric scalars** (2026-06) — lateral-connectivity (distance-to-open-edge,
      `run_lateral_connectivity.py`); 2-D ventilation-susceptibility (regime × depth,
      `run_ventilation_susceptibility.py`, pooled 41.8 % doubly constrained); effective
      wind-exposure (directional λf × wind rose, `run_wind_exposure.py`).
- [x] **Track C (terrain-following morphometry) resolved as a scoping null** (2026-06-25,
      commit 9923a1c) — σH already terrain-following (corr(slope,σH) negative); λf terrain-step
      gate cleared (<5 %); canonical λf untouched.
- [~] **Roughness z0/zd via UMEP** — invalid for **53–75 % of cells** (zd>H_mean, λp>0.5 out of
      Kanda envelope); the validity envelope *is* the result. CFD drag-centroid anchor gated on
      real OpenFOAM. `src/morphometry/roughness.py`, `scripts/build_roughness.py`.
- [x] **Morpho-signature track (WS-0 → WS-B) + review hub** (2026-06-19, branch
      `track/morpho-signature`) — two-table feature substrate respecting the street/grid
      change-of-support (`src/morphometry/features.py`); favela morphotype clustering
      (GMM k=6, cross-site recurrence validated, fabric×experience contingency,
      `signature.py`); spatial mode-filter + bootstrap stability (ARI 0.90); a
      geometry-only prioritization index (`prioritization.py`). Expert-reviewed figure set
      (`figures_v2/`) + a Tailscale-served **project hub** (`hubkit.py` +
      `build_project_hub.py`) with rendered docs, named per-favela Sites, and the TR HTML
      view. Plans + append-only decision log under `docs/morpho_signature_*`. Reusable
      **`project-hub` skill** distilled. Awaiting user figure review before merge to main.
- [~] **Roughness-estimation track** (branch `track/roughness`, 2026-06-19) — morphometric
      z0(θ)/zd(θ) per cell + per wind sector via vendored UMEP (Kanda primary). **Done:**
      R-A per-cell z0/zd + method-spread + extrapolation flags (`src/morphometry/roughness.py`);
      R-B spatial z0 map + directional rose + cross-method figure; R-D-prep per-patch z0(θ)
      for the CFD inlet (`patch_roughness.csv`, 119 patches); TR §6.5; CFD handoff contract
      (`src/cfd_integration/README.md`); brisaverse `roughness_canonical.md` facts. Findings:
      zd>H_mean 70–93%, λp>0.5 56–88% (out of envelope, flagged), methods diverge ~20×.
      **Gated:** R-C CFD drag-centroid anchor (needs real OpenFOAM). Plan/decisions:
      `docs/roughness_plan.md`, `docs/roughness_decisions.md`.
- [x] **5-site street-SVF re-baseline + branch consolidation** (2026-06-18) — all five
      sites re-baselined on the corrected ray-caster (boundary-clip street sampling +
      facade-normal fix for CW polygons); removes external-highway edge bias. Validated
      against Mingze's independent Ladybug ground truth (Vidigal MAE 1.77 → 1.70 h,
      Pearson 0.565 → 0.589). Outputs now emit `run_meta.json` provenance (git sha) so
      silent drift is detectable. The 41-commit BRISA chain (`feat/brisa-paper` →
      `fix/brisa-figures-fig5-fig6`) fast-forwarded onto `main`; 13 merged remote
      branches pruned. Suite green (539 tests), ruff clean.
- [x] **Grid λf repair** (2026-06-02) — facade segments now clipped to the cell; per-site
      medians dropped 2-3× (canonical band λp>0: p50 = 1.15–2.69). Grids overwritten in
      place; anything quoting pre-June λf is superseded. See `docs/brisa_ventilation_handoff.md`.
- [x] **Interim λf diagnostic taxonomy** (2026-06-03) — four-state at pooled p75 = 2.75;
      values renamed failure → constraint; Figs 03/04 regenerated from corrected λf.
- [x] **3 calibration sites onboarded** (2026-06-03) — Borel, Jacarezinho, Morro do
      Juramento (calibration-only, lighter outputs layout; not CFD campaign sites).
- [x] **Canonical street-observer networks** for all 5 sites + length-weighted segment
      SVF (audit C1 fix, d8175ca) (2026-06-05).
- [x] **Site dashboards** (2026-06-05) — interactive HTML (Leaflet + Plotly) and A3
      *Folha de Rua* sheets for all 5 sites under `outputs/_distribution/`.
- [x] **Mingze cross-validation** (2026-06-11) — 4-site 3D bundle shipped; Vidigal
      Ladybug-vs-raycast comparison: MAE 1.77 h, bias −0.10 h, WHO <2 h deprivation-flag
      agreement 72 %.
- [x] **Two-pass repo audit** landed (2026-06-03) — `docs/audit_2026-06-03/`.
- [x] **5-site UMEP cross-validation** closed (2026-05-01) — slopes within ±8 % of unity
      at 4/5 sites, r² 0.68–0.94, all biases ≥ 0 (height-shifted MorphoFavela input zeros sub-1.5 m
      structures). Vendored UMEP at `vendor/umep_processing/`. Closes §10.3 limitation.
- [x] **§6.5 Blocken margin claim corrected** (2026-05-02) — min margin 114 m at RDP-P15
      (not "≥150 m" as previously stated); 11/119 patches under 150 m, all at RdP/Rocinha;
      `blocken_ok = true` for all 119.
- [x] **Result-side analysis pipeline** shipped (2026-04-29) —
      `scripts/analyze_cfd_results.py` + `generate_synthetic_cfd_results.py`; end-to-end
      synthetic validation on all 5 sites (covered cells 359–405).
- [x] **Six project subagents + CFD adapter** (2026-04-29) — `cfd-results-ingestor`
      auto-detects MorphoFavela native (cardinal + CSV) and Airflow native (`wind_NNN/` +
      parquet) layouts.
- [x] **Wind input for all 5 sites** (2026-04-27) — measured roses 2015–2024,
      n = 64,088–89,439 hourly records, `quality_flag: "measured"`.
- [x] **Pilot patch in Airflow** (2026-04-26) — VDG-P07 placed, preflight 7/7.
- [x] Tregenza 145 equal-area patches (PR #6)
- [x] Makefile build automation (PR #5)
- [x] CFD patch selection framework — tiling, 31 features, clustering, export (PR #7)
      — superseded by Phase 5 stratified sampler in April 2026; clustering-based
      `src/patch_selection/` removed.
- [x] Mare onboarding — SVF + morphology + patch selection complete
- [x] Complexo do Alemao re-extraction (21,729 buildings)

### High Priority

#### HPC SVF Computation ✅ COMPLETE (April 2026)
- [x] Full SVF for Complexo do Alemão
- [x] Full SVF for Vidigal (full extent, not just TLS subset)
- [x] Full SVF for Rocinha, Rio das Pedras, Maré

#### Cross-Area Clustering — DEPRECATED
- ~~Cross-area patch clustering across all 5 CFD campaign areas~~
- ~~Unified typology labels for OpenFOAM domain selection~~
- *Replaced by 12-strata stratified sampling (Phase 5, 2026-04-09); no longer needed.*

#### OpenFOAM Handoff (in `~/SCL/SCR/Airflow`)
- [x] Per-patch exports ready (buildings.gpkg, terrain.tif, patch_meta.json) for all 119
- [x] Airflow build_patch_case / preflight / write_summary updated for circular patches
- [x] VDG-P07 preflight 7/7 in `~/SCL/SCR/Airflow/cases/VDG-P07/`
- [ ] **Submit VDG-P07 mesh + 8-direction wind campaign to MIT ORCD** (user-driven)
- [ ] Validate end-to-end pipeline against real CFD return for VDG-P07
- [ ] Submit remaining 118 patches incrementally

#### Validation
- [x] **SVF benchmarked against UMEP** (5-site, 2026-05-01) — see Recently Completed
- [ ] Cross-validate solar irradiance against measured data or pvlib baselines
      (deferred — not on critical path for CFD campaign)

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

## Phase 5: CFD Sampling Campaign ✅ COMPLETE (2026-04-09)

### Completed
- [x] 10m morphometric grid computation across 5 sites (82,314 cells total)
- [x] Extended building + terrain context from city-wide RJ data (300m buffer)
- [x] Stratified pilot sampling (12-strata: SVF × slope × λp)
- [x] Pilot batch: 69 patches across 5 sites
- [x] Full-campaign incremental allocation with SVF-priority weighting
- [x] Campaign total: 119 patches (Vidigal 22, Rocinha 25, RdP 22, CDA 25, Maré 25)
- [x] Per-patch exports: buildings.gpkg, terrain.tif, patch_meta.json
- [x] Cross-site summary with pooled strata table
- [x] Rocinha topology audit (10 invalid geometries, all repaired by buffer(0))

### Recommended CFD pipeline test patch
**MAR-P07** (Maré): slope 0.8°, H_max 12.2m, 1,200 buildings in domain,
mid-range SVF/λp. Simplest geometry for end-to-end pipeline validation.

---

## Phase 6: CFD Simulations (in other repo at ~/SCL/SCR/Airflow)

### Wind input ✅ COMPLETE (2026-04-27)
- [x] Iowa State ASOS METAR ingestion for SBGL Galeão (Maré)
- [x] INMET BDMEP yearly ZIP downloader (`scripts/download_inmet_zips.py`) +
      per-station extractor (`scripts/extract_inmet_stations.py`)
- [x] Measured wind roses for all 5 sites, full 2015–2024 window,
      n = 64,088–89,439 hourly records per station, `quality_flag: "measured"`
- [x] Figure S5 in technical report §2.3

### Pilot patch in Airflow ✅ READY
- [x] Vidigal patch VDG-P07 placed in `~/SCL/SCR/Airflow/cases/VDG-P07/`
      (buildings.gpkg + terrain.tif + patch_meta.json), preflight 7/7
- [x] Airflow build_patch_case.py + preflight.py + write_summary.py
      updated to read circular `analysis_patch_diameter` (was hardcoded
      to old square `analysis_patch_size`)

### Next steps (in ~/SCL/SCR/Airflow)
- [ ] Submit VDG-P07 mesh + 8-direction wind campaign to MIT ORCD
- [ ] Validate end-to-end pipeline on VDG-P07 results
- [ ] Submit remaining 118 patches incrementally
- [ ] Post-processing: wind velocity at pedestrian height, aggregate to grid cells

### Next steps (in this repo)
- [x] **Result-side pipeline implemented + synthetic-validated on all 5 sites**
      (2026-04-29). Entry points: `mf-analyze-cfd`, `mf-synthetic-cfd`. Outputs in
      `outputs/{site}/cfd_analysis/`.
- [ ] Ingest first real CFD return (VDG-P07) via `cfd-results-ingestor` agent →
      `src/cfd_integration/` aggregation
- [ ] Annual-weighted maps using measured wind roses (`src/cfd_integration/weighting.py`
      already in place)
- [ ] Graduate `src/cfd_integration/io.py` to versioned-stable once VDG-P07 ingestion
      validates the on-disk contract end-to-end

---

## Phase 7: Analysis + Publication (after CFD)

### Nature Cities submission
- [x] Figures 1-5 main + S1/S2/S4 supplementary (S3 awaits 20m grids)
- [x] Site colors and style system (`outputs/paper_figures/fig_style.py`)
- [x] Cross-site feature space, allocation summary, morphometric maps
- [ ] Integrate CFD-derived wind velocity into figures (risk maps, threshold analysis)
- [ ] Statistical analysis: SVF/λp/terrain as predictors of wind conditions
- [ ] Draft manuscript

### Follow-up analyses (candidates)
- [ ] 20m morphometric grids for resolution sensitivity (Fig S3)
- [ ] Facade solar for remaining 4 sites (only vidigal_tls has results)
- [ ] Typology clustering validated against CFD results
- [ ] Cross-site regression: morphology → wind conditions

---

## Future Work (Deferred)

#### Advanced Morphometric Analysis
- [ ] Neighborhood-level metrics beyond the 12 in the grid
- [ ] Spatial autocorrelation analysis
- [ ] Building adjacency analysis
- [ ] Fractal dimension calculation

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

### v5.0.0 (April 2026)
- CFD sampling pipeline: `run_pilot_sampling.py`, `run_campaign_sampling.py`, `build_extended_context.py`
- 12-strata stratified sampling (SVF × slope × λp) with maximin spacing
- Extended building + DTM context from city-wide RJ data (300m buffer, reduces domain exclusion from 55% to ~3%)
- 119 patches allocated across 5 sites (69 pilot + 50 campaign)
- SVF-priority weighting for health-relevant low-SVF strata (2×)
- Per-patch exports ready for OpenFOAM: buildings.gpkg, terrain.tif, patch_meta.json
- Cross-site comparative outputs in `outputs/comparative/`
- Canonical output layout: `morphometrics/`, `sampling_cfd/`, `comparative/`
- Nature Cities paper figures: 9 scripts + shared style module (`outputs/paper_figures/`)
- **Status**: Phase 5 complete, ready for OpenFOAM simulations in CFD repo

### v6.0.0 (June 2026)
- Grid λf repair: cell clipping in `src/urban_morphology.py` / `src/morphometry/indicators.py`;
  all per-site grids overwritten in place (medians 2-3× lower)
- BRISA ventilation workstream: interim λf taxonomy (pooled p75 = 2.75), failure → constraint
  rename, Figs 03/04/05 regenerated; 3 calibration sites onboarded
- Street level: canonical observer networks (`build_street_observers.py`), length-weighted
  segment SVF (audit C1)
- Distribution: HTML + A3 site dashboards, Mingze 3D bundle + Vidigal Ladybug
  cross-validation (`scripts/compare_mingze_vidigal.py`)
- 5-site street-SVF re-baseline on the corrected ray-caster (boundary-clip + facade-normal),
  validated vs Mingze Ladybug (Vidigal MAE 1.77 → 1.70 h); `run_meta.json` provenance manifests
- Dissolved (party-wall corrected) λf canonicalised + test-locked (`lambda_f_canonical.json`,
  pooled n = 64,389; ≈ 65/30/5 skimming/wake/isolated); regime-stratified taxonomy replaces the
  interim λf>2.75 cut; k=6 morphotypes + k=3 morphotopes re-fit; predictor FLIP (continuous vector
  >> discrete type) with spatial-CV CIs; new lateral-connectivity / ventilation-susceptibility /
  wind-exposure scalars; Track C resolved as a null; roughness z0 out-of-envelope 53–75 %
- **Status**: BRISA chain consolidated onto `main` (2026-06-18); dissolved-λf re-baseline +
  geometric-scalar tracks landed (2026-06-25); standing plan in
  `docs/council_roadmap_2026-06-25.md` + `docs/repo_parallel_plan_2026-06-26.md`

### v5.5.0 (late April – early May 2026)
- Wind input for all 5 sites: INMET BDMEP + Iowa State ASOS pipelines, measured roses
  2015–2024, `quality_flag: "measured"`
- Six project subagents under `.claude/agents/` (3 validators + 3 workflow accelerators)
- `src/cfd_integration/io.py` auto-detects MorphoFavela native and Airflow native on-disk layouts
- Result-side pipeline: `analyze_cfd_results.py` + `generate_synthetic_cfd_results.py`,
  end-to-end synthetic-validated on all 5 sites
- 5-site UMEP cross-validation closed (§10.3 limitation): vendored UMEP at
  `vendor/umep_processing/`, slopes within ±8 % at 4/5 sites
- Pre-commit hook (`.claude/hooks/check_report_sync.py`) keeps `technical_report.md` ↔
  `.pdf` ↔ `figures/` in sync; blocking on .md ↔ .pdf pairing, advisory elsewhere
- Technical report polished: §6.5 Blocken claim corrected, §10.3 closed, full §4 cross-val
  table, figS6–figS10 supplementary scatters
- **Status**: Pipeline ready for real CFD returns; awaiting VDG-P07 from MIT ORCD

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
- SVF validated against UMEP `svfForProcessing153` on all 5 campaign sites (2026-05-01)
- All Phase 2, 3, 3.5, 4, 4.5, and 5 analyses are complete and production-ready
  (λf re-baselined 2026-06-02 by the cell-clipping repair — see Recently Completed)
- CFD campaign: 5 areas — vidigal, rocinha, riodaspedras, complexo_do_alemao, maré
  (CDD excluded from the campaign at allocation time; its building data was later
  confirmed valid — see `project_cdd_data_bug` memory — but the 5-site lock stood)
- CFD simulation execution is handled in a separate repo (`~/SCL/SCR/Airflow`); this repo
  produces the patches (`sampling_cfd/campaign_sampling/patches/`) and ingests
  results via `cfd-results-ingestor` + `src/cfd_integration/` for Phase 7 analysis
- **Current focus**: VDG-P07 on MIT ORCD (user-driven), then incremental submission
  of remaining 118 patches; ingestion + annual weighting on this side is plumbed and
  awaiting real data
