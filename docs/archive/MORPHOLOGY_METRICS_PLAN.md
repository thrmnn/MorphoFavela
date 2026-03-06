# Morphology Metrics Development Plan

## Branch Information
- **Branch**: `feature/morphology-metrics`
- **Purpose**: Add multiple morphology metrics to the analysis pipeline
- **Test Areas**: RioDasPedras (informal), Vidigal_TLS (informal)
- **Status**: Planning Phase

---

## Overview

This document outlines the plan for adding new morphology metrics to the analysis pipeline. The goal is to expand beyond the current 6 basic metrics (height, area, volume, perimeter, h/w ratio, inter-building distance) and environmental metrics (SVF, solar access, porosity) to include additional urban morphology indicators.

---

## Current Metrics Inventory

### Basic Morphometric Metrics (Phase 1) ✅
1. **height** - Building height (m)
2. **area** - Footprint area (m²)
3. **volume** - Building volume (m³)
4. **perimeter** - Footprint perimeter (m)
5. **hw_ratio** - Street canyon ratio (height/width)
6. **inter_building_distance** - Distance to nearest neighbor (m)

### Environmental Performance Metrics (Phase 2) ✅
1. **Sky View Factor (SVF)** - Ground-level and street-level
2. **Solar Access** - Hours of direct sunlight
3. **Sectional Porosity** - Vertical permeability
4. **Occupancy Density Proxy** - Building density indicator
5. **Sky Exposure Plane Exceedance** - Building code compliance proxy

### Deprivation Indices (Phase 3) ✅
1. **Morphological Environmental Deprivation Index** - Unit-level and raster-based

---

## New Morphology Metrics to Add

### Category 1: Building Form Metrics
*[Add metrics related to building shape, compactness, and form complexity]*

#### Metric 1.1: [Metric Name]
- **Description**: [Brief description of what this metric measures]
- **Formula/Calculation**: [Mathematical definition or algorithm]
- **Units**: [Units of measurement]
- **Range**: [Expected value range]
- **Interpretation**: [What high/low values mean]
- **References**: [Academic papers or standards]
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

**Example Placeholders:**
- Compactness ratio (4π×area/perimeter²)
- Form factor (area/area of bounding box)
- Shape index (perimeter/(2×√(π×area)))
- Fractal dimension
- Building elongation ratio

---

#### Metric 1.2: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

---

### Category 2: Spatial Configuration Metrics
*[Add metrics related to spatial arrangement, clustering, and urban fabric]*

#### Metric 2.1: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

**Example Placeholders:**
- Building density (buildings per hectare)
- Floor area ratio (FAR)
- Coverage ratio (building footprint area / total area)
- Open space ratio
- Clustering coefficient
- Nearest neighbor index (R-statistic)
- Spatial autocorrelation (Moran's I)
- Block size distribution
- Street network density

---

#### Metric 2.2: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

---

### Category 3: Street-Level Morphology Metrics
*[Add metrics that require street network data]*

#### Metric 3.1: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

**Example Placeholders:**
- Street width (from building footprints)
- Street canyon aspect ratio (H/W)
- Street orientation distribution
- Street connectivity (betweenness, closeness)
- Street network integration
- Visibility graph analysis metrics
- Street-level enclosure
- Pedestrian network accessibility

---

#### Metric 3.2: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

---

### Category 4: Height Distribution Metrics
*[Add metrics related to vertical morphology and height patterns]*

#### Metric 4.1: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

**Example Placeholders:**
- Height coefficient of variation
- Height entropy
- Height gradient (spatial variation)
- Vertical density profile
- Height-to-street-width ratio distribution
- Building height diversity index

---

#### Metric 4.2: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

---

### Category 5: Connectivity and Accessibility Metrics
*[Add metrics related to spatial connectivity and movement]*

#### Metric 5.1: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

**Example Placeholders:**
- Building-to-street connectivity
- Pedestrian accessibility index
- Spatial integration (space syntax)
- Angular integration
- Choice (space syntax)
- Connectivity index
- Permeability index

---

#### Metric 5.2: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

---

### Category 6: Aggregated/Composite Metrics
*[Add metrics that combine multiple indicators]*

#### Metric 6.1: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

**Example Placeholders:**
- Urban complexity index
- Morphological diversity index
- Compactness index
- Mixed-use index (if land use data available)
- Urban intensity index

---

#### Metric 6.2: [Metric Name]
- **Description**: 
- **Formula/Calculation**: 
- **Units**: 
- **Range**: 
- **Interpretation**: 
- **References**: 
- **Status**: [ ] Not started | [ ] In progress | [ ] Implemented | [ ] Tested

---

## Implementation Plan

### Phase 1: Metric Selection and Prioritization
- [ ] Review literature and select metrics most relevant to informal settlements
- [ ] Prioritize metrics based on:
  - Data availability (buildings, streets, STL mesh)
  - Computational feasibility
  - Research relevance
  - Comparative value (RioDasPedras vs Vidigal_TLS)
- [ ] Document selected metrics in detail above

### Phase 2: Code Structure Design
- [ ] Create new module: `src/morphology_metrics.py`
- [ ] Design function signatures for each metric category
- [ ] Plan data dependencies (buildings, streets, mesh, etc.)
- [ ] Design output format (GeoDataFrame columns, raster, etc.)

### Phase 3: Implementation
- [ ] Implement Category 1 metrics (Building Form)
- [ ] Implement Category 2 metrics (Spatial Configuration)
- [ ] Implement Category 3 metrics (Street-Level) - requires street data
- [ ] Implement Category 4 metrics (Height Distribution)
- [ ] Implement Category 5 metrics (Connectivity)
- [ ] Implement Category 6 metrics (Composite)

### Phase 4: Testing and Validation

#### 4.1 Unit Tests
- [ ] Create test file: `tests/test_morphology_metrics.py`
- [ ] Test each metric function with synthetic data
- [ ] Test edge cases (empty geometries, single building, etc.)
- [ ] Test with known values (e.g., square building = compactness = 1.0)

#### 4.2 Validation Tests (RioDasPedras)
- [ ] Run all metrics on RioDasPedras data
- [ ] Validate output ranges (check for NaN, inf, negative values)
- [ ] Compare with manual calculations for sample buildings
- [ ] Check spatial consistency (adjacent buildings should have similar values)
- [ ] Validate against known characteristics of RioDasPedras

**RioDasPedras Validation Checklist:**
- [ ] Building count matches expected (~X buildings)
- [ ] Metric distributions are reasonable
- [ ] No computational errors or warnings
- [ ] Output files are correctly formatted
- [ ] Memory usage is acceptable

#### 4.3 Validation Tests (Vidigal_TLS)
- [ ] Run all metrics on Vidigal_TLS data
- [ ] Validate output ranges
- [ ] Compare with manual calculations
- [ ] Check spatial consistency
- [ ] Validate against known characteristics of Vidigal_TLS

**Vidigal_TLS Validation Checklist:**
- [ ] Building count matches expected (~X buildings)
- [ ] Metric distributions are reasonable
- [ ] No computational errors or warnings
- [ ] Output files are correctly formatted
- [ ] Memory usage is acceptable

#### 4.4 Comparative Validation
- [ ] Compare metric distributions between RioDasPedras and Vidigal_TLS
- [ ] Verify expected differences (e.g., density, compactness)
- [ ] Check for consistency in metric relationships
- [ ] Validate that metrics capture morphological differences

### Phase 5: Visualization

#### 5.1 Thematic Maps
- [ ] Create thematic map function for each metric category
- [ ] Design color schemes appropriate for each metric
- [ ] Add legends and scale bars
- [ ] Create side-by-side comparisons (RioDasPedras vs Vidigal_TLS)

**Visualization Requirements:**
- [ ] Building-level maps (colored by metric value)
- [ ] Raster maps (for aggregated metrics)
- [ ] Street-level maps (for street-related metrics)
- [ ] Statistical distribution plots
- [ ] Scatter plots (metric relationships)
- [ ] Comparative visualizations (RioDasPedras vs Vidigal_TLS)

#### 5.2 Statistical Visualizations
- [ ] Histograms for each metric
- [ ] Box plots (RioDasPedras vs Vidigal_TLS)
- [ ] Scatter plots (metric correlations)
- [ ] Summary statistics tables

#### 5.3 Debug Visualizations
- [ ] Intermediate calculation visualizations
- [ ] Spatial indexing visualizations (if applicable)
- [ ] Validation plots (e.g., manual vs computed)

### Phase 6: Integration
- [ ] Integrate new metrics into `scripts/calculate_metrics.py`
- [ ] Add command-line arguments for metric selection
- [ ] Update output format (GeoPackage with new columns)
- [ ] Update summary statistics generation
- [ ] Update comparative analysis script (`scripts/compare_areas.py`)

### Phase 7: Documentation
- [ ] Update `README.md` with new metrics
- [ ] Add metric descriptions to `claude.md`
- [ ] Create usage examples
- [ ] Document metric interpretation guidelines
- [ ] Add references to academic literature

---

## Test Data Requirements

### RioDasPedras
- **Buildings**: `data/riodaspedras/raw/riodaspedras_buildings.shp` ✅
- **Streets**: `data/riodaspedras/raw/roads_riodaspedras.shp` ✅
- **STL Mesh**: `data/riodaspedras/raw/full_scan.stl` ✅
- **DTM**: `data/riodaspedras/raw/riodaspedras_dtm.tif` ✅

### Vidigal_TLS
- **Buildings**: `data/vidigal_tls/raw/vidigal_buildings.shp` ✅
- **Streets**: `data/vidigal_tls/raw/roads_vidigal.shp` ✅
- **STL Mesh**: `data/vidigal_tls/raw/full_scan.stl` ✅
- **DTM**: `data/vidigal_tls/raw/vidigal_dtm_cropped.tif` ✅

---

## Validation Strategy

### 1. Synthetic Data Tests
Create simple test cases with known expected values:
- Single square building (10m × 10m × 5m)
- Two adjacent buildings
- Regular grid of buildings
- Irregular cluster of buildings

### 2. Real Data Validation
- **Sanity Checks**: Values within expected ranges
- **Spatial Consistency**: Adjacent buildings have similar values (where appropriate)
- **Distribution Checks**: Histograms show reasonable distributions
- **Cross-Validation**: Compare with manual calculations for sample buildings

### 3. Comparative Validation
- **RioDasPedras vs Vidigal_TLS**: Expected differences should be captured
- **Metric Correlations**: Check relationships between metrics
- **Literature Comparison**: Compare with published values for similar settlements

### 4. Performance Validation
- **Computation Time**: Should complete in reasonable time (< 1 hour for full dataset)
- **Memory Usage**: Should not exceed available RAM
- **Scalability**: Should work for datasets of different sizes

---

## Visualization Strategy

### Building-Level Metrics
- **Format**: Thematic maps with buildings colored by metric value
- **Output**: PNG files (300 DPI)
- **Color Schemes**: 
  - Sequential colormaps for continuous metrics (viridis, plasma)
  - Diverging colormaps for metrics with meaningful center (RdBu, RdYlBu)
  - Categorical colormaps for discrete metrics

### Street-Level Metrics
- **Format**: Street centerlines colored by metric value
- **Output**: PNG files with building footprints as context
- **Integration**: Overlay on building footprint maps

### Aggregated Metrics
- **Format**: Raster maps or choropleth maps
- **Output**: PNG files with appropriate legends
- **Resolution**: Match grid spacing used in SVF/solar access analyses

### Statistical Visualizations
- **Format**: Multi-panel figures (histograms, box plots, scatter plots)
- **Output**: PNG files
- **Layout**: Side-by-side comparisons (RioDasPedras vs Vidigal_TLS)

---

## Output Structure

### File Organization
```
outputs/
├── riodaspedras/
│   └── morphology_metrics/
│       ├── buildings_with_morphology.gpkg  # All metrics as columns
│       ├── morphology_summary_stats.csv
│       ├── morphology_maps/
│       │   ├── [metric_name]_map.png
│       │   └── ...
│       └── morphology_statistics/
│           ├── distributions.png
│           ├── correlations.png
│           └── summary_stats.csv
└── vidigal_tls/
    └── morphology_metrics/
        └── [same structure]
```

### Data Format
- **GeoPackage**: Building footprints with metric columns
- **CSV**: Summary statistics and aggregated values
- **Raster** (if applicable): Gridded metric values (NumPy arrays)

---

## Success Criteria

### Functional Requirements
- [ ] All selected metrics implemented and tested
- [ ] Metrics computed correctly for RioDasPedras and Vidigal_TLS
- [ ] Output files generated in correct format
- [ ] Visualizations created for all metrics
- [ ] Comparative analysis possible between areas

### Quality Requirements
- [ ] No computational errors or warnings
- [ ] All values within expected ranges
- [ ] Spatial consistency validated
- [ ] Performance acceptable (< 1 hour per area)
- [ ] Documentation complete

### Research Requirements
- [ ] Metrics capture morphological differences between areas
- [ ] Metrics align with literature on informal settlements
- [ ] Results interpretable and meaningful
- [ ] Ready for comparative analysis and publication

---

## Next Steps

1. **Populate this document** with specific metrics you want to implement
2. **Review and prioritize** metrics based on research goals
3. **Design implementation** approach for each metric category
4. **Create test cases** for validation
5. **Implement metrics** one category at a time
6. **Test and validate** on both RioDasPedras and Vidigal_TLS
7. **Create visualizations** for all metrics
8. **Integrate** into existing pipeline
9. **Document** usage and interpretation

---

## Notes

- This is a living document - update as metrics are implemented
- Mark status of each metric as you progress
- Add implementation notes and challenges encountered
- Document any deviations from the plan
- Keep track of computational performance for each metric

---

## References

*[Add relevant academic papers, standards, or methodologies here]*

Example references to consider:
- Space syntax methodology
- Urban morphology literature
- Informal settlement morphology studies
- Building form metrics (compactness, shape indices)
- Spatial configuration metrics (density, connectivity)
