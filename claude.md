# Favela Morphometric Analysis - AI Context

## Project Overview
Python pipeline for calculating morphometric metrics from building footprints with height attributes. Designed for analyzing informal settlement geometry.

## Key Metrics
- **height**: Building height (m)
- **area**: Footprint area (m²)
- **volume**: Building volume (m³)
- **perimeter**: Footprint perimeter (m)
- **hw_ratio**: Street canyon ratio (height/width)
- **inter_building_distance**: Distance to nearest neighbor building (m)

## Data Input
- Formats: `.gpkg`, `.geojson`, `.shp`
- Columns: `base_height`/`top_height` OR `base`/`altura` (auto-normalized)
- CRS: Projected (UTM preferred)

## Filtering Pipeline
**Note**: Filtering is only applied to informal settlements (e.g., Vidigal). Formal settlements (e.g., Copacabana) skip filtering.

For informal areas:
1. Height filter (max 20m)
2. Metrics calculation
3. Area filter (max 500m²)
4. Volume filter (max 3000m³)
5. H/W ratio filter (max 100)
6. Height/area ratio outlier filter (99th percentile)

## Configuration
All parameters in `src/config.py`:
- Filtering thresholds
- Visualization settings
- Data paths

## Code Structure
- `src/metrics.py`: Metrics calculation & validation
- `src/visualize.py`: Visualization functions
- `src/config.py`: Configuration
- `scripts/calculate_metrics.py`: Main pipeline

## Notes
- Uses logging, not print statements
- Type hints and docstrings throughout
- Data validation before processing
- Percentile-based outlier filtering for h/w ratio visualization

## Phase 2: SVF and Solar Access Computation ✅ COMPLETE
- **SVF**: STL-based computation using discretized hemispherical dome
- **Solar Access**: Winter solstice direct sunlight hours using pvlib
- **Shared Utilities**: `src/svf_utils.py` contains common functions
- **Scripts**: `compute_svf.py` and `compute_solar_access.py` (separated)
- **Ground Masking**: Both scripts exclude building interiors using footprint shapefile
- **Progress Monitoring**: Both scripts use tqdm for real-time progress tracking

## Phase 2.6: Sky Exposure Plane Exceedance Analysis ✅ COMPLETE
- **Script**: `analyze_sky_exposure.py`
- **Method**: Environmental performance envelope with base height (7.5m), setbacks (front: 5m, side/rear: 3m), sky plane angle (45°)
- **Outputs**: Exceedance map, vertical sections, CSV with metrics
- **Purpose**: Evaluate environmental implications (solar access, ventilation), NOT code compliance

## Phase 2.7: Sectional Porosity Computation ✅ COMPLETE
- **Script**: `compute_sectional_porosity.py`
- **Method**: Plan-view porosity as proxy for wind access (fraction of open area in horizontal slice)
- **Outputs**: Porosity raster (.npy), CSV, heatmap visualization

## Phase 2.8: Occupancy Density Proxy ✅ COMPLETE
- **Script**: `compute_occupancy_density.py`
- **Method**: Built volume / open space ratio per analysis unit (auto-generated grid)
- **Outputs**: GeoDataFrame, density map, CSV summary

## Phase 2.9: Morphological Environmental Deprivation Index (Unit-level) ✅ COMPLETE
- **Script**: `compute_deprivation_index.py`
- **Method**: Composite index combining solar deficit, ventilation deficit, and occupancy pressure at unit level
- **Outputs**: Hotspot map, deficit overlap map, ranking table

## Phase 2.9.5: Morphological Environmental Deprivation Index (Raster-based) ✅ COMPLETE
- **Script**: `compute_deprivation_index_raster.py`
- **Method**: Continuous 2D raster of deprivation index with pixel-level occupancy pressure computation
- **Features**: Works with native raster resolutions, building mask, continuous and classified visualizations, unit-level aggregation
- **Outputs**: Raster (.npy), continuous heatmap, classified hotspot map, unit-level aggregation

## Phase 3: Multi-Area Comparative Analysis ✅ COMPLETE

### Phase 3.0: Data Organization ✅ COMPLETE
- **Structure**: Area-based data organization (`data/{area}/raw/`) for comparative analysis
- **Supported Areas**: 
  - `vidigal` (informal settlement)
  - `copacabana` (formal neighborhood)
- **Configuration**: `src/config.py` includes area classification and helper functions
- **Filtering Policy**: Formal areas (Copacabana) skip filtering; informal areas (Vidigal) apply filtering

### Phase 3.1: Comparative Analysis Framework ✅ COMPLETE
- **Script**: `scripts/compare_areas.py`
- **Features**: 
  - Comprehensive comparison of all metrics between formal and informal settlements
  - Statistical tests (Mann-Whitney U) with significance indicators
  - Area-normalized statistics for fair comparisons
  - Side-by-side visualizations preserving aspect ratios
  - Professional PDF report generation with clean Swiss design
- **Outputs**: PDF report, comparison tables, side-by-side visualizations
- **Results**: `outputs/comparative/comparison_report.pdf`

## Data Organization
- **Area-based structure**: `data/{area}/raw/` for input files
- **Area-based outputs**: `outputs/{area}/` for analysis results
- **Helper functions**: `src/config.py` provides `get_area_data_dir(area)` and `get_area_output_dir(area)`
- **Backward compatibility**: Legacy `data/raw/` path still supported