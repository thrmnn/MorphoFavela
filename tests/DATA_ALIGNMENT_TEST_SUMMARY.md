# Data Alignment Test Suite Summary

## Overview

This test suite validates the alignment and integration of multiple geospatial data sources:
- **DTM (Digital Terrain Model)**: Raster elevation data
- **Building Polygons**: Vector footprint data
- **Road Networks**: Vector line data
- **STL Files**: 3D mesh files (typically ungeoreferenced)

## Key Features

### 1. Data Alignment
- **Coordinate System Alignment**: Ensures all datasets use the same CRS
- **Spatial Alignment**: Validates datasets are aligned within 0.1m tolerance
- **Auto-Correction**: Automatically corrects misaligned data with warnings

### 2. Road Redirection
- **Intersection Detection**: Detects roads that intersect building polygons
- **Parallel Offset**: Redirects roads using parallel offset method (default)
- **Simple Rerouting**: Alternative method for complex cases
- **Preprocessing**: Runs automatically before SVF computation

### 3. Visualization
- **Debug Plots**: Show original roads, redirected roads, and buildings
- **Color Coding**: 
  - Blue = Filtered streets
  - Purple dashed = Redirected roads
  - Red = Main cluster buildings
  - Green = Sample points

## Test Coverage

### Data Loading Tests
- ✅ Building polygon loading and validation
- ✅ Road network loading and validation
- ✅ STL mesh loading and validation
- ✅ DTM raster loading (if rasterio available)

### CRS Alignment Tests
- ✅ Same CRS detection
- ✅ Different CRS detection
- ✅ CRS auto-correction

### Spatial Alignment Tests
- ✅ Aligned bounds validation
- ✅ Misaligned bounds detection
- ✅ Spatial auto-correction

### STL Alignment Tests
- ✅ STL bounds alignment with buildings
- ✅ STL mesh alignment validation

### Road-Building Intersection Tests
- ✅ No intersections detection
- ✅ Intersection detection
- ✅ Multiple intersections handling

### Road Redirection Tests
- ✅ Parallel offset redirection
- ✅ Simple reroute redirection
- ✅ Network connectivity preservation

### Integration Tests
- ✅ Full alignment pipeline
- ✅ Alignment + redirection pipeline

## Usage

### Running Tests

```bash
# Run all data alignment tests
pytest tests/test_data_alignment.py -v

# Run specific test category
pytest tests/test_data_alignment.py::TestRoadRedirection -v

# Run with coverage
pytest tests/test_data_alignment.py --cov=src/data_alignment_utils
```

### Integration with SVF Computation

Road redirection is automatically integrated into `compute_svf_streets.py`:

```bash
python scripts/compute_svf_streets.py \
    --stl data/vidigal_tls/raw/full_scan.stl \
    --roads data/vidigal_tls/raw/roads_vidigal.shp \
    --footprints data/vidigal_tls/raw/vidigal_buildings.shp \
    --spacing 3.0 \
    --height 1.5
```

The script will:
1. Load and align all datasets
2. Detect road-building intersections
3. Redirect intersecting roads automatically
4. Show redirected roads in debug visualization
5. Compute SVF on redirected road network

## Configuration

### Alignment Tolerance
- Default: **0.1m** (10cm)
- Configurable via `ALIGNMENT_TOLERANCE` in `src/data_alignment_utils.py`

### Road Redirection Parameters
- **Method**: `parallel_offset` (default) or `simple_reroute`
- **Offset Distance**: 2.0m (default)
- **Buffer Distance**: 2.0m (for simple reroute)

## Implementation Details

### Files Created
1. **`src/data_alignment_utils.py`**: Core utilities for alignment and redirection
2. **`tests/test_data_alignment.py`**: Comprehensive test suite
3. **`tests/TEST_DATA_ALIGNMENT_DESIGN.md`**: Design documentation

### Integration Points
- **`scripts/compute_svf_streets.py`**: Road redirection preprocessing
- **`src/svf_utils.py`**: Building footprint alignment (existing)

## Future Enhancements

1. **Advanced Redirection**: Pathfinding algorithms for complex cases
2. **DTM Integration**: Validate STL terrain matches DTM elevation
3. **Performance Optimization**: Batch processing for large datasets
4. **Visualization Improvements**: Interactive plots showing redirection process
