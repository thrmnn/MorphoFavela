# Data Alignment and Road Redirection Implementation

## Summary

Successfully implemented comprehensive test suite and functionality for:
1. **Data alignment** between DTM, building polygons, roads, and STL files
2. **Road redirection** to avoid building intersections
3. **Auto-correction** for misaligned data with warnings
4. **Visualization** of redirected roads in debug plots

## Implementation Status

✅ **All 20 tests passing**

### Files Created/Modified

1. **`src/data_alignment_utils.py`** (NEW)
   - `check_crs_alignment()`: Validates CRS and spatial alignment
   - `auto_correct_alignment()`: Automatically corrects misaligned data
   - `detect_road_building_intersections()`: Finds roads intersecting buildings
   - `redirect_roads()`: Redirects roads using parallel offset or simple rerouting
   - `redirect_road_parallel_offset()`: Parallel offset implementation
   - `redirect_road_simple_reroute()`: Simple rerouting implementation

2. **`tests/test_data_alignment.py`** (NEW)
   - 20 comprehensive tests covering all functionality
   - Test categories: Data Loading, CRS Alignment, Spatial Alignment, STL Alignment, Road Intersections, Road Redirection, Integration

3. **`scripts/compute_svf_streets.py`** (MODIFIED)
   - Integrated road redirection as preprocessing step
   - Added visualization of redirected roads (purple dashed lines)
   - Added intersection statistics to debug plots

4. **`tests/TEST_DATA_ALIGNMENT_DESIGN.md`** (NEW)
   - Comprehensive design documentation

5. **`tests/DATA_ALIGNMENT_TEST_SUMMARY.md`** (NEW)
   - Test suite summary and usage guide

## Key Features

### 1. Alignment Validation
- **Tolerance**: 0.1m (10cm) default
- **CRS Detection**: Validates all datasets use same coordinate system
- **Bounds Checking**: Validates spatial alignment of dataset extents
- **Auto-Correction**: Automatically transforms/translates misaligned data

### 2. Road Redirection
- **Methods**: 
  - `parallel_offset`: Offsets road parallel to avoid buildings (default)
  - `simple_reroute`: Clips and reroutes around buildings
- **Preprocessing**: Runs automatically before SVF computation
- **Validation**: Verifies redirected roads don't intersect buildings

### 3. Visualization
- **Debug Plots**: Show original and redirected roads
- **Color Coding**:
  - Blue = Filtered streets
  - **Purple dashed = Redirected roads** (NEW)
  - Red = Main cluster buildings
  - Green = Sample points
  - Yellow = Area-filtered buildings
  - Gray = Isolated buildings

## Usage Example

```python
from src.data_alignment_utils import check_crs_alignment, auto_correct_alignment, redirect_roads

# Check alignment
datasets = {
    'buildings': buildings_gdf,
    'roads': roads_gdf,
    'stl': stl_bounds
}

is_aligned, warnings, corrections = check_crs_alignment(datasets)

# Auto-correct if needed
if not is_aligned:
    datasets = auto_correct_alignment(datasets, corrections)

# Redirect roads
redirected_roads, intersections = redirect_roads(
    datasets['roads'],
    datasets['buildings'],
    method='parallel_offset',
    offset_distance=2.0
)
```

## Integration with SVF Computation

Road redirection is automatically integrated into `compute_svf_streets.py`:

1. **Loads and aligns** all datasets (buildings, roads, STL)
2. **Detects intersections** between roads and buildings
3. **Redirects roads** automatically (if intersections found)
4. **Shows redirected roads** in debug visualization
5. **Computes SVF** on redirected road network

## Test Results

```
======================== 20 passed, 1 warning in 0.76s =========================
```

### Test Coverage
- ✅ Data loading and validation (4 tests)
- ✅ CRS alignment (3 tests)
- ✅ Spatial alignment (3 tests)
- ✅ STL alignment (2 tests)
- ✅ Road-building intersections (3 tests)
- ✅ Road redirection (3 tests)
- ✅ Integration (2 tests)

## Next Steps

1. **Run on real data**: Test with actual project datasets
2. **Performance optimization**: Batch processing for large datasets
3. **Advanced redirection**: Pathfinding algorithms for complex cases
4. **DTM validation**: Verify STL terrain matches DTM elevation

## Notes

- Alignment tolerance is 0.1m (configurable)
- Road redirection uses 2.0m offset distance (configurable)
- Auto-correction warns before applying transformations
- Visualization shows both original and redirected roads for comparison
