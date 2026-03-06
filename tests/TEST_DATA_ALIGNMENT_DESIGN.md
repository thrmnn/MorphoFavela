# Test Suite Design: Data Alignment and Integration

## Overview

This test suite ensures proper alignment and integration of multiple geospatial data sources:
- **DTM (Digital Terrain Model)**: Raster elevation data
- **Building Polygons**: Vector footprint data (shapefiles/GeoPackages)
- **Road Networks**: Vector line data (shapefiles/GeoPackages)
- **STL Files**: 3D mesh files (typically ungeoreferenced)

## Key Requirements

1. **Alignment**: All datasets must be in the same coordinate system and aligned spatially
2. **STL Georeferencing**: STL files lack georeference info but are created by extruding building footprints on DTM
3. **Road-Building Intersection Handling**: Roads intersecting buildings must be redirected to run between buildings

## Test Categories

### 1. Data Loading and Validation Tests

#### 1.1 DTM Loading
- Load DTM raster file
- Validate CRS is present
- Validate data type (float/int)
- Validate bounds (min/max elevation)
- Validate no-data handling
- Validate resolution and extent

#### 1.2 Building Polygon Loading
- Load building polygons (shapefile/GeoPackage/GeoJSON)
- Validate CRS is present
- Validate geometry types (Polygon/MultiPolygon)
- Validate required attributes (height columns)
- Validate non-empty geometries
- Validate no duplicate geometries

#### 1.3 Road Network Loading
- Load road network (shapefile/GeoPackage)
- Validate CRS is present
- Validate geometry types (LineString/MultiLineString)
- Validate network connectivity
- Validate no self-intersections (optional)
- Validate no duplicate segments

#### 1.4 STL File Loading
- Load STL mesh
- Validate mesh structure (points, faces)
- Validate mesh bounds
- Validate no degenerate triangles
- Validate mesh is closed (optional)

### 2. Coordinate System Alignment Tests

#### 2.1 CRS Detection and Validation
- Detect CRS from each dataset
- Validate CRS is not None
- Validate CRS is valid (can be transformed)
- Test with missing CRS (should raise error or use default)

#### 2.2 CRS Transformation
- Transform all datasets to common CRS
- Validate transformation accuracy
- Test with different source CRS (e.g., WGS84, UTM)
- Test with projected vs geographic CRS
- Validate coordinates after transformation

#### 2.3 STL Alignment with Building Footprints
- Test center-based alignment (current approach)
- Test corner-based alignment
- Test extent-based alignment
- Validate alignment accuracy (should match within tolerance)
- Test with misaligned data (should detect and correct)

#### 2.4 STL Alignment with DTM
- Test STL terrain matches DTM elevation
- Validate STL bounds match DTM extent
- Test elevation extraction from DTM matches STL terrain
- Validate vertical alignment (Z coordinates)

### 3. Spatial Extent Alignment Tests

#### 3.1 Bounds Comparison
- Compare bounds of all datasets
- Validate datasets overlap
- Test with non-overlapping datasets (should warn/error)
- Validate extent differences are within tolerance

#### 3.2 Extent Cropping
- Crop datasets to common extent
- Validate cropped datasets maintain geometry validity
- Test with partial overlaps
- Validate no data loss for overlapping regions

#### 3.3 Grid Alignment
- Validate grid points align with DTM pixels
- Validate grid points align with building footprints
- Test grid spacing consistency
- Validate grid covers all datasets

### 4. STL Georeferencing Tests

#### 4.1 STL Creation from Building Footprints
- Test STL creation by extruding footprints
- Validate building heights match STL geometry
- Validate building positions match STL positions
- Test with different height sources (attribute vs constant)

#### 4.2 STL Creation from DTM
- Test terrain extraction from DTM
- Validate DTM elevation matches STL terrain
- Test interpolation for non-grid points
- Validate terrain smoothness

#### 4.3 Combined STL (Terrain + Buildings)
- Test STL contains both terrain and buildings
- Validate no gaps between terrain and buildings
- Validate building bases sit on terrain
- Test building heights are correct

#### 4.4 STL Coordinate System
- Test STL uses local coordinates (centered at origin)
- Validate STL coordinates match transformed building footprints
- Test coordinate transformation for STL
- Validate STL can be georeferenced post-hoc

### 5. Road-Building Intersection Tests

#### 5.1 Intersection Detection
- Detect roads that intersect building polygons
- Count intersection points/segments
- Identify which buildings are intersected
- Validate intersection geometry (Point/LineString)

#### 5.2 Road Redirection Logic
- Redirect roads to run between buildings
- Validate redirected roads don't intersect buildings
- Test with different building configurations
- Validate road connectivity is maintained

#### 5.3 Redirection Algorithms
- Test parallel offset method (shift road away from building)
- Test pathfinding method (find route around building)
- Test simplification method (remove intersecting segments)
- Validate method preserves road network topology

#### 5.4 Redirection Validation
- Validate redirected roads are valid LineStrings
- Validate redirected roads maintain connectivity
- Validate redirected roads don't create new intersections
- Test with complex intersection scenarios

### 6. Integration Tests

#### 6.1 Full Pipeline Integration
- Test complete data loading pipeline
- Validate all datasets are aligned
- Test SVF computation with aligned data
- Validate results are spatially correct

#### 6.2 Error Handling
- Test with missing files
- Test with invalid CRS
- Test with non-overlapping extents
- Test with corrupted data
- Validate appropriate error messages

#### 6.3 Performance Tests
- Test loading large datasets
- Test transformation performance
- Test intersection detection performance
- Test redirection algorithm performance

### 7. Edge Cases

#### 7.1 Empty Datasets
- Test with empty building polygons
- Test with empty road network
- Test with empty STL (terrain only)

#### 7.2 Single Feature Datasets
- Test with single building
- Test with single road segment
- Test with minimal STL

#### 7.3 Complex Geometries
- Test with MultiPolygon buildings
- Test with MultiLineString roads
- Test with complex building shapes

#### 7.4 Boundary Cases
- Test with buildings at dataset boundaries
- Test with roads at dataset boundaries
- Test with STL at coordinate system limits

## Test Implementation Strategy

### Phase 1: Unit Tests
- Individual data loading functions
- CRS detection and transformation
- Bounds calculation and comparison
- Intersection detection algorithms

### Phase 2: Integration Tests
- Multi-dataset alignment
- STL creation from multiple sources
- Road redirection with buildings
- Full pipeline validation

### Phase 3: Real-World Tests
- Test with actual project data
- Validate against known good results
- Performance benchmarking
- Regression testing

## Expected Test Outcomes

1. **All datasets load successfully** with proper validation
2. **All datasets align** within specified tolerance (e.g., 0.1m)
3. **STL matches building footprints** in position and height
4. **STL terrain matches DTM** elevation within tolerance
5. **Roads are redirected** to avoid building intersections
6. **SVF computation works** with aligned data
7. **Error handling** provides clear messages for misaligned data

## Tolerance Values

- **Spatial alignment tolerance**: 0.1m (10cm)
- **Elevation alignment tolerance**: 0.5m (50cm)
- **CRS transformation tolerance**: 1e-6 (for coordinate precision)
- **Grid alignment tolerance**: Half grid spacing

## Test Data Requirements

### Synthetic Test Data
- Simple building footprints (rectangles)
- Simple road network (grid pattern)
- Synthetic DTM (flat or sloped)
- Generated STL from footprints + DTM

### Real Test Data
- Use existing project data (vidigal_tls, copacabana)
- Validate against known good configurations
- Test with different CRS and extents
