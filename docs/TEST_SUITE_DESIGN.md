# SVF Algorithm Test Suite Design

## Overview

This document describes the comprehensive test suite for the Sky View Factor (SVF) algorithm, designed to validate both CPU and GPU implementations and ensure they produce consistent, accurate results.

## Test Categories

### 1. Unit Tests

#### A. Sky Patch Generation
- **Test**: Number of patches matches requested count
- **Test**: Patches are within upper hemisphere (elevation 0 to π/2)
- **Test**: Patches are approximately equal-area or well-distributed
- **Test**: Patch centroids are at expected radius
- **Test**: Azimuth and elevation ranges are correct

#### B. Mesh Conversion Utilities
- **Test**: PyVista to PyTorch3D conversion preserves geometry
- **Test**: Invalid/NaN vertices are handled correctly
- **Test**: Face indices are remapped correctly after filtering
- **Test**: Device placement (CPU/CUDA) works correctly

#### C. Observer Point Preparation
- **Test**: Evaluation height is added correctly
- **Test**: Points are converted to correct tensor format
- **Test**: Device placement matches mesh device

### 2. Synthetic Scene Tests (Ground Truth)

#### A. Empty Scene (No Obstructions)
- **Expected**: SVF = 1.0 for all points
- **Purpose**: Verify algorithm doesn't introduce false obstructions

#### B. Single Building (Simple Geometry)
- **Test Cases**:
  - Point far from building: SVF ≈ 1.0
  - Point directly under building center: SVF ≈ 0.0
  - Point at building edge: SVF ≈ 0.5 (partial obstruction)
- **Purpose**: Validate basic ray-casting logic with known geometry

#### C. Two Buildings (Known Configuration)
- **Test**: Point between buildings has predictable SVF
- **Test**: Point behind one building has correct partial obstruction

#### D. Flat vs Sloped Terrain
- **Test**: SVF accounts for terrain elevation correctly
- **Test**: Points at different elevations have correct observer heights

### 3. CPU vs GPU Consistency Tests

#### A. Numerical Accuracy
- **Thresholds**:
  - Mean absolute difference < 0.01
  - Maximum absolute difference < 0.05
  - Correlation coefficient > 0.95
- **Test**: Distribution similarity (Kolmogorov-Smirnov test)

#### B. Edge Cases
- Points at building boundaries
- Points very close to building surfaces
- Points at terrain edges
- Points with extreme coordinates
- Points with zero elevation

#### C. Parameter Sensitivity
- Different sky patch counts (36, 72, 145, 290)
- Different evaluation heights (0.5m, 1.5m, 2.0m)
- Different batch sizes (GPU only)
- Different ray/triangle chunk sizes (GPU only)

### 4. Real-World Data Tests

#### A. Small Dataset Validation
- Run on subset of real streets (10-50 points)
- Compare CPU vs GPU results
- Verify results are reasonable (SVF in [0, 1])

#### B. Medium Dataset Consistency
- Run on medium dataset (100-1000 points)
- Compare statistical distributions
- Check for systematic biases

### 5. Performance Tests

#### A. Speed Comparison
- Measure computation time for both implementations
- Test scaling with number of points
- Test scaling with number of sky patches
- **Expected**: GPU speedup should be significant for large datasets

#### B. Memory Usage
- Monitor GPU memory usage
- Test with different batch sizes
- Ensure no memory leaks

### 6. Robustness Tests

#### A. Invalid Inputs
- Empty mesh
- Mesh with no triangles
- Points outside mesh bounds
- Invalid sky patches

#### B. Boundary Conditions
- Very small sky patch counts (e.g., 4 patches)
- Very large sky patch counts (e.g., 1000 patches)
- Very high/low evaluation heights

#### C. Numerical Stability
- Points with very small/large coordinates
- Mesh with degenerate triangles
- Mesh with duplicate vertices

### 7. Regression Tests

#### A. Known Good Results
- Store reference results for specific test scenes
- Ensure new code changes don't break existing results
- Track changes in accuracy over time

### 8. Integration Tests

#### A. End-to-End Pipeline
- Full workflow: load mesh → generate patches → compute SVF → save results
- Test both CPU and GPU paths
- Verify output formats are correct

## Test Structure

### Test Files Organization

```
tests/
├── test_svf_unit.py              # Unit tests
├── test_svf_synthetic.py          # Synthetic scene tests
├── test_svf_cpu_gpu_consistency.py  # CPU vs GPU comparison
├── test_svf_performance.py        # Performance benchmarks
├── test_svf_robustness.py         # Edge cases and error handling
├── test_svf_integration.py        # End-to-end tests
└── fixtures/                      # Test data
    ├── synthetic_meshes/          # Synthetic STL files
    ├── reference_results/         # Known good results
    └── small_datasets/            # Small real-world subsets
```

### Test Utilities

- `create_synthetic_mesh()` - Generate test meshes
- `compare_cpu_gpu_results()` - Compare with statistics
- `generate_test_points()` - Create test point sets
- `assert_svf_valid()` - Validate SVF values

## Assertions and Thresholds

- **Absolute tolerance**: 0.01 for most comparisons
- **Relative tolerance**: 1% for statistical comparisons
- **Correlation threshold**: 0.95 for CPU vs GPU
- **Performance threshold**: GPU should be at least 2x faster for large datasets

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_svf_unit.py
pytest tests/test_svf_cpu_gpu_consistency.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Continuous Integration

Tests should be run:
- On every pull request
- Before merging to main
- On schedule (nightly) to catch regressions
