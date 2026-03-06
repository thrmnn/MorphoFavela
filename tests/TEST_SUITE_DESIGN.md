# SVF Algorithm Test Suite Design

## Overview

This document describes the comprehensive test suite for the Sky View Factor (SVF) algorithm, ensuring both CPU and GPU implementations produce consistent and accurate results.

## Test Categories

### 1. Unit Tests

#### A. Sky Patch Generation (`test_svf_unit.py`)
- **Number of patches**: Verify requested count matches generated count
- **Hemisphere constraint**: All patches within upper hemisphere (elevation 0 to π/2)
- **Distribution**: Patches are well-distributed across azimuth and elevation
- **Radius**: Patch centroids at expected distance from origin
- **Coordinate system**: Patches in correct 3D coordinate system

#### B. Mesh Conversion Utilities (`test_svf_unit.py`)
- **PyVista to PyTorch3D**: Geometry preserved during conversion
- **Invalid vertex handling**: NaN/invalid vertices filtered correctly
- **Face remapping**: Face indices remapped after vertex filtering
- **Device placement**: Correct device (CPU/CUDA) assignment

#### C. Observer Point Preparation (`test_svf_unit.py`)
- **Height offset**: Evaluation height correctly added
- **Tensor format**: Correct tensor dtype and shape
- **Device consistency**: Points on same device as mesh

### 2. Synthetic Scene Tests (`test_svf_synthetic.py`)

#### A. Empty Scene (No Obstructions)
- **Expected**: SVF = 1.0 for all points
- **Purpose**: Verify no false obstructions

#### B. Single Building (Simple Box)
- **Point far from building**: SVF ≈ 1.0
- **Point directly under building**: SVF ≈ 0.0
- **Point at building edge**: SVF ≈ 0.5 (partial obstruction)
- **Analytical verification**: Compare with calculated expected values

#### C. Two Buildings (Known Configuration)
- **Point between buildings**: Predictable SVF based on geometry
- **Point behind one building**: SVF blocked in specific directions

#### D. Terrain Variations
- **Flat terrain**: SVF accounts for elevation correctly
- **Sloped terrain**: Observer height adjusted properly

### 3. CPU vs GPU Consistency Tests (`test_svf_cpu_gpu_consistency.py`)

#### A. Numerical Accuracy
- **Mean absolute difference**: < 0.01
- **Maximum absolute difference**: < 0.05
- **Correlation coefficient**: > 0.95
- **Distribution similarity**: Kolmogorov-Smirnov test

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

### 4. Real-World Data Tests (`test_svf_integration.py`)

#### A. Small Dataset Validation
- Run on subset of real streets (10-50 points)
- Compare CPU vs GPU results
- Verify reasonable values (SVF in [0, 1])

#### B. Medium Dataset Consistency
- Run on medium dataset (100-1000 points)
- Compare statistical distributions
- Check for systematic biases

### 5. Performance Tests (`test_svf_performance.py`)

#### A. Speed Comparison
- Measure computation time for both implementations
- Test scaling with number of points
- Test scaling with number of sky patches
- GPU speedup should be significant for large datasets

#### B. Memory Usage
- Monitor GPU memory usage
- Test with different batch sizes
- Ensure no memory leaks

### 6. Robustness Tests (`test_svf_robustness.py`)

#### A. Invalid Inputs
- Empty mesh
- Mesh with no triangles
- Points outside mesh bounds
- Invalid sky patches

#### B. Boundary Conditions
- Very small sky patch counts (e.g., 4 patches)
- Very large sky patch counts (e.g., 1000 patches)
- Very high evaluation height
- Very low evaluation height

#### C. Numerical Stability
- Points with very small coordinates
- Points with very large coordinates
- Mesh with degenerate triangles
- Mesh with duplicate vertices

### 7. Regression Tests (`test_svf_regression.py`)

#### A. Known Good Results
- Store reference results for specific test scenes
- Ensure new code changes don't break existing results
- Track changes in accuracy over time

### 8. Integration Tests (`test_svf_integration.py`)

#### A. End-to-End Pipeline
- Full workflow: load mesh → generate patches → compute SVF → save results
- Test both CPU and GPU paths
- Verify output formats are correct

## Test Structure

```
tests/
├── TEST_SUITE_DESIGN.md          # This document
├── test_svf_unit.py              # Unit tests
├── test_svf_synthetic.py         # Synthetic scene tests
├── test_svf_cpu_gpu_consistency.py  # CPU vs GPU comparison
├── test_svf_performance.py       # Performance benchmarks
├── test_svf_robustness.py        # Edge cases and error handling
├── test_svf_integration.py       # End-to-end tests
├── test_svf_regression.py        # Regression tests
├── fixtures/                     # Test data
│   ├── synthetic/
│   │   ├── empty_scene.stl
│   │   ├── single_building.stl
│   │   └── two_buildings.stl
│   └── reference_results/
│       └── *.json
└── utils/
    └── test_helpers.py           # Test utilities
```

## Test Utilities

### Helper Functions

1. **`create_synthetic_mesh()`**: Generate synthetic STL meshes for testing
2. **`compare_cpu_gpu_results()`**: Compare CPU vs GPU results with statistics
3. **`generate_test_points()`**: Generate test points (grid, random, edge cases)
4. **`assert_svf_valid()`**: Assert SVF values are in valid range [0, 1]
5. **`assert_cpu_gpu_similar()`**: Assert CPU and GPU results are similar

## Assertions and Thresholds

- **Absolute tolerance**: 0.01 for most comparisons
- **Relative tolerance**: 1% for statistical comparisons
- **Correlation threshold**: 0.95 for CPU vs GPU
- **Performance threshold**: GPU should be at least 2x faster for large datasets
- **SVF range**: All values must be in [0, 1]

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_svf_unit.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Data Requirements

- Small synthetic STL files (included in repo)
- Small real-world STL subset (for integration tests)
- Reference results stored as JSON/CSV for regression testing

## Success Criteria

1. ✅ All unit tests pass
2. ✅ CPU and GPU produce similar results (correlation > 0.95)
3. ✅ Synthetic scene tests match expected ground truth
4. ✅ No regressions in known good results
5. ✅ Performance tests show GPU speedup for large datasets
6. ✅ Robustness tests handle edge cases gracefully
