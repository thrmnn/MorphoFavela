# SVF Algorithm Test Suite

## Overview

This test suite validates the Sky View Factor (SVF) algorithm implementation, ensuring both CPU and GPU versions produce consistent and accurate results.

## Test Structure

- `test_svf_unit.py` - Unit tests for individual components
- `test_svf_synthetic.py` - Synthetic scene tests with known ground truth
- `test_svf_cpu_gpu_consistency.py` - CPU vs GPU comparison tests
- `test_svf_robustness.py` - Edge cases and error handling
- `test_svf_integration.py` - End-to-end pipeline tests
- `utils/test_helpers.py` - Test utilities and helper functions

## Requirements

The test suite requires the same dependencies as the main project:
- numpy
- pyvista
- torch (for GPU tests)
- pytorch3d (for GPU tests)
- pytest
- All other project dependencies from `requirements.txt`

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_svf_unit.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run only CPU tests (skip GPU tests):
```bash
pytest tests/ -m "not cuda"
```

### Run only GPU tests:
```bash
pytest tests/ -m "cuda"
```

### Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Categories

### Unit Tests
- Sky patch generation
- Mesh conversion utilities
- Observer point preparation
- GPU availability checks

### Synthetic Scene Tests
- Empty scene (no obstructions)
- Single building scenarios
- Two building configurations
- Known ground truth validation

### CPU vs GPU Consistency
- Numerical accuracy comparison
- Parameter sensitivity
- Edge case handling
- Batch size variations

### Robustness Tests
- Invalid input handling
- Boundary conditions
- Numerical stability
- Large dataset handling

### Integration Tests
- End-to-end pipeline
- Output consistency
- Reproducibility

## Expected Results

- All unit tests should pass
- CPU and GPU should produce similar results (correlation > 0.95)
- Synthetic scenes should match expected ground truth
- No regressions in known good results
- Robustness tests should handle edge cases gracefully

## Troubleshooting

### GPU tests skipped
If GPU tests are skipped, ensure:
- CUDA-capable GPU is available
- PyTorch with CUDA support is installed
- `torch.cuda.is_available()` returns `True`

### Import errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install pytest
```

### Memory errors
For large dataset tests, you may need to:
- Reduce batch size in GPU tests
- Use smaller test datasets
- Increase available GPU memory
