#!/usr/bin/env python3
"""
Test runner script for SVF test suite.
Checks dependencies and runs available tests.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Check which dependencies are available."""
    deps = {
        'numpy': False,
        'pyvista': False,
        'torch': False,
        'pytorch3d': False,
        'pytest': False,
    }
    
    try:
        import numpy  # noqa: F401
        deps['numpy'] = True
    except ImportError:
        pass
    
    try:
        import pyvista  # noqa: F401
        deps['pyvista'] = True
    except ImportError:
        pass
    
    try:
        import torch
        deps['torch'] = True
        if torch.cuda.is_available():
            deps['cuda_available'] = True
        else:
            deps['cuda_available'] = False
    except ImportError:
        deps['cuda_available'] = False
    
    try:
        import pytorch3d  # noqa: F401
        deps['pytorch3d'] = True
    except ImportError:
        pass
    
    try:
        import pytest  # noqa: F401
        deps['pytest'] = True
    except ImportError:
        pass
    
    return deps

def main():
    """Main test runner."""
    print("=" * 60)
    print("SVF Test Suite Runner")
    print("=" * 60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    deps = check_dependencies()
    
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    # Check if we can run tests
    if not deps['numpy']:
        print("\nERROR: numpy is required but not installed.")
        sys.exit(1)
    
    if not deps['pytest']:
        print("\nERROR: pytest is required but not installed.")
        print("Install with: pip install pytest")
        sys.exit(1)
    
    # Determine which tests can run
    can_run_cpu = deps['numpy'] and deps['pyvista']
    can_run_gpu = deps['torch'] and deps['pytorch3d'] and deps['cuda_available']
    
    print(f"\nCPU tests: {'✓ Available' if can_run_cpu else '✗ Missing dependencies'}")
    print(f"GPU tests: {'✓ Available' if can_run_gpu else '✗ Missing dependencies'}")
    
    if not can_run_cpu:
        print("\nERROR: Cannot run CPU tests. Missing required dependencies.")
        print("Install with: pip install numpy pyvista")
        sys.exit(1)
    
    # Run pytest
    print("\n" + "=" * 60)
    print("Running tests...")
    print("=" * 60)
    
    import pytest
    
    # Build pytest arguments
    args = [
        'tests/',
        '-v',
        '--tb=short'
    ]
    
    # Skip GPU tests if GPU not available
    if not can_run_gpu:
        args.append('-m')
        args.append('not cuda')
        print("\nNote: Skipping GPU tests (CUDA not available)")
    
    # Run tests
    exit_code = pytest.main(args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("All tests passed!")
    else:
        print(f"Some tests failed (exit code: {exit_code})")
    print("=" * 60)
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
