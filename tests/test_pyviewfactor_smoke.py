"""
Smoke test for the PyViewFactor dependency.

This test lives in the test suite (not in scripts/) to ensure that:
- The IVF conda environment can import `pyviewfactor`
- Basic interoperability with PyVista works (we can create a simple mesh)

It does NOT yet exercise the full SVF pipeline; that will be wired through
`src.svf_compute` in a follow-up step and validated with the existing SVF tests.
"""

import numpy as np
import pyvista as pv


def test_pyviewfactor_import_and_mesh_construction():
    try:
        import pyviewfactor as pvf  # type: ignore[import]
    except ImportError as e:  # pragma: no cover - explicit failure path
        raise AssertionError(
            "Failed to import pyviewfactor. Make sure tests are run inside the IVF "
            "conda environment and that `pyviewfactor` is installed."
        ) from e

    # Basic sanity on the module
    assert hasattr(pvf, "compute_viewfactor") or hasattr(
        pvf, "batch_compute_viewfactors"
    ), "pyviewfactor does not expose expected viewfactor functions"

    # Build a trivial 10x10 m ground quad at z = 0 to ensure PyVista interop works
    x = np.array([0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    z = np.zeros_like(x)
    points = np.column_stack([x, y, z])

    # Single quad cell: [n_points, i0, i1, i2, i3]
    cells = np.hstack([[4], np.arange(4, dtype=np.int64)])

    mesh = pv.PolyData(points, cells)

    assert mesh.n_points == 4
    assert mesh.n_cells == 1

