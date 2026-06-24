"""Print-model geometry: the plinth must be a watertight solid and scaling exact."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.print3d.model import heightmap_to_solid


def _flat_grid(n=5, z=10.0):
    xs = np.arange(n) * 5.0
    X, Y = np.meshgrid(xs, xs)
    Z = np.full((n, n), z)
    return X, Y, Z


def test_plinth_is_watertight_solid():
    X, Y, Z = _flat_grid()
    mesh = heightmap_to_solid(X, Y, Z, floor_z=0.0)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0


def test_plinth_volume_matches_box():
    # flat 20×20 m top at z=10 over a floor at z=0 → a 20×20×10 box
    X, Y, Z = _flat_grid(n=5, z=10.0)
    mesh = heightmap_to_solid(X, Y, Z, floor_z=0.0)
    assert mesh.volume == pytest.approx(20 * 20 * 10, rel=1e-6)


def test_relief_preserved_in_solid_bounds():
    X, Y, Z = _flat_grid()
    Z[2, 2] = 40.0  # a peak
    mesh = heightmap_to_solid(X, Y, Z, floor_z=0.0)
    assert mesh.bounds[1][2] == pytest.approx(40.0)
    assert mesh.bounds[0][2] == pytest.approx(0.0)
