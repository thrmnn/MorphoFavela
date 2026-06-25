"""Distance-to-open-edge lateral-connectivity scalar (the pure EDT core)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lateral_connectivity import open_edge_distance  # noqa: E402


def _lattice(nx, ny, cell=10.0):
    """Full rectangular lattice of cell centroids (row-major)."""
    xs = np.arange(nx) * cell
    ys = np.arange(ny) * cell
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def test_single_built_cell_is_one_step_from_open():
    cx, cy = _lattice(3, 3)
    built = np.zeros(9, bool)
    built[4] = True  # centre of a 3x3 lattice, all neighbours open
    d = open_edge_distance(cx, cy, built)
    assert d[4] == 10.0  # one 10 m step to the nearest open cell
    assert (d[~built] == 0.0).all()


def test_interior_is_deeper_than_edge():
    # a fully-built 7x7 block: the centre must be deeper than an edge-row cell
    n = 7
    cx, cy = _lattice(n, n)
    built = np.ones(n * n, bool)
    d = open_edge_distance(cx, cy, built).reshape(n, n)
    assert d[n // 2, n // 2] == d.max()  # centre is the deepest
    assert d[0, n // 2] < d[n // 2, n // 2]  # an edge cell is shallower
    assert d[0, 0] == 10.0  # a corner sits one step from the padded open border


def test_perimeter_pad_counts_as_open():
    # one built cell at a lattice corner: the padded exterior is its open edge
    cx, cy = _lattice(4, 4)
    built = np.zeros(16, bool)
    built[0] = True  # corner cell
    d = open_edge_distance(cx, cy, built)
    assert d[0] == 10.0


def test_distance_scales_with_cell_size():
    cx, cy = _lattice(3, 3, cell=25.0)
    built = np.zeros(9, bool)
    built[4] = True
    d = open_edge_distance(cx, cy, built, cell=25.0)
    assert d[4] == 25.0
