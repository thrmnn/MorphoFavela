"""Unit tests for the fig04 compound-state spatial-clustering primitives.

Exercises the two pure helpers that turn the regular analysis grid into a rook
contiguity graph and extract connected-component (patch) sizes — the machinery
behind the Moran's I / join-count evidence that compound cells cluster. Pure
geometry, no site data required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "fig04_diagnostic_taxonomy",
    ROOT / "outputs" / "paper_figures" / "fig04_diagnostic_taxonomy.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _lattice_df(coords, spacing=10.0):
    """Build a frame with centroid_x/centroid_y from integer lattice coords."""
    xs = [c[0] * spacing for c in coords]
    ys = [c[1] * spacing for c in coords]
    return pd.DataFrame({"centroid_x": xs, "centroid_y": ys})


def test_rook_neighbours_on_full_block():
    # 3×3 fully-populated block: corners have 2 rook neighbours, edges 3,
    # the centre 4 — diagonals are NOT neighbours (rook, not queen).
    coords = [(i, j) for i in range(3) for j in range(3)]
    g = _lattice_df(coords)
    nb = mod._lattice_neighbors(g)
    deg = {k: len(v) for k, v in nb.items()}
    centre = coords.index((1, 1))
    corner = coords.index((0, 0))
    edge = coords.index((1, 0))
    assert deg[centre] == 4
    assert deg[corner] == 2
    assert deg[edge] == 3
    # adjacency is symmetric
    for k, vs in nb.items():
        for j in vs:
            assert k in nb[j]


def test_patch_sizes_connected_and_isolated():
    # An L-tromino (3 contiguous cells) plus one detached cell → patches [3, 1].
    coords = [(0, 0), (0, 1), (1, 1), (5, 5)]
    g = _lattice_df(coords)
    nb = mod._lattice_neighbors(g)
    mask = np.ones(len(coords), dtype=bool)
    sizes = mod._patch_sizes(nb, mask)
    assert sizes == [3, 1]


def test_patch_sizes_respects_mask():
    # A straight run of 4 contiguous cells, but mask drops the middle two →
    # two isolated single-cell patches.
    coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
    g = _lattice_df(coords)
    nb = mod._lattice_neighbors(g)
    mask = np.array([True, False, False, True])
    sizes = mod._patch_sizes(nb, mask)
    assert sizes == [1, 1]
