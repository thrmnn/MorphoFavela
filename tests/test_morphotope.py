"""Block-scale morphotope — neighbourhood composition + diversity."""

import numpy as np
import pandas as pd

from src.morphometry.morphotope import (
    N_TYPES,
    cell_type_profile,
    neighborhood_composition,
)


def _grid(labels, spacing=10.0):
    n = len(labels)
    return pd.DataFrame({
        "site": ["s"] * n,
        "zone_id": range(n),
        "centroid_x": np.arange(n) * spacing,
        "centroid_y": np.zeros(n),
        "morphotype_smooth": labels,
    })


def test_uniform_tissue_zero_diversity():
    # all-same-type neighbourhood → composition concentrated, diversity 0
    comp = neighborhood_composition(_grid([5, 5, 5, 5, 5]), radius=25)
    assert np.allclose(comp["comp_5"], 1.0)
    assert np.allclose(comp["diversity"], 0.0)


def test_mixed_tissue_high_diversity():
    comp = neighborhood_composition(_grid([0, 1, 2, 3, 4, 5]), radius=15)
    # middle cell sees ~3 distinct types within 15 m → positive diversity
    assert comp["diversity"].max() > 0.4


def test_composition_sums_to_one():
    comp = neighborhood_composition(_grid([0, 0, 5, 5, 2, 2]), radius=25)
    cols = [f"comp_{c}" for c in range(N_TYPES)]
    assert np.allclose(comp[cols].sum(axis=1), 1.0)


def test_cell_type_profile_shapes():
    comp = neighborhood_composition(_grid([0, 0, 5, 5]), radius=25)
    labels = np.array([0, 0, 1, 1])
    prof = cell_type_profile(comp, labels)
    assert set(prof.index) == {0, 1}
    assert "diversity" in prof and "n_cells" in prof
