"""WS-A.2 — contiguity mode filter + spatial purity."""

import numpy as np

from src.morphometry.signature import spatial_mode_smooth, spatial_purity


def _line_neighbors(n):
    # 1-D chain: each cell neighbours the previous and next
    return {i: [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)}


def test_mode_filter_removes_isolated_outlier():
    # a single type-1 cell embedded in a run of type 0 -> flips to 0
    labels = np.array([0.0, 0, 1, 0, 0])
    neigh = _line_neighbors(5)
    out = spatial_mode_smooth(labels, neigh, passes=1)
    assert list(out) == [0, 0, 0, 0, 0]


def test_mode_filter_keeps_genuine_block_and_ties():
    # two equal blocks: a 3-wide tie at the seam keeps current (conservative)
    labels = np.array([0.0, 0, 0, 1, 1, 1])
    out = spatial_mode_smooth(labels, _line_neighbors(6), passes=2)
    assert list(out) == [0, 0, 0, 1, 1, 1]


def test_mode_filter_leaves_nan_untouched():
    labels = np.array([0.0, np.nan, 1.0, 1.0])
    out = spatial_mode_smooth(labels, _line_neighbors(4), passes=1)
    assert np.isnan(out[1])
    assert not np.isnan(out[[0, 2, 3]]).any()


def test_spatial_purity_increases_after_smoothing():
    labels = np.array([0.0, 0, 1, 0, 0, 1, 0])
    neigh = _line_neighbors(7)
    before = spatial_purity(labels, neigh)
    after = spatial_purity(spatial_mode_smooth(labels, neigh, passes=2), neigh)
    assert after > before
    assert 0.0 <= before <= 1.0 and 0.0 <= after <= 1.0
