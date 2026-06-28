"""Unit tests for the E2 geometric multi-constraint ventilation-tendency index.

Pins the pure constraint-counting core: an ordinal 0–3 COUNT of independent
qualitative axes (no weighted sum of incommensurable continuous scales — the
council no-sum rule), and NaN-as-not-triggered semantics."""

import numpy as np

from scripts.run_ventilation_index import SKIM_MIN, count_constraints

DEPTH_MED = 32.0


def _one(lf, dist, ratio):
    return int(count_constraints(
        np.array([lf]), np.array([dist]), np.array([ratio]), DEPTH_MED)[0])


def test_unconstrained_is_zero():
    assert _one(lf=0.2, dist=10.0, ratio=0.5) == 0


def test_each_axis_contributes_exactly_one():
    assert _one(lf=SKIM_MIN, dist=0.0, ratio=0.0) == 1          # skimming only
    assert _one(lf=0.0, dist=DEPTH_MED, ratio=0.0) == 1         # deep only
    assert _one(lf=0.0, dist=0.0, ratio=1.0) == 1              # wind-aligned only


def test_all_three_is_max():
    assert _one(lf=0.9, dist=100.0, ratio=1.4) == 3


def test_thresholds_are_inclusive():
    # exactly at each threshold counts as triggered
    assert _one(lf=SKIM_MIN, dist=DEPTH_MED, ratio=1.0) == 3


def test_nan_never_manufactures_a_constraint():
    # a missing signal on any axis must not add a constraint
    assert _one(lf=np.nan, dist=np.nan, ratio=np.nan) == 0
    assert _one(lf=np.nan, dist=100.0, ratio=np.nan) == 1


def test_vectorised_count_matches_elementwise():
    lf = np.array([0.1, 0.7, 0.9, np.nan])
    dist = np.array([5.0, 40.0, 50.0, 50.0])
    ratio = np.array([0.3, 0.9, 1.2, 1.5])
    got = count_constraints(lf, dist, ratio, DEPTH_MED)
    assert got.tolist() == [0, 2, 3, 2]
    assert got.dtype.kind in "iu"
