"""Unit tests for the E5 cross-site risk-map out-of-envelope flag.

Pins the extrapolation gate: a calibration cell is out-of-envelope iff ANY feature
leaves the campaign [p1,p99] training range (column-aligned)."""

import numpy as np

from scripts.run_cross_site_riskmap import out_of_envelope_mask

ENV = [(0.0, 1.0), (10.0, 20.0)]  # two features, (lo, hi) each


def test_inside_envelope_is_false():
    X = np.array([[0.5, 15.0], [1.0, 20.0], [0.0, 10.0]])  # interior + both bounds
    assert out_of_envelope_mask(X, ENV).tolist() == [False, False, False]


def test_any_feature_out_triggers():
    X = np.array([
        [-0.1, 15.0],   # feature 0 below lo
        [1.1, 15.0],    # feature 0 above hi
        [0.5, 9.9],     # feature 1 below lo
        [0.5, 20.1],    # feature 1 above hi
    ])
    assert out_of_envelope_mask(X, ENV).tolist() == [True, True, True, True]


def test_bounds_are_inclusive():
    X = np.array([[0.0, 10.0], [1.0, 20.0]])
    assert not out_of_envelope_mask(X, ENV).any()


def test_returns_bool_array_of_right_length():
    X = np.zeros((7, 2))
    m = out_of_envelope_mask(X, ENV)
    assert m.dtype == bool and m.shape == (7,)
