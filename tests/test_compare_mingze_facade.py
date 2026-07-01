"""Unit tests for the Mingze façade-solar cross-check pure helper."""

import numpy as np

from scripts.compare_mingze_facade import deprivation_fraction


def test_deprivation_fraction_basic():
    # 2.0 is the floor: < 2 counts as deprived, exactly 2 does not.
    h = np.array([0.0, 1.0, 1.999, 2.0, 3.0, 5.0])
    assert abs(deprivation_fraction(h) - 3 / 6) < 1e-9


def test_deprivation_fraction_custom_threshold():
    h = np.array([0.5, 1.5, 2.5, 3.5])
    assert abs(deprivation_fraction(h, threshold=3.0) - 3 / 4) < 1e-9


def test_deprivation_fraction_drops_nan_and_empty():
    assert abs(deprivation_fraction(np.array([np.nan, 1.0, np.nan, 3.0])) - 0.5) < 1e-9
    assert np.isnan(deprivation_fraction(np.array([np.nan, np.nan])))
    assert np.isnan(deprivation_fraction(np.array([])))
