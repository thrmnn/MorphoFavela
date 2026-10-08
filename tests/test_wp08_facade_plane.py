import numpy as np
import pandas as pd

from scripts.wp08_facade_plane import first_sustained, robust_storeys, stats, wquantile


def test_wquantile_equal_weights_matches_median():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert wquantile(x, np.ones(5), 0.5) == 3.0


def test_stats_share_and_counts():
    v = np.array([0.0, 0.4, 0.5, 2.0])
    s = stats(v, np.ones(4))
    assert s["bands"] == 4
    assert s["share_below_screen_count"] == 0.5


def test_first_sustained_needs_next_storey():
    assert first_sustained({1: False, 2: True, 3: False, 4: True, 5: True}) == 4
    assert first_sustained({1: True, 2: False}) is None
    assert first_sustained({1: False, 2: True}) == 2


def test_robust_storeys_stops_at_first_gap():
    c = pd.Series({1: 600, 2: 700, 3: 10, 4: 900})
    assert robust_storeys(c) == [1, 2]
