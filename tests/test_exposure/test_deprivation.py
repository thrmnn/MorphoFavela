"""Tests for the shared deprivation-index formulas.

These verify that the formulas behave identically across the two
caller types (numpy ndarray vs pandas Series) so the unit-level
and raster-level deprivation scripts can't silently drift.
"""

import numpy as np
import pandas as pd
import pytest

from src.exposure import deprivation as d


class TestSolarDeficit:
    def test_zero_hours_full_deficit(self):
        assert d.solar_deficit(pd.Series([0.0]), reference=3.0).iloc[0] == pytest.approx(1.0)

    def test_at_reference_zero_deficit(self):
        assert d.solar_deficit(pd.Series([3.0]), reference=3.0).iloc[0] == pytest.approx(0.0)

    def test_above_reference_clipped_to_zero(self):
        assert d.solar_deficit(pd.Series([5.0]), reference=3.0).iloc[0] == pytest.approx(0.0)

    def test_default_reference(self):
        assert d.DEFAULT_SOLAR_REFERENCE_HOURS == 3.0
        # Default kwarg lands on 3.0
        assert d.solar_deficit(pd.Series([1.5])).iloc[0] == pytest.approx(0.5)

    def test_numpy_array(self):
        arr = np.array([0.0, 1.5, 3.0, 6.0])
        out = d.solar_deficit(arr, reference=3.0)
        np.testing.assert_allclose(out, [1.0, 0.5, 0.0, 0.0])

    def test_pandas_and_numpy_match(self):
        values = [0.0, 0.75, 1.5, 2.25, 3.0, 4.5]
        ref = 3.0
        np_out = d.solar_deficit(np.array(values), reference=ref)
        pd_out = d.solar_deficit(pd.Series(values), reference=ref)
        np.testing.assert_allclose(np_out, pd_out.values)


class TestVentilationDeficit:
    def test_full_svf_full_porosity_zero_deficit(self):
        assert d.ventilation_deficit(pd.Series([1.0]), pd.Series([1.0])).iloc[0] == pytest.approx(
            0.0
        )

    def test_zero_svf_zero_porosity_full_deficit(self):
        assert d.ventilation_deficit(pd.Series([0.0]), pd.Series([0.0])).iloc[0] == pytest.approx(
            1.0
        )

    def test_average_inputs(self):
        # svf=0.4, por=0.6 → score=0.5 → deficit=0.5
        assert d.ventilation_deficit(pd.Series([0.4]), pd.Series([0.6])).iloc[0] == pytest.approx(
            0.5
        )

    def test_numpy_array(self):
        svf = np.array([0.0, 0.5, 1.0])
        por = np.array([0.0, 0.5, 1.0])
        out = d.ventilation_deficit(svf, por)
        np.testing.assert_allclose(out, [1.0, 0.5, 0.0])

    def test_pandas_and_numpy_match(self):
        svf = [0.1, 0.3, 0.5, 0.7, 0.9]
        por = [0.2, 0.4, 0.6, 0.8, 1.0]
        np_out = d.ventilation_deficit(np.array(svf), np.array(por))
        pd_out = d.ventilation_deficit(pd.Series(svf), pd.Series(por))
        np.testing.assert_allclose(np_out, pd_out.values)


class TestHotspotIndex:
    def test_all_zero(self):
        out = d.hotspot_index(pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.0]))
        assert out.iloc[0] == pytest.approx(0.0)

    def test_all_one(self):
        out = d.hotspot_index(pd.Series([1.0]), pd.Series([1.0]), pd.Series([1.0]))
        assert out.iloc[0] == pytest.approx(1.0)

    def test_equal_weighting(self):
        # 0.6 + 0.3 + 0.0 = 0.9 → /3 = 0.3
        out = d.hotspot_index(pd.Series([0.6]), pd.Series([0.3]), pd.Series([0.0]))
        assert out.iloc[0] == pytest.approx(0.3)

    def test_numpy_array(self):
        out = d.hotspot_index(
            np.array([0.0, 0.6, 1.0]),
            np.array([0.0, 0.3, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(out, [0.0, 0.3, 1.0])

    def test_pandas_and_numpy_match(self):
        s = [0.1, 0.3, 0.7, 0.9]
        v = [0.2, 0.4, 0.6, 0.8]
        o = [0.05, 0.15, 0.5, 1.0]
        np_out = d.hotspot_index(np.array(s), np.array(v), np.array(o))
        pd_out = d.hotspot_index(pd.Series(s), pd.Series(v), pd.Series(o))
        np.testing.assert_allclose(np_out, pd_out.values)
