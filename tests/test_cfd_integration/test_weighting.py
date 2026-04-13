"""Tests for wind-rose weighting across 8 cardinal directions."""

import json

import numpy as np
import pytest

from src.cfd_integration.schema import WindRose
from src.cfd_integration.weighting import (
    load_wind_rose,
    weighted_by_wind_rose,
    worst_case_direction,
)


@pytest.fixture
def per_direction_metrics():
    """Mock per-direction results for one patch."""
    return {
        "N":  {"U_mean": 1.0, "stagnation_frac": 0.6},
        "NE": {"U_mean": 1.5, "stagnation_frac": 0.4},
        "E":  {"U_mean": 3.0, "stagnation_frac": 0.1},
        "SE": {"U_mean": 4.0, "stagnation_frac": 0.05},
        "S":  {"U_mean": 2.0, "stagnation_frac": 0.3},
    }


class TestWeightedByWindRose:
    def test_uniform_default(self, per_direction_metrics):
        """With no wind rose, uniform weighting."""
        result = weighted_by_wind_rose(per_direction_metrics, wind_rose=None)
        # Uniform mean of U_mean = (1 + 1.5 + 3 + 4 + 2) / 5 = 2.3
        assert result["annual_U_mean"] == pytest.approx(2.3)

    def test_frequency_weighting(self, per_direction_metrics):
        """Frequency weighting shifts the average toward high-frequency directions."""
        wr = WindRose(
            site="x",
            frequencies={
                "N": 0.05, "NE": 0.05, "E": 0.10, "SE": 0.60, "S": 0.20,
            },
        )
        result = weighted_by_wind_rose(per_direction_metrics, wr, weight_by="frequency")
        # SE dominates (60%) with U_mean=4, so weighted mean should be high
        manual = (0.05 * 1.0 + 0.05 * 1.5 + 0.10 * 3.0
                  + 0.60 * 4.0 + 0.20 * 2.0) / (0.05 + 0.05 + 0.10 + 0.60 + 0.20)
        assert result["annual_U_mean"] == pytest.approx(manual)

    def test_freq_speed_weighting_differs(self, per_direction_metrics):
        """freq × speed weighting should differ from frequency-only."""
        wr = WindRose(
            site="x",
            frequencies={"N": 0.2, "NE": 0.2, "E": 0.2, "SE": 0.2, "S": 0.2},
            mean_speeds={"N": 1.0, "NE": 1.5, "E": 3.0, "SE": 4.0, "S": 2.0},
        )
        f_only = weighted_by_wind_rose(per_direction_metrics, wr, weight_by="frequency")
        f_x_s = weighted_by_wind_rose(per_direction_metrics, wr, weight_by="freq_speed")
        # freq×speed weighting emphasises stronger directions → higher mean U
        assert f_x_s["annual_U_mean"] > f_only["annual_U_mean"]

    def test_returns_weight_method(self, per_direction_metrics):
        result = weighted_by_wind_rose(per_direction_metrics, None, weight_by="uniform")
        assert result["weight_method"] == "uniform"

    def test_empty_input(self):
        assert weighted_by_wind_rose({}, None) == {}


class TestWorstCase:
    def test_worst_case_stagnation(self, per_direction_metrics):
        d, v = worst_case_direction(per_direction_metrics, metric="stagnation_frac")
        assert d == "N"
        assert v == 0.6

    def test_empty(self):
        d, v = worst_case_direction({}, metric="stagnation_frac")
        assert d == ""
        assert np.isnan(v)


class TestLoadWindRose:
    def test_roundtrip(self, tmp_path, synthetic_wind_rose):
        path = tmp_path / "wind_rose.json"
        with open(path, "w") as f:
            json.dump(synthetic_wind_rose, f)

        wr = load_wind_rose(path, site="test_site")
        assert wr.site == "test_site"
        assert wr.frequencies["SE"] == pytest.approx(0.30)
        assert wr.mean_speeds["SE"] == pytest.approx(4.0)
