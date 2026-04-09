"""Tests for src/solar/seasonal.py — multi-day seasonal solar analysis."""

import numpy as np

from src.solar.seasonal import compute_seasonal_solar
from src.solar.sun import REFERENCE_DATES


# ===================================================================
# Tests: compute_seasonal_solar
# ===================================================================


class TestSeasonalSolar:
    """Test seasonal solar access computation across multiple reference dates."""

    def test_returns_expected_structure(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        assert "dates" in result
        assert "summary" in result
        assert len(result["dates"]) == 4  # 4 reference dates

    def test_all_dates_present(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        expected_dates = set(REFERENCE_DATES.values())
        assert set(result["dates"].keys()) == expected_dates

    def test_summer_more_hours_than_winter(self, flat_ground_mesh):
        """Summer solstice should yield more solar hours than winter (open sky)."""
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        winter = result["dates"][REFERENCE_DATES["winter_solstice"]]["solar_hours"][0]
        summer = result["dates"][REFERENCE_DATES["summer_solstice"]]["solar_hours"][0]
        assert summer > winter

    def test_summary_statistics(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        s = result["summary"]
        assert "solar_hours_mean" in s
        assert "seasonal_range" in s
        assert s["seasonal_range"][0] > 0  # summer != winter

    def test_summary_shape(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5], [3.0, 3.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        s = result["summary"]
        assert s["solar_hours_mean"].shape == (2,)
        assert s["solar_hours_min"].shape == (2,)
        assert s["solar_hours_max"].shape == (2,)
        assert s["seasonal_range"].shape == (2,)

    def test_custom_dates(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh, dates=["2026-06-21"])
        assert len(result["dates"]) == 1
        assert "2026-06-21" in result["dates"]

    def test_per_date_keys(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh, dates=["2026-06-21"])
        date_entry = result["dates"]["2026-06-21"]
        assert "solar_hours" in date_entry
        assert "sunlit_matrix" in date_entry
        assert "sun_positions" in date_entry

    def test_solar_hours_nonnegative(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        for date_str, entry in result["dates"].items():
            assert np.all(entry["solar_hours"] >= 0), f"Negative hours on {date_str}"

    def test_with_svf_array(self, flat_ground_mesh):
        """When SVF is provided, irradiance should be included."""
        obs = np.array([[5.0, 5.0, 1.5]])
        svf = np.array([0.8])
        result = compute_seasonal_solar(
            obs, flat_ground_mesh, svf_array=svf, dates=["2026-06-21"]
        )
        entry = result["dates"]["2026-06-21"]
        assert "irradiance_wh" in entry
        assert entry["irradiance_wh"][0] > 0

    def test_shadow_frequency_in_summary(self, flat_ground_mesh):
        obs = np.array([[5.0, 5.0, 1.5]])
        result = compute_seasonal_solar(obs, flat_ground_mesh)
        s = result["summary"]
        assert "shadow_frequency" in s
        # Open sky: shadow frequency should be low (near 0)
        assert s["shadow_frequency"][0] < 0.1
