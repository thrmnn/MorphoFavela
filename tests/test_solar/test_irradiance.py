"""Tests for src/solar/irradiance.py — clear-sky irradiance models."""

import numpy as np
import pytest

from src.solar.irradiance import (
    direct_normal_irradiance,
    direct_horizontal_irradiance,
    diffuse_horizontal_irradiance,
    compute_instantaneous_ghi,
    integrate_daily_irradiance,
    compute_daily_irradiance_array,
)


# ===================================================================
# Tests: direct_normal_irradiance (Hottel 1976)
# ===================================================================


class TestDirectNormalIrradiance:
    """Test clear-sky DNI using Hottel transmittance model."""

    def test_zero_altitude_zero_dni(self):
        """Sun at/below horizon -> DNI = 0."""
        assert direct_normal_irradiance(0.0, 172) == 0.0
        assert direct_normal_irradiance(-5.0, 172) == 0.0

    def test_very_low_altitude_low_dni(self):
        """Sun barely above horizon -> DNI very low (massive air mass).

        The Hottel (1976) model still yields ~170 W/m^2 at 0.1 deg due
        to its empirical transmittance floor; this is much lower than
        mid-altitude values (>700 W/m^2).
        """
        assert direct_normal_irradiance(0.1, 172) < 300

    def test_high_altitude_high_dni(self):
        """Sun at 60 deg -> DNI should be 700-1000 W/m^2."""
        dni = direct_normal_irradiance(60.0, 172)
        assert 700 < dni < 1000

    def test_monotonic_with_altitude(self):
        """Higher sun altitude -> higher DNI (less atmosphere)."""
        vals = [direct_normal_irradiance(a, 172) for a in [10, 30, 50, 70]]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_positive_for_high_sun(self):
        """Sun well above horizon should always give positive DNI."""
        for alt in [20, 40, 60, 80]:
            assert direct_normal_irradiance(alt, 172) > 0

    def test_reasonable_magnitude_at_zenith(self):
        """At 90 deg altitude, DNI should be near max (~900-1100 W/m^2)."""
        dni = direct_normal_irradiance(90.0, 172)
        assert 800 < dni < 1200


# ===================================================================
# Tests: direct_horizontal_irradiance
# ===================================================================


class TestDirectHorizontalIrradiance:
    """Test direct irradiance on a horizontal surface (DHI = DNI * sin(alt))."""

    def test_zero_altitude(self):
        assert direct_horizontal_irradiance(500.0, 0.0) == 0.0

    def test_zenith(self):
        """At 90 deg altitude, DHI == DNI (sin(90) = 1)."""
        assert direct_horizontal_irradiance(800.0, 90.0) == pytest.approx(800.0, rel=1e-6)

    def test_45_degrees(self):
        """At 45 deg, DHI = DNI * sin(45) = DNI * sqrt(2)/2."""
        expected = 800.0 * np.sin(np.radians(45))
        assert direct_horizontal_irradiance(800.0, 45.0) == pytest.approx(expected, rel=1e-6)

    def test_negative_dni(self):
        """Negative DNI should return 0."""
        assert direct_horizontal_irradiance(-10.0, 45.0) == 0.0


# ===================================================================
# Tests: diffuse_horizontal_irradiance
# ===================================================================


class TestDiffuseIrradiance:
    """Test clear-sky diffuse irradiance model."""

    def test_positive(self):
        assert diffuse_horizontal_irradiance(45.0, 172) > 0

    def test_zero_at_horizon(self):
        assert diffuse_horizontal_irradiance(0.0, 172) == 0.0

    def test_increases_with_altitude(self):
        """Higher sun -> more scattering energy -> higher diffuse (generally)."""
        vals = [diffuse_horizontal_irradiance(a, 172) for a in [10, 30, 50, 70]]
        # Diffuse should generally increase with altitude
        assert vals[-1] > vals[0]

    def test_reasonable_magnitude(self):
        """Diffuse should be ~50-200 W/m^2 for moderate altitudes."""
        d = diffuse_horizontal_irradiance(45.0, 172)
        assert 30 < d < 300


# ===================================================================
# Tests: compute_instantaneous_ghi
# ===================================================================


class TestInstantaneousGHI:
    """Test instantaneous Global Horizontal Irradiance computation."""

    def test_sunlit_higher_than_shaded(self):
        ghi_sun = compute_instantaneous_ghi(45.0, 172, svf=1.0, is_sunlit=True)
        ghi_shade = compute_instantaneous_ghi(45.0, 172, svf=1.0, is_sunlit=False)
        assert ghi_sun > ghi_shade

    def test_svf_scales_diffuse(self):
        """Lower SVF -> less diffuse irradiance (shaded point)."""
        ghi_open = compute_instantaneous_ghi(45.0, 172, svf=1.0, is_sunlit=False)
        ghi_canyon = compute_instantaneous_ghi(45.0, 172, svf=0.3, is_sunlit=False)
        assert ghi_open > ghi_canyon
        # Shaded GHI is purely diffuse, so scaling should be proportional to SVF
        assert ghi_canyon == pytest.approx(ghi_open * 0.3, rel=0.05)

    def test_zero_svf_shaded_zero(self):
        """SVF=0 and shaded -> no irradiance at all."""
        assert compute_instantaneous_ghi(45.0, 172, svf=0.0, is_sunlit=False) == pytest.approx(0.0)

    def test_zero_altitude_zero_ghi(self):
        """Sun at/below horizon -> GHI = 0 regardless of other params."""
        assert compute_instantaneous_ghi(0.0, 172, svf=1.0, is_sunlit=True) == 0.0
        assert compute_instantaneous_ghi(-5.0, 172, svf=1.0, is_sunlit=True) == 0.0

    def test_sunlit_includes_both_components(self):
        """Sunlit GHI = direct + diffuse, so it should exceed diffuse alone."""
        ghi_sunlit = compute_instantaneous_ghi(60.0, 172, svf=1.0, is_sunlit=True)
        diffuse_only = compute_instantaneous_ghi(60.0, 172, svf=1.0, is_sunlit=False)
        assert ghi_sunlit > diffuse_only
        # The difference is the direct component
        direct_component = ghi_sunlit - diffuse_only
        assert direct_component > 100  # should be substantial at 60 deg altitude


# ===================================================================
# Tests: integrate_daily_irradiance
# ===================================================================


class TestIntegrateDailyIrradiance:
    """Test daily irradiance integration for a single point."""

    def test_all_sunlit_positive(self):
        positions = [(30.0, 90.0), (45.0, 135.0), (60.0, 180.0), (45.0, 225.0), (30.0, 270.0)]
        sunlit = np.ones(5, dtype=bool)
        result = integrate_daily_irradiance(positions, sunlit, svf=1.0, interval_minutes=60, day_of_year=172)
        assert result > 0

    def test_all_shaded_only_diffuse(self):
        positions = [(45.0, 180.0)]
        sunlit = np.zeros(1, dtype=bool)
        result = integrate_daily_irradiance(positions, sunlit, svf=1.0, interval_minutes=60, day_of_year=172)
        # Should still be positive (diffuse component with SVF=1)
        assert result > 0

    def test_zero_svf_shaded_zero_irradiance(self):
        positions = [(45.0, 180.0)]
        sunlit = np.zeros(1, dtype=bool)
        result = integrate_daily_irradiance(positions, sunlit, svf=0.0, interval_minutes=60, day_of_year=172)
        assert result == pytest.approx(0.0)

    def test_sunlit_greater_than_shaded(self):
        positions = [(45.0, 180.0), (60.0, 180.0)]
        sunlit_yes = np.ones(2, dtype=bool)
        sunlit_no = np.zeros(2, dtype=bool)
        result_sun = integrate_daily_irradiance(positions, sunlit_yes, svf=1.0, interval_minutes=60, day_of_year=172)
        result_shade = integrate_daily_irradiance(positions, sunlit_no, svf=1.0, interval_minutes=60, day_of_year=172)
        assert result_sun > result_shade


# ===================================================================
# Tests: compute_daily_irradiance_array (vectorized)
# ===================================================================


class TestDailyIrradianceArray:
    """Test vectorized daily irradiance computation for N points."""

    def test_open_sky_positive(self):
        positions = [(45.0, 180.0), (60.0, 180.0)]
        sunlit = np.ones((3, 2), dtype=bool)
        svf = np.ones(3)
        result = compute_daily_irradiance_array(positions, sunlit, svf, 60, 172)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_fully_shaded_only_diffuse(self):
        positions = [(45.0, 180.0)]
        sunlit = np.zeros((2, 1), dtype=bool)
        svf = np.array([1.0, 0.0])
        result = compute_daily_irradiance_array(positions, sunlit, svf, 60, 172)
        assert result[0] > 0   # diffuse only, but SVF=1
        assert result[1] == pytest.approx(0.0)  # SVF=0, no diffuse either

    def test_shape_matches_n_points(self):
        positions = [(30.0, 90.0), (45.0, 180.0), (60.0, 270.0)]
        n_pts = 5
        sunlit = np.ones((n_pts, 3), dtype=bool)
        svf = np.full(n_pts, 0.8)
        result = compute_daily_irradiance_array(positions, sunlit, svf, 30, 172)
        assert result.shape == (n_pts,)

    def test_higher_svf_more_irradiance(self):
        """Points with higher SVF should get more total irradiance (same sunlit)."""
        positions = [(45.0, 180.0), (60.0, 180.0)]
        sunlit = np.zeros((2, 2), dtype=bool)  # all shaded -> diffuse only
        svf = np.array([1.0, 0.3])
        result = compute_daily_irradiance_array(positions, sunlit, svf, 60, 172)
        assert result[0] > result[1]

    def test_consistent_with_scalar_integration(self):
        """Vectorized result should match scalar integration for single point."""
        positions = [(45.0, 180.0), (60.0, 200.0)]
        sunlit_vec = np.array([[True, False]])
        svf_vec = np.array([0.7])
        result_vec = compute_daily_irradiance_array(positions, sunlit_vec, svf_vec, 60, 172)

        sunlit_scalar = np.array([True, False])
        result_scalar = integrate_daily_irradiance(positions, sunlit_scalar, svf=0.7, interval_minutes=60, day_of_year=172)

        assert result_vec[0] == pytest.approx(result_scalar, rel=0.01)
