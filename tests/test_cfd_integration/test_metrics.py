"""Tests for health-relevant wind metrics."""

import numpy as np
import pytest

from src.cfd_integration.metrics import (
    ach,
    canyon_ventilation_efficiency,
    low_wind_percentile,
    stagnation_fraction,
    turbulent_intensity,
    velocity_magnitude,
)


class TestVelocityMagnitude:
    def test_basic(self):
        u = np.array([3.0])
        v = np.array([4.0])
        w = np.array([0.0])
        assert velocity_magnitude(u, v, w)[0] == pytest.approx(5.0)

    def test_zero(self):
        z = np.zeros(3)
        assert np.all(velocity_magnitude(z, z, z) == 0)


class TestStagnationFraction:
    def test_all_stagnant(self):
        u = np.array([0.1, 0.2, 0.3])
        assert stagnation_fraction(u, threshold=0.5) == 1.0

    def test_none_stagnant(self):
        u = np.array([1.0, 2.0, 3.0])
        assert stagnation_fraction(u, threshold=0.5) == 0.0

    def test_half(self):
        u = np.array([0.1, 0.2, 1.0, 2.0])
        assert stagnation_fraction(u, threshold=0.5) == 0.5

    def test_empty(self):
        assert np.isnan(stagnation_fraction(np.array([])))


class TestTurbulentIntensity:
    def test_basic(self):
        # TI = sqrt(2/3 × k) / U_ref
        # For k=1.5, U=1: TI = sqrt(1) = 1.0
        ti = turbulent_intensity(np.array([1.5]), u_ref=1.0)
        assert ti[0] == pytest.approx(1.0)

    def test_zero_uref(self):
        ti = turbulent_intensity(np.array([1.0]), u_ref=0.0)
        assert np.isnan(ti[0])


class TestACH:
    def test_standard_formulation(self):
        """ACH = 3600 × U / L for square cell."""
        # U=1 m/s, L=10m cell → ACH = 360/h
        assert ach(np.array([1.0]), canopy_height=10.0, cell_area=100.0) == pytest.approx(360.0)

    def test_scales_with_velocity(self):
        # Doubling U doubles ACH
        base = ach(np.array([1.0]), 10.0, 100.0)
        doubled = ach(np.array([2.0]), 10.0, 100.0)
        assert doubled == pytest.approx(2 * base)

    def test_scales_inversely_with_length(self):
        # 20m cell (4× area) → half ACH
        small = ach(np.array([1.0]), 10.0, 100.0)  # L=10
        large = ach(np.array([1.0]), 10.0, 400.0)  # L=20
        assert large == pytest.approx(small / 2)

    def test_uses_mean_velocity(self):
        # Multiple samples → uses mean
        uniform = ach(np.array([2.0]), 10.0, 100.0)
        mixed = ach(np.array([1.0, 3.0]), 10.0, 100.0)  # mean = 2
        assert mixed == pytest.approx(uniform)

    def test_empty(self):
        assert np.isnan(ach(np.array([]), 10.0, 100.0))

    def test_zero_canopy_height(self):
        assert np.isnan(ach(np.array([1.0]), 0.0, 100.0))


class TestLowWindPercentile:
    def test_p10(self):
        u = np.arange(100, dtype=float)  # 0..99
        # 10th percentile of 0..99
        assert low_wind_percentile(u, pct=10) == pytest.approx(9.9, abs=0.1)

    def test_empty(self):
        assert np.isnan(low_wind_percentile(np.array([])))


class TestCanyonVentilationEfficiency:
    def test_basic(self):
        # U_canyon = 1 m/s, U_ref = 4 m/s → efficiency 0.25
        eff = canyon_ventilation_efficiency(np.array([1.0]), u_ref=4.0)
        assert eff == pytest.approx(0.25)

    def test_poor_threshold(self):
        # efficiency < 0.2 is "poorly ventilated"
        eff = canyon_ventilation_efficiency(np.array([0.3]), u_ref=2.0)
        assert eff == pytest.approx(0.15)
        assert eff < 0.2
