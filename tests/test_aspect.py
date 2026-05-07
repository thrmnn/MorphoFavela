"""Tests for terrain-aspect helpers (sin/cos encoding, wind alignment, quadrants)."""

from __future__ import annotations

import numpy as np
import pytest

from src.morphometry.aspect import (
    aspect_quadrant,
    aspect_to_sincos,
    aspect_wind_alignment,
    dominant_wind_direction_deg,
)


class TestAspectToSincos:
    def test_cardinal_directions(self):
        # 0=N, 90=E, 180=S, 270=W. sin(N)=0, cos(N)=1, sin(E)=1, cos(E)=0, ...
        s, c = aspect_to_sincos(np.array([0.0, 90.0, 180.0, 270.0]))
        np.testing.assert_allclose(s, [0.0, 1.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(c, [1.0, 0.0, -1.0, 0.0], atol=1e-12)

    def test_unit_norm(self):
        # sin² + cos² = 1 for any aspect, including non-cardinal ones
        a = np.linspace(0, 720, 50)
        s, c = aspect_to_sincos(a)
        np.testing.assert_allclose(s**2 + c**2, np.ones_like(a), atol=1e-12)

    def test_nan_propagates(self):
        s, c = aspect_to_sincos(np.array([0.0, np.nan, 180.0]))
        assert np.isnan(s[1]) and np.isnan(c[1])
        assert not np.isnan(s[0]) and not np.isnan(c[0])

    def test_scalar_input(self):
        s, c = aspect_to_sincos(45.0)
        np.testing.assert_allclose(s, np.sin(np.pi / 4))
        np.testing.assert_allclose(c, np.cos(np.pi / 4))


class TestAspectWindAlignment:
    def test_aligned(self):
        # South-facing slope (180), wind FROM south (180): alignment = +1 (windward)
        a = aspect_wind_alignment(180.0, 180.0)
        np.testing.assert_allclose(a, 1.0, atol=1e-12)

    def test_anti_aligned(self):
        # South-facing slope, wind from north (0): -1 (lee)
        a = aspect_wind_alignment(180.0, 0.0)
        np.testing.assert_allclose(a, -1.0, atol=1e-12)

    def test_perpendicular(self):
        # North-facing slope (0), wind from east (90): 0
        a = aspect_wind_alignment(0.0, 90.0)
        np.testing.assert_allclose(a, 0.0, atol=1e-12)

    def test_circular_wraparound(self):
        # 359 vs 1 (both ~north-facing) should be ~aligned, not anti-aligned
        a = aspect_wind_alignment(np.array([359.0, 1.0]), 0.0)
        np.testing.assert_allclose(a, [np.cos(np.deg2rad(1)), np.cos(np.deg2rad(1))], atol=1e-12)
        assert (a > 0.99).all()

    def test_array_input(self):
        a = aspect_wind_alignment(np.array([0.0, 90.0, 180.0, 270.0]), 90.0)
        np.testing.assert_allclose(a, [0.0, 1.0, 0.0, -1.0], atol=1e-12)


class TestAspectQuadrant:
    def test_cardinal_centres(self):
        q = aspect_quadrant(np.array([0.0, 90.0, 180.0, 270.0]))
        assert list(q) == ["N", "E", "S", "W"]

    def test_edge_inclusive_lower(self):
        # 45 is the N/E edge; consumers expect 45 -> E (the canonical
        # right-inclusive convention for compass quadrants).
        q = aspect_quadrant(np.array([45.0, 135.0, 225.0, 315.0]))
        assert list(q) == ["E", "S", "W", "N"]

    def test_n_wraps_zero(self):
        # 350, 10, 359 are all N-facing
        q = aspect_quadrant(np.array([350.0, 10.0, 359.0]))
        assert list(q) == ["N", "N", "N"]

    def test_modular_input(self):
        # 360, 720 should be treated as 0 (N)
        q = aspect_quadrant(np.array([360.0, 720.0]))
        assert list(q) == ["N", "N"]

    def test_nan_returns_empty(self):
        q = aspect_quadrant(np.array([np.nan, 90.0]))
        assert q[0] == ""
        assert q[1] == "E"


class TestDominantWindDirection:
    def test_single_bin(self):
        wind_rose = type("R", (), {"frequencies": {"E": 1.0}})()
        d = dominant_wind_direction_deg(wind_rose)
        np.testing.assert_allclose(d, 90.0)

    def test_circular_mean_two_bins(self):
        # Equal NW and NE -> mean is N (not 180°/S, which is what arithmetic
        # mean of 315 and 45 would give).
        wind_rose = type("R", (), {"frequencies": {"NW": 0.5, "NE": 0.5}})()
        d = dominant_wind_direction_deg(wind_rose)
        np.testing.assert_allclose(d % 360, 0.0, atol=1e-9)

    def test_empty_rose(self):
        wind_rose = type("R", (), {"frequencies": {}})()
        assert dominant_wind_direction_deg(wind_rose) is None

    def test_zero_rose(self):
        wind_rose = type("R", (), {"frequencies": {d: 0.0 for d in ("N", "E", "S", "W")}})()
        assert dominant_wind_direction_deg(wind_rose) is None


def test_alignment_matches_sincos_decomposition():
    """alignment(a, w) should equal sin(a)·sin(w) + cos(a)·cos(w)."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 360, 100)
    w = 67.0
    sa, ca = aspect_to_sincos(a)
    sw, cw = aspect_to_sincos(w)
    expected = sa * sw + ca * cw
    actual = aspect_wind_alignment(a, w)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
