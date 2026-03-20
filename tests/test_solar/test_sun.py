"""Tests for src/solar/sun.py — sun position computation."""

import math

import numpy as np
import pytest

from src.solar.sun import (
    compute_sun_positions,
    sun_position_to_direction,
    compute_sun_positions_multi_day,
    compute_extraterrestrial_irradiance,
    REFERENCE_DATES,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
)


# ===================================================================
# Tests: compute_sun_positions
# ===================================================================


class TestComputeSunPositions:
    """Tests for compute_sun_positions (expanded from test_solar_access.py)."""

    def test_returns_list_of_tuples(self):
        positions = compute_sun_positions(
            latitude=-22.97, longitude=-43.17, date="2026-06-21"
        )
        assert isinstance(positions, list)
        assert len(positions) > 0
        for item in positions:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_all_altitudes_positive(self):
        """All returned positions must have altitude > 0 (sun above horizon)."""
        positions = compute_sun_positions(
            latitude=-22.97, longitude=-43.17, date="2026-06-21"
        )
        for alt, az in positions:
            assert alt > 0, f"Altitude {alt} should be > 0"

    def test_azimuths_in_range(self):
        """Azimuths should be in [0, 360)."""
        positions = compute_sun_positions(
            latitude=-22.97, longitude=-43.17, date="2026-06-21"
        )
        for alt, az in positions:
            assert 0 <= az < 360, f"Azimuth {az} out of range"

    def test_winter_solstice_rio_limited_hours(self):
        """Rio winter solstice: roughly 10-11 hours of daylight."""
        positions = compute_sun_positions(
            latitude=-22.97,
            longitude=-43.17,
            date="2026-06-21",
            hour_start=5,
            hour_end=19,
            interval_minutes=60,
        )
        # Winter solstice at lat -23: sunrise ~06:30, sunset ~17:15 local
        assert 7 <= len(positions) <= 13, f"Expected 7-13 positions, got {len(positions)}"

    def test_max_altitude_winter_solstice_rio(self):
        """At lat -23 winter solstice, max altitude ~ 43-44 degrees.

        Theoretical: 90 - 23.45 - 22.97 ~ 43.6 deg.
        """
        positions = compute_sun_positions(
            latitude=-22.97,
            longitude=-43.17,
            date="2026-06-21",
            hour_start=5,
            hour_end=19,
            interval_minutes=15,
        )
        max_alt = max(alt for alt, _ in positions)
        assert 35 <= max_alt <= 50, f"Max altitude {max_alt:.1f} out of expected range"

    def test_summer_more_than_winter(self):
        """Summer solstice should have more sun positions than winter."""
        pos_winter = compute_sun_positions(
            latitude=-22.97, longitude=-43.17, date="2026-06-21",
            hour_start=5, hour_end=19, interval_minutes=60,
        )
        pos_summer = compute_sun_positions(
            latitude=-22.97, longitude=-43.17, date="2026-12-21",
            hour_start=5, hour_end=19, interval_minutes=60,
        )
        assert len(pos_summer) > len(pos_winter)

    def test_finer_interval_more_positions(self):
        """Smaller interval should yield more positions."""
        pos_60 = compute_sun_positions(interval_minutes=60)
        pos_30 = compute_sun_positions(interval_minutes=30)
        assert len(pos_30) >= len(pos_60)

    def test_equator_equinox_high_sun(self):
        """Equator on equinox: sun nearly overhead, ~12h daylight."""
        positions = compute_sun_positions(
            latitude=0.0, longitude=0.0, date="2026-03-20",
            hour_start=4, hour_end=20, interval_minutes=60,
        )
        assert len(positions) >= 10
        max_alt = max(alt for alt, _ in positions)
        assert max_alt > 80  # near-zenith


# ===================================================================
# Tests: sun_position_to_direction
# ===================================================================


class TestSunPositionToDirection:
    """Test altitude/azimuth to direction vector conversion."""

    def test_straight_up(self):
        """altitude=90, any azimuth -> (0, 0, 1)."""
        d = sun_position_to_direction(90.0, 0.0)
        np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=1e-10)

    def test_horizontal_cardinal(self):
        """Cardinal directions at the horizon.

        az=0   -> north (0, 1, 0)
        az=90  -> east  (1, 0, 0)
        az=180 -> south (0, -1, 0)
        az=270 -> west  (-1, 0, 0)
        """
        np.testing.assert_allclose(
            sun_position_to_direction(0.0, 0.0), [0.0, 1.0, 0.0], atol=1e-10
        )
        np.testing.assert_allclose(
            sun_position_to_direction(0.0, 90.0), [1.0, 0.0, 0.0], atol=1e-10
        )
        np.testing.assert_allclose(
            sun_position_to_direction(0.0, 180.0), [0.0, -1.0, 0.0], atol=1e-10
        )
        np.testing.assert_allclose(
            sun_position_to_direction(0.0, 270.0), [-1.0, 0.0, 0.0], atol=1e-10
        )

    def test_unit_length(self):
        """Result must always be a unit vector."""
        for alt in [0, 15, 30, 45, 60, 90]:
            for az in [0, 45, 90, 135, 180, 225, 270, 315]:
                d = sun_position_to_direction(float(alt), float(az))
                np.testing.assert_allclose(
                    np.linalg.norm(d), 1.0, atol=1e-12,
                    err_msg=f"Non-unit at alt={alt}, az={az}",
                )

    def test_45_degree_altitude(self):
        """altitude=45, azimuth=0 -> z = sin(45) = sqrt(2)/2."""
        d = sun_position_to_direction(45.0, 0.0)
        expected_z = math.sqrt(2) / 2
        assert abs(d[2] - expected_z) < 1e-10

    def test_straight_up_any_azimuth(self):
        """altitude=90 with different azimuths still points up."""
        for az in [0, 45, 90, 180, 270]:
            d = sun_position_to_direction(90.0, float(az))
            np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=1e-10)


# ===================================================================
# Tests: compute_sun_positions_multi_day
# ===================================================================


class TestMultiDay:
    """Test multi-day sun position computation."""

    def test_returns_dict(self):
        result = compute_sun_positions_multi_day()
        assert isinstance(result, dict)
        assert len(result) == 4  # 4 reference dates

    def test_summer_longer_than_winter(self):
        result = compute_sun_positions_multi_day()
        n_winter = len(result[REFERENCE_DATES["winter_solstice"]])
        n_summer = len(result[REFERENCE_DATES["summer_solstice"]])
        assert n_summer > n_winter

    def test_custom_dates(self):
        result = compute_sun_positions_multi_day(dates=["2026-06-21"])
        assert len(result) == 1
        assert "2026-06-21" in result

    def test_all_dates_have_positions(self):
        result = compute_sun_positions_multi_day()
        for date_str, positions in result.items():
            assert len(positions) > 0, f"No positions for {date_str}"

    def test_keys_match_dates(self):
        result = compute_sun_positions_multi_day()
        expected_dates = set(REFERENCE_DATES.values())
        assert set(result.keys()) == expected_dates


# ===================================================================
# Tests: compute_extraterrestrial_irradiance
# ===================================================================


class TestExtraterrestrialIrradiance:
    """Test extra-terrestrial irradiance computation (Spencer 1971)."""

    def test_perihelion_higher(self):
        """Day ~3 (perihelion) should have higher I than day ~186 (aphelion)."""
        i_peri = compute_extraterrestrial_irradiance(3)
        i_aph = compute_extraterrestrial_irradiance(186)
        assert i_peri > i_aph

    def test_range(self):
        """Should be within 1300-1420 W/m^2 throughout the year."""
        for doy in [1, 91, 182, 274]:
            i = compute_extraterrestrial_irradiance(doy)
            assert 1300 < i < 1420, f"Day {doy}: {i:.1f} W/m^2 out of range"

    def test_symmetry_around_perihelion(self):
        """Days equidistant from perihelion should have similar irradiance."""
        i_before = compute_extraterrestrial_irradiance(350)  # ~Dec 16
        i_after = compute_extraterrestrial_irradiance(20)    # ~Jan 20
        assert abs(i_before - i_after) / i_before < 0.02  # within 2%
