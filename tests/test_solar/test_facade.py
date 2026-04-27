"""Tests for src/solar/facade.py — facade solar irradiance pipeline."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from src.solar.facade import (
    aggregate_by_building,
    aggregate_by_building_floor,
    assess_who_threshold,
    compute_facade_daily_irradiance_array,
    compute_facade_incident_angle,
    compute_facade_sunlit_hours,
    compute_floor_level,
    diffuse_on_tilted_surface,
    direct_beam_on_facade,
)

# ---------------------------------------------------------------------------
# Section A: Incident angle geometry
# ---------------------------------------------------------------------------


class TestFacadeIncidentAngle:
    """Verify cos(theta_i) = cos(alt) * cos(az_sun - az_facade)."""

    def test_sun_directly_in_front(self):
        """Sun azimuth matches facade normal: cos(theta) = cos(alt)."""
        cos_theta = compute_facade_incident_angle(
            sun_altitude_deg=45.0,
            sun_azimuth_deg=180.0,
            facade_azimuth_deg=180.0,
        )
        assert cos_theta == pytest.approx(np.cos(np.radians(45.0)), abs=1e-10)

    def test_sun_behind_facade(self):
        """Sun 180 deg opposite facade normal: cos(theta) < 0."""
        cos_theta = compute_facade_incident_angle(
            sun_altitude_deg=45.0,
            sun_azimuth_deg=0.0,
            facade_azimuth_deg=180.0,
        )
        assert cos_theta < 0

    def test_sun_perpendicular(self):
        """Sun 90 deg from facade normal: cos(theta) ~ 0."""
        cos_theta = compute_facade_incident_angle(
            sun_altitude_deg=45.0,
            sun_azimuth_deg=90.0,
            facade_azimuth_deg=180.0,
        )
        assert cos_theta == pytest.approx(0.0, abs=1e-10)

    def test_sun_at_zenith(self):
        """Sun at 90 deg altitude: cos(theta) = 0 for any vertical facade."""
        cos_theta = compute_facade_incident_angle(
            sun_altitude_deg=90.0,
            sun_azimuth_deg=0.0,
            facade_azimuth_deg=0.0,
        )
        assert cos_theta == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """Symmetric offsets from facade normal give same magnitude."""
        cos_left = compute_facade_incident_angle(45.0, 160.0, 180.0)
        cos_right = compute_facade_incident_angle(45.0, 200.0, 180.0)
        assert cos_left == pytest.approx(cos_right, abs=1e-10)

    def test_low_altitude_high_cos(self):
        """Low sun altitude + facing sun: large cos(theta)."""
        cos_theta = compute_facade_incident_angle(
            sun_altitude_deg=10.0,
            sun_azimuth_deg=0.0,
            facade_azimuth_deg=0.0,
        )
        # cos(10 deg) ~ 0.985
        assert cos_theta > 0.9


class TestDirectBeamOnFacade:
    def test_positive_when_sun_in_front(self):
        result = direct_beam_on_facade(
            dni=800.0,
            sun_altitude_deg=45.0,
            sun_azimuth_deg=180.0,
            facade_azimuth_deg=180.0,
        )
        assert result > 0

    def test_zero_when_sun_behind(self):
        result = direct_beam_on_facade(
            dni=800.0,
            sun_altitude_deg=45.0,
            sun_azimuth_deg=0.0,
            facade_azimuth_deg=180.0,
        )
        assert result == 0.0

    def test_less_than_dni(self):
        """Facade irradiance always <= DNI."""
        result = direct_beam_on_facade(
            dni=1000.0,
            sun_altitude_deg=30.0,
            sun_azimuth_deg=0.0,
            facade_azimuth_deg=0.0,
        )
        assert result <= 1000.0
        assert result > 0

    def test_zero_dni_zero_beam(self):
        result = direct_beam_on_facade(
            dni=0.0,
            sun_altitude_deg=45.0,
            sun_azimuth_deg=0.0,
            facade_azimuth_deg=0.0,
        )
        assert result == 0.0


class TestDiffuseOnTiltedSurface:
    def test_vertical_half_diffuse(self):
        """Vertical surface: factor = 0.5."""
        result = diffuse_on_tilted_surface(100.0, tilt_deg=90.0)
        assert result == pytest.approx(50.0, abs=1e-10)

    def test_horizontal_full_diffuse(self):
        """Horizontal surface: factor = 1.0."""
        result = diffuse_on_tilted_surface(100.0, tilt_deg=0.0)
        assert result == pytest.approx(100.0, abs=1e-10)

    def test_inverted_zero_diffuse(self):
        """Fully inverted (180 deg): factor = 0."""
        result = diffuse_on_tilted_surface(100.0, tilt_deg=180.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_monotonic_decrease(self):
        """Diffuse decreases monotonically with tilt from 0 to 180."""
        tilts = [0, 30, 60, 90, 120, 150, 180]
        values = [diffuse_on_tilted_surface(100.0, t) for t in tilts]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


# ---------------------------------------------------------------------------
# Section C: Facade sunlit hours
# ---------------------------------------------------------------------------


class TestFacadeSunlitHours:
    def test_all_sunlit_facing_sun(self):
        """Facade facing sun, all timesteps unobstructed -> full hours."""
        N, M = 1, 4
        sunlit_matrix = np.ones((N, M), dtype=bool)
        # Sun always at az=0 (north), alt=45 — facade faces north
        sun_positions = [(45.0, 0.0)] * M
        facade_azimuths = np.array([0.0])

        hours = compute_facade_sunlit_hours(
            sunlit_matrix,
            sun_positions,
            facade_azimuths,
            interval_minutes=60,
        )
        assert hours[0] == pytest.approx(4.0)

    def test_sunlit_but_sun_behind(self):
        """Geometrically sunlit but sun behind facade -> zero hours."""
        N, M = 1, 4
        sunlit_matrix = np.ones((N, M), dtype=bool)
        # Sun at az=180 (south), facade faces north (az=0) -> sun is behind
        sun_positions = [(45.0, 180.0)] * M
        facade_azimuths = np.array([0.0])

        hours = compute_facade_sunlit_hours(
            sunlit_matrix,
            sun_positions,
            facade_azimuths,
            interval_minutes=60,
        )
        assert hours[0] == pytest.approx(0.0)

    def test_half_day_east_facade(self):
        """East-facing facade: only morning sun (az < 180) counts."""
        N, M = 1, 4
        sunlit_matrix = np.ones((N, M), dtype=bool)
        # Sun moves from east (90) to south (180) to west (270)
        sun_positions = [(45.0, 90.0), (60.0, 135.0), (45.0, 225.0), (30.0, 270.0)]
        facade_azimuths = np.array([90.0])  # East-facing

        hours = compute_facade_sunlit_hours(
            sunlit_matrix,
            sun_positions,
            facade_azimuths,
            interval_minutes=60,
        )
        # First two timesteps: sun within 90 deg of facade (az=90,135 vs facade=90)
        # Third: az=225 vs 90 -> diff=135 -> cos(135) < 0 -> behind
        # Fourth: az=270 vs 90 -> diff=180 -> behind
        assert hours[0] == pytest.approx(2.0)

    def test_shadowed_gives_zero(self):
        """If sunlit_matrix is False, hours = 0 regardless of orientation."""
        N, M = 1, 4
        sunlit_matrix = np.zeros((N, M), dtype=bool)
        sun_positions = [(45.0, 0.0)] * M
        facade_azimuths = np.array([0.0])

        hours = compute_facade_sunlit_hours(
            sunlit_matrix,
            sun_positions,
            facade_azimuths,
            interval_minutes=60,
        )
        assert hours[0] == 0.0


# ---------------------------------------------------------------------------
# Section D: WHO threshold
# ---------------------------------------------------------------------------


class TestWHOThreshold:
    def test_above_threshold(self):
        result = assess_who_threshold(np.array([3.0]))
        assert result[0] is np.True_

    def test_below_threshold(self):
        result = assess_who_threshold(np.array([1.0]))
        assert result[0] is np.False_

    def test_exactly_threshold(self):
        result = assess_who_threshold(np.array([2.0]))
        assert result[0] is np.True_

    def test_vectorized(self):
        result = assess_who_threshold(np.array([0.5, 2.0, 3.5]))
        np.testing.assert_array_equal(result, [False, True, True])

    def test_custom_threshold(self):
        result = assess_who_threshold(np.array([1.5]), threshold_hours=1.0)
        assert result[0] is np.True_


class TestFloorLevel:
    def test_ground_floor(self):
        result = compute_floor_level(np.array([0.5]), floor_height=3.0)
        assert result[0] == 0

    def test_first_floor(self):
        result = compute_floor_level(np.array([3.5]), floor_height=3.0)
        assert result[0] == 1

    def test_upper_floor(self):
        result = compute_floor_level(np.array([9.0]), floor_height=3.0)
        assert result[0] == 3

    def test_vectorized(self):
        heights = np.array([0.5, 3.5, 6.5, 9.5])
        result = compute_floor_level(heights, floor_height=3.0)
        np.testing.assert_array_equal(result, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Section B: Vectorized daily irradiance
# ---------------------------------------------------------------------------


class TestFacadeDailyIrradianceArray:
    def test_shape_matches(self):
        N, M = 5, 3
        sun_positions = [(45.0, 0.0), (60.0, 30.0), (45.0, 60.0)]
        sunlit_matrix = np.ones((N, M), dtype=bool)
        facade_azimuths = np.zeros(N)
        svf_array = np.ones(N)

        result = compute_facade_daily_irradiance_array(
            sun_positions,
            sunlit_matrix,
            facade_azimuths,
            svf_array,
            interval_minutes=30,
            day_of_year=172,
        )
        assert result.shape == (N,)

    def test_north_vs_south_winter_solstice_rio(self):
        """At Rio (-22.95), June 21 sun transits north. North > South."""
        from src.solar.sun import compute_sun_positions

        sun_positions = compute_sun_positions(
            -22.95,
            -43.23,
            "2026-06-21",
            hour_start=6,
            hour_end=18,
            interval_minutes=30,
        )
        if not sun_positions:
            pytest.skip("No sun positions computed")

        N = 2
        M = len(sun_positions)
        sunlit_matrix = np.ones((N, M), dtype=bool)
        facade_azimuths = np.array([0.0, 180.0])  # North, South
        svf_array = np.ones(N)

        result = compute_facade_daily_irradiance_array(
            sun_positions,
            sunlit_matrix,
            facade_azimuths,
            svf_array,
            interval_minutes=30,
            day_of_year=172,
        )
        # North-facing should get more direct beam than south-facing
        assert result[0] > result[1], (
            f"North-facing ({result[0]:.0f}) should exceed south-facing ({result[1]:.0f}) "
            f"at Rio winter solstice"
        )

    def test_all_shaded_only_diffuse(self):
        """Fully shaded points still receive diffuse irradiance."""
        N, M = 2, 3
        sun_positions = [(45.0, 0.0), (60.0, 30.0), (45.0, 60.0)]
        sunlit_matrix = np.zeros((N, M), dtype=bool)
        facade_azimuths = np.array([0.0, 180.0])
        svf_array = np.ones(N)

        result = compute_facade_daily_irradiance_array(
            sun_positions,
            sunlit_matrix,
            facade_azimuths,
            svf_array,
            interval_minutes=30,
            day_of_year=172,
        )
        # Should be > 0 (diffuse component)
        assert np.all(result > 0)

    def test_higher_svf_more_diffuse(self):
        """Higher SVF gives more diffuse irradiance."""
        N, M = 2, 3
        sun_positions = [(45.0, 0.0), (60.0, 0.0), (45.0, 0.0)]
        sunlit_matrix = np.zeros((N, M), dtype=bool)  # No direct
        facade_azimuths = np.array([0.0, 0.0])
        svf_array = np.array([0.2, 0.8])

        result = compute_facade_daily_irradiance_array(
            sun_positions,
            sunlit_matrix,
            facade_azimuths,
            svf_array,
            interval_minutes=30,
            day_of_year=172,
        )
        assert result[1] > result[0]


# ---------------------------------------------------------------------------
# Section E: Aggregation
# ---------------------------------------------------------------------------


class TestAggregateByBuilding:
    @pytest.fixture
    def sample_facade_gdf(self):
        return gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0)] * 6,
                "building_id": [1, 1, 1, 2, 2, 2],
                "facade_sunlit_hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "facade_irradiance_wh": [100, 200, 300, 400, 500, 600],
                "who_compliant": [False, True, True, True, True, True],
                "floor_level": [0, 0, 1, 0, 1, 2],
                "height_above_ground": [0.5, 1.5, 3.5, 0.5, 3.5, 6.5],
            }
        )

    def test_building_count(self, sample_facade_gdf):
        result = aggregate_by_building(sample_facade_gdf)
        assert len(result) == 2

    def test_compliance_rate(self, sample_facade_gdf):
        result = aggregate_by_building(sample_facade_gdf)
        bldg1 = result[result["building_id"] == 1].iloc[0]
        # 2 out of 3 compliant
        assert bldg1["who_compliance_rate"] == pytest.approx(2 / 3, abs=1e-10)

    def test_sunlit_hours_stats(self, sample_facade_gdf):
        result = aggregate_by_building(sample_facade_gdf)
        bldg2 = result[result["building_id"] == 2].iloc[0]
        assert bldg2["sunlit_hours_mean"] == pytest.approx(5.0)
        assert bldg2["sunlit_hours_min"] == pytest.approx(4.0)
        assert bldg2["sunlit_hours_max"] == pytest.approx(6.0)


class TestAggregateByBuildingFloor:
    def test_floor_grouping(self):
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0)] * 4,
                "building_id": [1, 1, 1, 1],
                "floor_level": [0, 0, 1, 1],
                "facade_sunlit_hours": [1.0, 2.0, 3.0, 4.0],
                "facade_irradiance_wh": [100, 200, 300, 400],
                "who_compliant": [False, True, True, True],
            }
        )
        result = aggregate_by_building_floor(gdf)
        assert len(result) == 2  # 2 floors

        floor0 = result[result["floor_level"] == 0].iloc[0]
        assert floor0["sunlit_hours_mean"] == pytest.approx(1.5)
        assert floor0["who_compliance_rate"] == pytest.approx(0.5)
