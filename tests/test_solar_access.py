"""Tests for solar computation (src.solar package)."""

import math

import numpy as np
import pandas as pd
import pyvista as pv

from src.solar.sun import compute_sun_positions, sun_position_to_direction
from src.solar.compute import (
    _build_obb_tree,
    _ray_hits,
    compute_solar_access_streets,
    compute_sunlit_matrix,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic meshes
# ---------------------------------------------------------------------------


def _flat_terrain(size: float = 50.0, n: int = 21) -> pv.PolyData:
    """Flat terrain mesh at z=0."""
    xs = np.linspace(-size, size, n)
    ys = np.linspace(-size, size, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            faces.extend(
                [[3, idx, idx + 1, idx + n], [3, idx + 1, idx + n + 1, idx + n]]
            )
    return pv.PolyData(points, faces)


def _box_mesh(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> pv.PolyData:
    """Axis-aligned closed box mesh."""
    verts = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float32,
    )
    faces = [
        [4, 0, 3, 2, 1],  # bottom
        [4, 4, 5, 6, 7],  # top
        [4, 0, 1, 5, 4],  # front
        [4, 2, 3, 7, 6],  # back
        [4, 0, 4, 7, 3],  # left
        [4, 1, 2, 6, 5],  # right
    ]
    flat = []
    for f in faces:
        flat.extend(f)
    return pv.PolyData(verts, np.array(flat, dtype=np.int32))


def _open_sky_scene() -> pv.PolyData:
    """Flat terrain only -- full sky exposure."""
    return _flat_terrain()


def _enclosed_scene() -> pv.PolyData:
    """Flat terrain + tall enclosing walls on all four sides.

    The walls form a continuous ring so no ray can escape through gaps.
    A point at the centre will be completely shaded for any sun altitude
    below ~80 degrees.
    """
    terrain = _flat_terrain(size=50.0, n=21)
    # Four continuous walls forming a closed rectangle
    wall_h = 100.0
    walls = [
        _box_mesh(-10, 10, 8, 10, 0, wall_h),  # north wall (full width)
        _box_mesh(-10, 10, -10, -8, 0, wall_h),  # south wall (full width)
        _box_mesh(8, 10, -10, 10, 0, wall_h),  # east wall  (full depth)
        _box_mesh(-10, -8, -10, 10, 0, wall_h),  # west wall  (full depth)
    ]
    combined = terrain
    for w in walls:
        combined = combined.merge(w)
    return combined


# ===================================================================
# Tests: compute_sun_positions
# ===================================================================


class TestComputeSunPositions:
    """Test sun position computation."""

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
        """Rio winter solstice: roughly 10-11 hours of daylight.

        With 1-hour intervals we expect ~10-11 positions.
        """
        positions = compute_sun_positions(
            latitude=-22.97,
            longitude=-43.17,
            date="2026-06-21",
            hour_start=5,
            hour_end=19,
            interval_minutes=60,
        )
        # Winter solstice at lat -23: sunrise ~06:30, sunset ~17:15 local
        # Expect roughly 10-11 valid positions
        assert 7 <= len(positions) <= 13, (
            f"Expected 7-13 positions, got {len(positions)}"
        )

    def test_max_altitude_winter_solstice_rio(self):
        """At lat -23 winter solstice, max altitude ~ 43-44 degrees."""
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

    def test_finer_interval_more_positions(self):
        """Smaller interval should yield more positions."""
        pos_60 = compute_sun_positions(interval_minutes=60)
        pos_30 = compute_sun_positions(interval_minutes=30)
        assert len(pos_30) >= len(pos_60)

    def test_equator_summer(self):
        """Equator on equinox: sun nearly overhead, ~12h daylight."""
        positions = compute_sun_positions(
            latitude=0.0,
            longitude=0.0,
            date="2026-03-20",
            hour_start=4,
            hour_end=20,
            interval_minutes=60,
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

    def test_straight_up_any_azimuth(self):
        """altitude=90 with different azimuths still points up."""
        for az in [0, 45, 90, 180, 270]:
            d = sun_position_to_direction(90.0, float(az))
            np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=1e-10)

    def test_horizontal_north(self):
        """altitude=0, azimuth=0 -> north (0, 1, 0)."""
        d = sun_position_to_direction(0.0, 0.0)
        np.testing.assert_allclose(d, [0.0, 1.0, 0.0], atol=1e-10)

    def test_horizontal_east(self):
        """altitude=0, azimuth=90 -> east (1, 0, 0)."""
        d = sun_position_to_direction(0.0, 90.0)
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-10)

    def test_horizontal_south(self):
        """altitude=0, azimuth=180 -> south (0, -1, 0)."""
        d = sun_position_to_direction(0.0, 180.0)
        np.testing.assert_allclose(d, [0.0, -1.0, 0.0], atol=1e-10)

    def test_horizontal_west(self):
        """altitude=0, azimuth=270 -> west (-1, 0, 0)."""
        d = sun_position_to_direction(0.0, 270.0)
        np.testing.assert_allclose(d, [-1.0, 0.0, 0.0], atol=1e-10)

    def test_unit_length(self):
        """Result must always be a unit vector."""
        for alt in [0, 15, 30, 45, 60, 90]:
            for az in [0, 45, 90, 135, 180, 225, 270, 315]:
                d = sun_position_to_direction(float(alt), float(az))
                np.testing.assert_allclose(
                    np.linalg.norm(d),
                    1.0,
                    atol=1e-12,
                    err_msg=f"Non-unit at alt={alt}, az={az}",
                )

    def test_45_degree_altitude(self):
        """altitude=45, azimuth=0 -> z = sin(45) = sqrt(2)/2."""
        d = sun_position_to_direction(45.0, 0.0)
        expected_z = math.sqrt(2) / 2
        assert abs(d[2] - expected_z) < 1e-10


# ===================================================================
# Tests: batch solar access with synthetic scenes
# ===================================================================


class TestSunlitMatrix:
    """Test ray-casting solar access with controlled geometry."""

    @staticmethod
    def _sunlit_counts(observer, sun_dirs, scene):
        """Helper: compute unobstructed sun-position counts per observer."""
        sunlit = compute_sunlit_matrix(observer, sun_dirs, scene, n_jobs=1)
        return sunlit.sum(axis=1).astype(np.int32)

    def test_open_sky_full_access(self):
        """A point on flat terrain with no obstructions gets full sun."""
        scene = _open_sky_scene()
        observer = np.array([[0.0, 0.0, 1.5]])

        sun_dirs = np.array(
            [
                sun_position_to_direction(45.0, 0.0),
                sun_position_to_direction(45.0, 90.0),
                sun_position_to_direction(45.0, 180.0),
                sun_position_to_direction(45.0, 270.0),
                sun_position_to_direction(60.0, 0.0),
            ]
        )

        counts = self._sunlit_counts(observer, sun_dirs, scene)
        assert counts[0] == 5

    def test_enclosed_zero_access(self):
        """A point surrounded by tall walls gets zero sun."""
        scene = _enclosed_scene()
        observer = np.array([[0.0, 0.0, 1.5]])

        sun_dirs = np.array(
            [
                sun_position_to_direction(20.0, 0.0),
                sun_position_to_direction(20.0, 90.0),
                sun_position_to_direction(20.0, 180.0),
                sun_position_to_direction(20.0, 270.0),
                sun_position_to_direction(30.0, 45.0),
                sun_position_to_direction(30.0, 135.0),
                sun_position_to_direction(30.0, 225.0),
                sun_position_to_direction(30.0, 315.0),
            ]
        )

        counts = self._sunlit_counts(observer, sun_dirs, scene)
        assert counts[0] == 0, f"Expected 0 unobstructed, got {counts[0]}"

    def test_partial_obstruction(self):
        """A single wall obstructs some directions but not others."""
        terrain = _flat_terrain(size=50.0, n=11)
        wall = _box_mesh(-5, 5, 3, 5, 0, 20)
        scene = terrain.merge(wall)

        observer = np.array([[0.0, 0.0, 1.5]])

        sun_dirs = np.array(
            [
                sun_position_to_direction(30.0, 0.0),  # north -> blocked
                sun_position_to_direction(30.0, 180.0),  # south -> clear
            ]
        )

        counts = self._sunlit_counts(observer, sun_dirs, scene)
        assert counts[0] == 1

    def test_multiple_points(self):
        """Multiple observer points in open sky should all get full access."""
        scene = _open_sky_scene()
        observers = np.array(
            [
                [0.0, 0.0, 1.5],
                [10.0, 10.0, 1.5],
                [-10.0, -10.0, 1.5],
            ]
        )

        sun_dirs = np.array(
            [
                sun_position_to_direction(45.0, 0.0),
                sun_position_to_direction(45.0, 180.0),
            ]
        )

        counts = self._sunlit_counts(observers, sun_dirs, scene)
        np.testing.assert_array_equal(counts, [2, 2, 2])

    def test_no_sun_positions_zero_counts(self):
        """If there are no sun positions, counts should be zero."""
        scene = _open_sky_scene()
        observer = np.array([[0.0, 0.0, 1.5]])
        sun_dirs = np.empty((0, 3))

        counts = self._sunlit_counts(observer, sun_dirs, scene)
        assert counts[0] == 0


# ===================================================================
# Tests: compute_solar_access_streets
# ===================================================================


class TestComputeSolarAccessStreets:
    """Test street-based solar access function."""

    def test_open_sky_streets(self):
        """Street points in open sky should get full solar hours."""
        import geopandas as gpd
        from shapely.geometry import Point

        scene = _open_sky_scene()

        pts = gpd.GeoDataFrame(
            {
                "z": [0.0, 0.0],
                "z_observer": [1.5, 1.5],
                "street_id": [0, 0],
            },
            geometry=[Point(0, 0), Point(10, 10)],
            crs="EPSG:31983",
        )

        result = compute_solar_access_streets(
            pts,
            scene,
            latitude=-22.97,
            longitude=-43.17,
            date="2026-06-21",
            interval_minutes=60,
        )

        assert "solar_hours" in result.columns
        assert "n_sun_positions" in result.columns
        # Open sky: every sun position should be unobstructed
        n_sun = result["n_sun_positions"].iloc[0]
        # solar_hours = n_unobstructed * (interval / 60)
        for _, row in result.iterrows():
            assert row["solar_hours"] == n_sun * 1.0  # 60 min -> 1h per step

    def test_enclosed_streets(self):
        """Street points inside an enclosure should get zero solar hours."""
        import geopandas as gpd
        from shapely.geometry import Point

        scene = _enclosed_scene()

        pts = gpd.GeoDataFrame(
            {"z_observer": [1.5]},
            geometry=[Point(0, 0)],
            crs="EPSG:31983",
        )

        result = compute_solar_access_streets(
            pts,
            scene,
            latitude=-22.97,
            longitude=-43.17,
            date="2026-06-21",
        )

        assert result["solar_hours"].iloc[0] == 0.0

    def test_preserves_original_columns(self):
        """The function should preserve all original columns."""
        import geopandas as gpd
        from shapely.geometry import Point

        scene = _open_sky_scene()
        pts = gpd.GeoDataFrame(
            {"z_observer": [1.5], "my_col": ["hello"]},
            geometry=[Point(0, 0)],
            crs="EPSG:31983",
        )

        result = compute_solar_access_streets(pts, scene)
        assert "my_col" in result.columns
        assert result["my_col"].iloc[0] == "hello"

    def test_empty_input(self):
        """Empty input should return empty with correct columns."""
        import geopandas as gpd

        scene = _open_sky_scene()
        pts = gpd.GeoDataFrame(
            {"z_observer": pd.Series(dtype=float)},
            geometry=[],
            crs="EPSG:31983",
        )

        result = compute_solar_access_streets(pts, scene)
        assert "solar_hours" in result.columns
        assert len(result) == 0


# ===================================================================
# Tests: OBB tree and ray helpers
# ===================================================================


class TestRayHelpers:
    """Test low-level ray-casting helpers."""

    def test_obb_tree_builds(self):
        """OBB tree should build without error."""
        mesh = _flat_terrain(size=10, n=5)
        obb = _build_obb_tree(mesh)
        assert obb is not None

    def test_ray_hits_ground(self):
        """A downward ray should hit the terrain."""
        mesh = _flat_terrain(size=10, n=5)
        obb = _build_obb_tree(mesh)
        origin = np.array([0.0, 0.0, 5.0])
        endpoint = np.array([0.0, 0.0, -5.0])
        assert _ray_hits(obb, origin, endpoint) is True

    def test_ray_misses_upward(self):
        """An upward ray from above terrain should miss."""
        mesh = _flat_terrain(size=10, n=5)
        obb = _build_obb_tree(mesh)
        origin = np.array([0.0, 0.0, 1.0])
        endpoint = np.array([0.0, 0.0, 100.0])
        assert _ray_hits(obb, origin, endpoint) is False
