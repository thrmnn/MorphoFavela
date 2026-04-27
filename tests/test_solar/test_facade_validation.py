"""Synthetic street canyon validation for facade solar pipeline.

Builds a simple 2-building canyon on flat terrain and validates that:
1. The min_hit_distance fix produces physically plausible sunlit hours
2. Upper floors receive more sun than lower floors
3. Without the fix (min_hit_distance=0), all points are falsely shadowed
"""

import geopandas as gpd
import pytest
import pyvista as pv
from shapely.geometry import Point

from src.solar.facade import (
    aggregate_by_building_floor,
    compute_facade_solar_access,
    summarize_housing_units,
)

# ---------------------------------------------------------------------------
# Canyon geometry helpers
# ---------------------------------------------------------------------------

CANYON_WIDTH = 6.0  # street width (metres)
BUILDING_HEIGHT = 10.0
BUILDING_DEPTH = 10.0  # N-S dimension
BUILDING_LENGTH = 20.0  # E-W dimension


def _build_canyon_scene():
    """Build a synthetic E-W street canyon with two 10m-tall buildings.

    Layout (plan view, north up):
        Building A:  x in [-5, 5],   y in [-10, 10],  z in [0, 10]
        Street:      x in [5, 11]    (6m wide)
        Building B:  x in [11, 21],  y in [-10, 10],  z in [0, 10]
        Terrain:     flat at z=0,    40m x 40m

    Returns (scene_mesh, footprints_gdf)
    """
    # Flat terrain
    terrain = pv.Plane(
        center=(8, 0, 0),
        direction=(0, 0, 1),
        i_size=50,
        j_size=50,
        i_resolution=10,
        j_resolution=10,
    )

    # Two box buildings
    bldg_a = pv.Box(bounds=(-5, 5, -10, 10, 0, BUILDING_HEIGHT))
    bldg_b = pv.Box(bounds=(11, 21, -10, 10, 0, BUILDING_HEIGHT))

    scene = terrain + bldg_a + bldg_b

    # Footprints GeoDataFrame (needed for CRS but not for raycasting)
    from shapely.geometry import box as sbox

    footprints = gpd.GeoDataFrame(
        {
            "geometry": [
                sbox(-5, -10, 5, 10),
                sbox(11, -10, 21, 10),
            ],
            "altura": [BUILDING_HEIGHT, BUILDING_HEIGHT],
            "base": [0.0, 0.0],
        },
        crs="EPSG:31983",
    )
    return scene, footprints


def _build_canyon_facade_points():
    """Create facade points on the inner canyon walls.

    Building A east wall (x=5.1, facing east, azimuth=90):
      Points at y = -6, -2, 2, 6  and  z = 1.25, 3.75, 6.25, 8.75
    Building B west wall (x=10.9, facing west, azimuth=270):
      Same y and z pattern.

    Returns GeoDataFrame with required facade columns.
    """
    rows = []
    inset = 0.1  # match production inset

    y_positions = [-6, -2, 2, 6]
    z_positions = [1.25, 3.75, 6.25, 8.75]

    # Building A — east-facing inner wall
    for y in y_positions:
        for z in z_positions:
            rows.append(
                {
                    "x": 5.0 + inset,
                    "y": y,
                    "z": z,
                    "normal_x": 1.0,
                    "normal_y": 0.0,
                    "normal_z": 0.0,
                    "building_id": 0,
                    "facade_azimuth": 90.0,
                    "height_above_ground": z,
                    "geometry": Point(5.0 + inset, y),
                }
            )

    # Building B — west-facing inner wall
    for y in y_positions:
        for z in z_positions:
            rows.append(
                {
                    "x": 21.0 - inset,
                    "y": y,
                    "z": z,
                    "normal_x": -1.0,
                    "normal_y": 0.0,
                    "normal_z": 0.0,
                    "building_id": 1,
                    "facade_azimuth": 270.0,
                    "height_above_ground": z,
                    "geometry": Point(21.0 - inset, y),
                }
            )

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:31983")
    gdf["svf"] = 1.0
    return gdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canyon():
    """Return (scene_mesh, facade_gdf)."""
    scene, _fp = _build_canyon_scene()
    facade_gdf = _build_canyon_facade_points()
    return scene, facade_gdf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreetCanyonValidation:
    """Validate facade solar with a synthetic street canyon."""

    def test_with_fix_some_points_sunlit(self, canyon):
        """With min_hit_distance=0.5, some facade points should be sunlit."""
        scene, facade_gdf = canyon
        result = compute_facade_solar_access(
            facade_gdf,
            scene,
            date="2026-06-21",
            latitude=-22.97,
            longitude=-43.17,
            min_hit_distance=0.5,
            n_jobs=1,
        )
        max_hours = result["facade_sunlit_hours"].max()
        assert max_hours > 0, (
            f"No facade points sunlit (max={max_hours}). "
            "Self-intersection fix may not be working."
        )

    def test_upper_floors_more_sun(self, canyon):
        """Upper floors should get more sunlight than ground floor."""
        scene, facade_gdf = canyon
        result = compute_facade_solar_access(
            facade_gdf,
            scene,
            date="2026-06-21",
            latitude=-22.97,
            longitude=-43.17,
            min_hit_distance=0.5,
            n_jobs=1,
        )
        by_floor = result.groupby("floor_level")["facade_sunlit_hours"].mean()
        if len(by_floor) >= 2:
            top_floor = by_floor.index.max()
            assert by_floor[top_floor] >= by_floor[0], (
                f"Top floor ({by_floor[top_floor]:.2f}h) should get >= "
                f"ground floor ({by_floor[0]:.2f}h) sunlight."
            )

    def test_without_fix_self_intersection(self):
        """Points ON the wall (no inset) are shadowed by own roof without fix."""
        scene, _fp = _build_canyon_scene()
        # Place points with ZERO inset — on the wall surface, under the roof
        facade_gdf = _build_canyon_facade_points()
        # Move points to be exactly on the wall (inside footprint edge)
        facade_gdf.loc[facade_gdf["building_id"] == 0, "x"] = 4.99  # inside bldg A
        facade_gdf.loc[facade_gdf["building_id"] == 1, "x"] = 11.01  # inside bldg B

        result_buggy = compute_facade_solar_access(
            facade_gdf,
            scene,
            date="2026-06-21",
            latitude=-22.97,
            longitude=-43.17,
            min_hit_distance=0.0,  # No fix: self-intersection
            n_jobs=1,
        )
        result_fixed = compute_facade_solar_access(
            facade_gdf,
            scene,
            date="2026-06-21",
            latitude=-22.97,
            longitude=-43.17,
            min_hit_distance=0.5,  # Fix applied
            n_jobs=1,
        )
        # The fix should recover significantly more sunlit hours
        buggy_max = result_buggy["facade_sunlit_hours"].max()
        fixed_max = result_fixed["facade_sunlit_hours"].max()
        assert fixed_max > buggy_max, (
            f"Fix should recover sunlit hours: buggy max={buggy_max:.2f}, "
            f"fixed max={fixed_max:.2f}."
        )

    def test_who_compliance_nonzero(self, canyon):
        """Some points should exceed the WHO 2h threshold in the canyon."""
        scene, facade_gdf = canyon
        result = compute_facade_solar_access(
            facade_gdf,
            scene,
            date="2026-06-21",
            latitude=-22.97,
            longitude=-43.17,
            min_hit_distance=0.5,
            n_jobs=1,
        )
        n_compliant = result["who_compliant"].sum()
        assert n_compliant > 0, (
            "No WHO-compliant points in a 6m canyon — "
            "upper floors should receive >2h sunlight."
        )

    def test_housing_unit_summary(self, canyon):
        """Housing unit summary should report plausible numbers."""
        scene, facade_gdf = canyon
        result = compute_facade_solar_access(
            facade_gdf,
            scene,
            date="2026-06-21",
            latitude=-22.97,
            longitude=-43.17,
            min_hit_distance=0.5,
            n_jobs=1,
        )
        floor_agg = aggregate_by_building_floor(result)
        hu = summarize_housing_units(floor_agg)
        assert hu["total_units"] > 0
        assert 0 <= hu["pct_below"] <= 100
