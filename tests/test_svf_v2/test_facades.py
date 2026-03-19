"""Tests for src/svf_v2/facades.py."""

import numpy as np
import pyvista as pv
import geopandas as gpd
from shapely.geometry import Point

from src.svf_v2.facades import compute_facade_svf
from tests.test_svf_v2.conftest import _make_box_building


class TestFacadeSVF:
    def test_free_standing_wall_unobstructed(self):
        """
        A point on a free-standing vertical wall facing open sky (no
        obstructions in the forward hemisphere).

        With only a flat terrain behind/below the observer, the forward
        hemisphere is entirely clear.  The ray-caster only tests directions
        with dot(dir, normal) > 0, and none of those hit the terrain
        below, so SVF should be ~1.0.
        """
        # Flat terrain only (no buildings in the scene)
        x = np.linspace(0, 20, 21)
        y = np.linspace(0, 20, 21)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        terrain = pv.StructuredGrid(xx, yy, zz).extract_surface()

        # Facade point at centre, facing north, at z=2.5
        facade_gdf = gpd.GeoDataFrame(
            {
                "x": [10.0],
                "y": [10.0],
                "z": [2.5],
                "normal_x": [0.0],
                "normal_y": [1.0],
                "normal_z": [0.0],
            },
            geometry=[Point(10.0, 10.0)],
        )

        result = compute_facade_svf(facade_gdf, terrain, n_sky_patches=50)
        svf = result["svf"].values[0]

        # No obstructions in forward hemisphere -> SVF ~1.0
        assert svf > 0.9, f"Expected ~1.0 for unobstructed facade, got {svf}"

    def test_enclosed_courtyard_low_svf(self):
        """Point inside a courtyard surrounded by tall walls -> low SVF."""
        # Build 4 walls around a central courtyard
        x = np.linspace(0, 20, 21)
        y = np.linspace(0, 20, 21)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        terrain = pv.StructuredGrid(xx, yy, zz).extract_surface()

        walls = [
            _make_box_building(4, 4, 6, 16, 0, 15),   # west wall
            _make_box_building(14, 4, 16, 16, 0, 15),  # east wall
            _make_box_building(4, 4, 16, 6, 0, 15),    # south wall
            _make_box_building(4, 14, 16, 16, 0, 15),  # north wall
        ]
        scene = terrain
        for w in walls:
            scene = scene + w

        # Point in centre of courtyard, facing east
        facade_gdf = gpd.GeoDataFrame(
            {
                "x": [10.0],
                "y": [10.0],
                "z": [1.0],
                "normal_x": [1.0],
                "normal_y": [0.0],
                "normal_z": [0.0],
            },
            geometry=[Point(10.0, 10.0)],
        )

        result = compute_facade_svf(facade_gdf, scene, n_sky_patches=50)
        svf = result["svf"].values[0]

        # Enclosed courtyard: most sky is blocked
        assert svf < 0.5, f"Expected low SVF in courtyard, got {svf}"
