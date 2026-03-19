"""Tests for src/svf_v2/scene.py."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from tests.test_svf_v2.conftest import _make_box_building


class TestExtrudePolygon:
    def test_basic_box(self):
        from src.svf_v2.scene import _extrude_polygon

        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        mesh = _extrude_polygon(poly, base_elev=10.0, height=5.0)

        assert mesh is not None
        assert mesh.n_points == 8  # 4 bottom + 4 top
        # Z range should be [10, 15]
        assert mesh.points[:, 2].min() == pytest.approx(10.0)
        assert mesh.points[:, 2].max() == pytest.approx(15.0)

    def test_zero_height_returns_none(self):
        from src.svf_v2.scene import _extrude_polygon

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert _extrude_polygon(poly, 0, 0) is None
        assert _extrude_polygon(poly, 0, -1) is None

    def test_nan_returns_none(self):
        from src.svf_v2.scene import _extrude_polygon

        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert _extrude_polygon(poly, np.nan, 5) is None
        assert _extrude_polygon(poly, 0, np.nan) is None


class TestMakeBoxBuilding:
    def test_world_coords_preserved(self):
        """Building mesh stays in world coordinates."""
        mesh = _make_box_building(100, 200, 110, 210, 50, 20)
        assert mesh.points[:, 0].min() == pytest.approx(100.0)
        assert mesh.points[:, 0].max() == pytest.approx(110.0)
        assert mesh.points[:, 1].min() == pytest.approx(200.0)
        assert mesh.points[:, 1].max() == pytest.approx(210.0)
        assert mesh.points[:, 2].min() == pytest.approx(50.0)
        assert mesh.points[:, 2].max() == pytest.approx(70.0)


class TestCombinedScene:
    def test_single_building_on_terrain(self, single_building_scene):
        """Combined scene contains both terrain and building."""
        mesh = single_building_scene
        # Terrain is 10x10 at z=0, building goes to z=5
        assert mesh.points[:, 2].min() == pytest.approx(0.0)
        assert mesh.points[:, 2].max() == pytest.approx(5.0)
        # Should have vertices from both terrain and building
        assert mesh.n_points > 8  # more than just building
