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


class TestTopoCorruptionFilter:
    """Rows with topo == 0 are a Rio edificações registry corruption where
    altura was mis-derived as the base elevation; they must be dropped before
    extrusion so they don't become phantom towers."""

    def _write_inputs(self, tmp_path):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin

        # Flat DTM at z=10, EPSG:31983, 1 m pixels, 40x40 m around the buildings
        arr = np.full((40, 40), 10.0, dtype="float32")
        dtm = tmp_path / "dtm.tif"
        with rasterio.open(
            dtm, "w", driver="GTiff", height=40, width=40, count=1,
            dtype="float32", crs="EPSG:31983",
            transform=from_origin(0, 40, 1, 1),
        ) as dst:
            dst.write(arr, 1)

        valid = Polygon([(5, 5), (9, 5), (9, 9), (5, 9)])
        corrupt = Polygon([(20, 20), (24, 20), (24, 24), (20, 24)])
        gdf = gpd.GeoDataFrame(
            {
                "base": [10.0, 232.0],
                "altura": [8.0, 232.0],   # corrupt: altura == base, topo == 0
                "topo": [18.0, 0.0],
                "geometry": [valid, corrupt],
            },
            crs="EPSG:31983",
        )
        fp = tmp_path / "buildings.gpkg"
        gdf.to_file(fp, driver="GPKG")
        return dtm, fp

    def test_corrupt_row_dropped(self, tmp_path):
        from src.svf_v2.scene import build_building_meshes

        dtm, fp = self._write_inputs(tmp_path)
        mesh, gdf = build_building_meshes(fp, dtm, area=None)
        # Only the valid building survives
        assert len(gdf) == 1
        assert (gdf["topo"] != 0).all()
        # No phantom tower: z stays near the valid 10 m base + 8 m height
        assert mesh is not None
        assert mesh.points[:, 2].max() < 50, "phantom tower not filtered"

    def test_no_topo_column_is_noop(self, tmp_path):
        import geopandas as gpd

        from src.svf_v2.scene import build_building_meshes

        dtm, fp = self._write_inputs(tmp_path)
        gdf = gpd.read_file(fp).drop(columns=["topo"])
        fp2 = tmp_path / "buildings_no_topo.gpkg"
        gdf.to_file(fp2, driver="GPKG")
        # Without a topo column the corruption gate cannot fire; both rows kept.
        _, out = build_building_meshes(fp2, dtm, area=None)
        assert len(out) == 2
